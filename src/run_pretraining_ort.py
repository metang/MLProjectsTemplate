# coding=utf-8
# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# ==================
import csv
import os
import sys
import time
import argparse
import random
import h5py
from tqdm import tqdm, trange
import os
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset
import math
import multiprocessing
import modeling
import shutil

from utils import format_step
from datetime import datetime

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
DIRECTORY_TO_WATCH = "/usr/share"

from multiprocessing import Value
from ctypes import c_bool

class PreemptHandler(FileSystemEventHandler):
    def __init__(self):
        super(PreemptHandler, self).__init__()
        self.is_preempted = Value(c_bool, False)

    def on_any_event(self, event):
        if not event.is_directory and event.src_path.endswith("/to-be-preempted"):
            print(datetime.utcnow(),"Detected Preempt Signal, should stop and return.")
            self.is_preempted.value = True

class PreemptDetector:
    def __init__(self):
        self.observer = Observer()
        self.event_handler = PreemptHandler()

    def run(self):
        self.observer.schedule(self.event_handler, DIRECTORY_TO_WATCH, recursive=False)
        self.observer.start()

    def is_preempted(self):
        return self.event_handler.is_preempted.value

    def stop(self):
        self.observer.stop()

# replace routine from utils.py as we dont use torch.distributed
def is_main_process(args):
    if hasattr(args, 'world_rank'):
        return args.world_rank == 0
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True

import dllogger
from concurrent.futures import ProcessPoolExecutor

torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)

skipped_steps = 0

import onnx

#Workaround because python functions are not picklable
class WorkerInitObj(object):
    def __init__(self, seed):
        self.seed = seed
    def __call__(self, id):
        np.random.seed(seed=self.seed + id)
        random.seed(self.seed + id)

def create_pretraining_dataset(input_file, max_pred_length, shared_list, args, worker_init):
    train_data = pretraining_dataset(input_file=input_file, max_pred_length=max_pred_length)
    train_sampler = RandomSampler(train_data)
    # --- ort training edit: we need to skip last batch when hard coding inputs as an optimization
    train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                  batch_size=args.train_batch_size * args.n_gpu, 
                                  num_workers=4, worker_init_fn=worker_init,
                                  pin_memory=True, drop_last=True)
    # ---
    return train_dataloader, input_file

class pretraining_dataset(Dataset):

    def __init__(self, input_file, max_pred_length):
        self.input_file = input_file
        self.max_pred_length = max_pred_length
        f = h5py.File(input_file, "r")
        keys = ['input_ids', 'input_mask', 'segment_ids', 'masked_lm_positions', 'masked_lm_ids',
                'next_sentence_labels']
        self.inputs = [np.asarray(f[key][:]) for key in keys]
        f.close()

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.inputs[0])

    def __getitem__(self, index):

        [input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids, next_sentence_labels] = [
            torch.from_numpy(input[index].astype(np.int64)) if indice < 5 else torch.from_numpy(
                np.asarray(input[index].astype(np.int64))) for indice, input in enumerate(self.inputs)]

        masked_lm_labels = torch.ones(input_ids.shape, dtype=torch.long) * -1
        index = self.max_pred_length
        # store number of  masked tokens in index
        padded_mask_indices = (masked_lm_positions == 0).nonzero()
        if len(padded_mask_indices) != 0:
            index = padded_mask_indices[0].item()
        masked_lm_labels[masked_lm_positions[:index]] = masked_lm_ids[:index]

        return [input_ids, segment_ids, input_mask,
                masked_lm_labels, next_sentence_labels]
class BertPretrainingCriterion(torch.nn.Module):
    def __init__(self, vocab_size, batch_size, seq_length):
        super(BertPretrainingCriterion, self).__init__()
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocab_size = vocab_size
    def forward(self, prediction_scores, seq_relationship_score, masked_lm_labels, next_sentence_labels):
        print("BertPretrainingCriterion: batch_size: ",self.batch_size, ", self.seq_length:", self.seq_length )
        masked_lm_loss = self.loss_fn(prediction_scores.view([self.batch_size * self.seq_length, self.vocab_size]), masked_lm_labels.view(self.batch_size * self.seq_length))
        next_sentence_loss = self.loss_fn(seq_relationship_score, next_sentence_labels.view(self.batch_size))
        total_loss = masked_lm_loss + next_sentence_loss
        return total_loss

# we manually add the loss function into the bert model
# currently ort front end support for this assumes a single tensor input for labels
class bert_model_with_loss(torch.nn.Module):
    def __init__(self, model, loss_fn):
        super(bert_model_with_loss, self).__init__()
        self.model_ = model
        self.loss_fn_ = loss_fn

    def forward(self, input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels):
        preds_score, seq_relation_score = self.model_(input_ids, segment_ids, input_mask)
        return self.loss_fn_(preds_score, seq_relation_score, masked_lm_labels, next_sentence_labels)

def parse_arguments():

    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--input_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain .hdf5 files  for the task.")

    parser.add_argument("--config_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The BERT model config")

    parser.add_argument("--bert_model", default="bert-large-uncased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")

    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--init_checkpoint",
                        default=None,
                        type=str,
                        help="The initial checkpoint to start training from.")

    parser.add_argument("--max_seq_length",
                        default=512,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--max_predictions_per_seq",
                        default=80,
                        type=int,
                        help="The maximum total of masked tokens in input sequence")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Per GPU batch size for training.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps",
                        default=1000,
                        type=float,
                        help="Total number of training steps to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.01,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0.0,
                        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')
    parser.add_argument('--log_freq',
                        type=float, default=1.0,
                        help='frequency of logging loss.')
    parser.add_argument('--checkpoint_activations',
                        default=False,
                        action='store_true',
                        help="Whether to use gradient checkpointing")
    parser.add_argument("--resume_from_checkpoint",
                        default=True,
                        action='store_true',
                        help="Whether to resume training from checkpoint.")
    parser.add_argument('--resume_step',
                        type=int,
                        default=-1,
                        help="Step to resume training from.")
    parser.add_argument('--num_steps_per_checkpoint',
                        type=int,
                        default=100,
                        help="Number of update steps until a model checkpoint is saved to disk.")
    parser.add_argument('--skip_checkpoint',
                        default=False,
                        action='store_true',
                        help="Whether to save checkpoints")
    parser.add_argument('--phase2',
                        default=False,
                        action='store_true',
                        help="Whether to train with seq len 512")
    parser.add_argument('--allreduce_post_accumulation',
                        default=False,
                        action='store_true',
                        help="Whether to do allreduces during gradient accumulation steps.")
    parser.add_argument('--allreduce_post_accumulation_fp16',
                        default=False,
                        action='store_true',
                        help="Whether to do fp16 allreduce post accumulation.")
    parser.add_argument('--phase1_end_step',
                        type=int,
                        default=7038,
                        help="Number of training steps in Phase1 - seq len 128")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument('--json-summary', type=str, default="dllogger.json",
                        help='If provided, the json summary will be written to'
                             'the specified file.')
    parser.add_argument("--use_env",
                        action='store_true',
                        help="Whether to read local rank from ENVVAR")
    parser.add_argument('--disable_progress_bar',
                        default=False,
                        action='store_true',
                        help='Disable tqdm progress bar')
    parser.add_argument('--use_ib',
                        default=False,
                        action='store_true',
                        help="Whether to use infiniband on Azure ML submission.")
    parser.add_argument('--partition_optimizer',
                        default=False,
                        action='store_true',
                        help="Whether ORT will partition optimizer.")
    parser.add_argument("--gpu_memory_limit_gb",
                        type=int,
                        default=32,
                        help="GPU memory limit in GBs")
    parser.add_argument('--schedule',
                        default='warmup_poly',
                        type=str)
    parser.add_argument('--tensorboard_dir',
                        default='./outputs',
                        type=str)
    args = parser.parse_args()
    
    return args

def setup_training(args):

    assert (torch.cuda.is_available())

    global ort_supplement
    import ort_supplement.ort_supplement as ort_supplement
    device = ort_supplement.setup_onnxruntime_with_mpi(args)
        
    if is_main_process(args):
        dllogger.init(backends=[dllogger.JSONStreamBackend(verbosity=dllogger.Verbosity.VERBOSE,
                                                           filename=args.json_summary),
                                dllogger.StdOutBackend(verbosity=dllogger.Verbosity.VERBOSE, step_format=format_step)])
    else:
        dllogger.init(backends=[])

    print("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}, world size: {}".format(
        device, args.n_gpu, bool(args.local_rank != -1), args.fp16, args.world_size))

    global_batch_size = 65536
    if args.phase2:
        global_batch_size = 32768

    args.gradient_accumulation_steps = int(round(global_batch_size / args.world_size / args.train_batch_size))
    print("real global batch size is ", args.train_batch_size * args.world_size * args.gradient_accumulation_steps,
        ", gradient_accumulation_steps: ", args.gradient_accumulation_steps)

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    if not args.do_train:
        raise ValueError(" `do_train`  must be True.")

    if not args.resume_from_checkpoint and os.path.exists(args.output_dir) and (
            os.listdir(args.output_dir) and any([i.startswith('ckpt') for i in os.listdir(args.output_dir)])):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))

    if (not args.resume_from_checkpoint or not os.path.exists(args.output_dir)) and is_main_process(args):
        os.makedirs(args.output_dir, exist_ok=True)

    return device, args

def prepare_model(args, device):

    # Prepare model
    config = modeling.BertConfig.from_json_file(args.config_file)

    # Padding for divisibility by 8
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)

    model = modeling.BertForPreTraining(config)
    criterion = BertPretrainingCriterion(config.vocab_size, args.train_batch_size, args.max_seq_length)

    model.enable_apex(False)
    model = bert_model_with_loss(model, criterion)
    model = ort_supplement.create_ort_trainer(args, device, model)

    checkpoint = None
    if not args.resume_from_checkpoint or os.path.exists(args.output_dir) == False or len(os.listdir(args.output_dir)) == 0:
        global_step = 0
    else:
        if args.resume_step == -1 and not args.init_checkpoint:
            model_names = [f for f in os.listdir(args.output_dir) if f.endswith(".pt")]
            args.resume_step = max([int(x.split('.pt')[0].split('_')[1].strip()) for x in model_names])

        global_step = args.resume_step if not args.init_checkpoint else 0
        print("resume global_step: ", global_step)

        if not args.init_checkpoint:
            checkpoint = torch.load(os.path.join(args.output_dir, "ckpt_{}.pt".format(global_step)), map_location="cpu")
        else:
            checkpoint = torch.load(args.init_checkpoint, map_location="cpu")
        print("after load checkpoint: ", global_step)
        model.load_state_dict(checkpoint['model'], strict=False)

        print("after load checkpoint 2: ", global_step)
        if args.phase2 and not args.init_checkpoint:
            global_step -= args.phase1_end_step
        if is_main_process(args):
            print("resume step from ", args.resume_step)

    return model, checkpoint, global_step
    
def main():

    args = parse_arguments()

    if args.use_env and 'LOCAL_RANK' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])
        
    random.seed(args.seed + args.local_rank)
    np.random.seed(args.seed + args.local_rank)
    torch.manual_seed(args.seed + args.local_rank)
    torch.cuda.manual_seed(args.seed + args.local_rank)
    worker_init = WorkerInitObj(args.seed + args.local_rank)

    device, args = setup_training(args)
    dllogger.log(step="PARAMETER", data={"Config": [str(args)]})

    # Prepare optimizer
    model, checkpoint, global_step = prepare_model(args, device)

    if is_main_process(args):
        dllogger.log(step="PARAMETER", data={"SEED": args.seed})
        writer = SummaryWriter(log_dir=args.tensorboard_dir)

    raw_train_start = time.time()
    if args.do_train:
        if is_main_process(args):
            dllogger.log(step="PARAMETER", data={"train_start": True})
            dllogger.log(step="PARAMETER", data={"batch_size_per_gpu": args.train_batch_size})
            dllogger.log(step="PARAMETER", data={"learning_rate": args.learning_rate})

        model.train()
        most_recent_ckpts_paths = []
        average_loss = 0.0  # averaged loss every args.log_freq steps
        epoch = 0
        training_steps = 0

        pool = ProcessPoolExecutor(1)

        detector = PreemptDetector()
        detector.run()

        # Note: We loop infinitely over epochs, termination is handled via iteration count
        while True:
            thread = None
            if not args.resume_from_checkpoint or os.path.exists(args.output_dir) == False or len(os.listdir(args.output_dir)) == 0 or epoch > 0 or (args.phase2 and global_step < 1) or args.init_checkpoint:
                files = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if
                         os.path.isfile(os.path.join(args.input_dir, f)) and 'training' in f]
                files.sort()
                num_files = len(files)
                random.shuffle(files)
                #f_start_id = 0
                rank_0_file_id = 0
            else:
                #f_start_id = checkpoint['files'][0]
                files = checkpoint['files'][1:]
                rank_0_file_id = checkpoint['rank_0_file_id']
                args.resume_from_checkpoint = False
                num_files = len(files)

            print("rank_0_file_id is ", rank_0_file_id)

            shared_file_list = {}

            if torch.distributed.is_initialized():
                world_size = torch.distributed.get_world_size()
                world_rank = torch.distributed.get_rank()
                print("torch.distributed.is_initialized world_size: ", world_size)
            elif hasattr(args, 'world_size'):
                world_size = args.world_size
                world_rank = args.world_rank
            else:
                world_size = 1
                world_rank = 0

            if world_size > num_files:
                remainder = world_size % num_files
                #data_file = files[(f_start_id*world_size + world_rank + remainder*f_start_id)%num_files]
                data_file = files[(rank_0_file_id + world_rank)%num_files]
            elif world_size > 1:
                #data_file = files[(f_start_id*world_size + world_rank)%num_files]
                data_file = files[(rank_0_file_id + world_rank)%num_files]
            else:
                #data_file = files[f_start_id % num_files]
                data_file = files[rank_0_file_id % num_files]
            # ---

            previous_file = data_file

            train_data = pretraining_dataset(data_file, args.max_predictions_per_seq)
            train_sampler = RandomSampler(train_data)
            # we need to skip last batch when we hard code inputs as an optimization
            train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                          batch_size=args.train_batch_size * args.n_gpu,
                                          num_workers=4, worker_init_fn=worker_init,
                                          pin_memory=True, drop_last=True)


            #if len(files) == 1:
            #    f_start_id = -1
            rank0_f_id = rank_0_file_id 
            while rank0_f_id < len(files):
                next_rank_0_f_id = rank0_f_id + world_size

                need_load_next = next_rank_0_f_id < len(files)
                if need_load_next:
                    if world_size > num_files:
                        data_file = files[(next_rank_0_f_id + world_rank)%num_files]
                    elif world_size > 1:
                        data_file = files[(next_rank_0_f_id + world_rank)%num_files]
                        print("current worker use file id ", (next_rank_0_f_id + world_rank)%num_files)
                    else:
                        data_file = files[next_rank_0_f_id % num_files]

                    previous_file = data_file

                    dataset_future = pool.submit(create_pretraining_dataset, data_file, args.max_predictions_per_seq, shared_file_list, args, worker_init)

                train_iter = tqdm(train_dataloader, desc="Iteration", disable=args.disable_progress_bar) if is_main_process(args) else train_dataloader
                prev_step_time = time.time()
                for step, batch in enumerate(train_iter):

                    training_steps += 1
                    batch = [t.to(device) for t in batch]
                    input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels = batch
                    divisor = args.gradient_accumulation_steps

                    loss, global_step = ort_supplement.run_ort_training_step(args, global_step, training_steps, model, batch)
                    average_loss += loss.item()

                    if global_step >= args.max_steps:
                        train_time_raw = time.time() - raw_train_start
                        last_num_steps = int(training_steps / args.gradient_accumulation_steps) % args.log_freq
                        last_num_steps = args.log_freq if last_num_steps == 0 else last_num_steps
                        average_loss = torch.tensor(average_loss, dtype=torch.float32).cuda()
                        average_loss = average_loss / (last_num_steps * divisor)
                        if (torch.distributed.is_initialized()):
                            average_loss /= torch.distributed.get_world_size()
                            torch.distributed.all_reduce(average_loss)
                        final_loss = average_loss.item()
                        if is_main_process(args):
                            dllogger.log(step=(epoch, global_step, ), data={"final_loss": final_loss})
                    elif training_steps % (args.log_freq * args.gradient_accumulation_steps) == 0:
                        throughput =  (args.train_batch_size * args.gradient_accumulation_steps) / (time.time() - prev_step_time)
                        print('throughput = ', throughput ,'seq/sec')
                        prev_step_time = time.time()
                        sys.stdout.flush()

                        if is_main_process(args):
                            data = {"average_loss": average_loss / (args.log_freq * divisor),
                                    "step_loss": loss.item() * args.gradient_accumulation_steps / divisor}
                            dllogger.log(step=(epoch, global_step,), data=data)
                            writer.add_scalar('train/summary/scalar/total_loss', average_loss / (args.log_freq * divisor),
                                    global_step + args.phase1_end_step if args.resume_step >= 0 and args.phase2 else global_step)
                            writer.add_scalar('train/summary/scalar/world_size', world_size,
                                        global_step + args.phase1_end_step if args.resume_step >= 0 and args.phase2 else global_step)
                            writer.add_scalar('train/summary/scalar/throughput', throughput,
                                        global_step + args.phase1_end_step if args.resume_step >= 0 and args.phase2 else global_step)
                            writer.flush()

                        average_loss = 0

                    if is_main_process(args):
                        print(datetime.utcnow(), "training loop is_preempted: ", detector.is_preempted())
                    if global_step >= args.max_steps or training_steps % (
                            args.num_steps_per_checkpoint*args.gradient_accumulation_steps) == 0 or detector.is_preempted():
                        if detector.is_preempted() and is_main_process(args):
                            print(datetime.utcnow(), "is_preempted: ", detector.is_preempted(), " global_step: ", global_step)

                        if is_main_process(args) and not args.skip_checkpoint:
                            # Save a trained model
                            dllogger.log(step="PARAMETER", data={"checkpoint_step": global_step})
                            model_to_save = model.module if hasattr(model,
                                                                    'module') else model  # Only save the model it-self
                            if args.resume_step < 0 or not args.phase2:
                                # output_save_file = os.path.join(args.output_dir, "ckpt_{}.pt".format(global_step))
                                file_name = "ckpt_{}.pt".format(global_step)
                            else:
                                # output_save_file = os.path.join(args.output_dir, "ckpt_{}.pt".format(global_step + args.phase1_end_step))
                                file_name = "ckpt_{}.pt".format(global_step + args.phase1_end_step)
                            if args.do_train:
                                output_save_file_tmp = os.path.join('/tmp/', file_name)
                                output_save_file_blob = os.path.join(args.output_dir, file_name)

                                if detector.is_preempted():
                                    to_save = rank0_f_id
                                else:
                                    if need_load_next:
                                        to_save = next_rank_0_f_id
                                    else:
                                        to_save = 0

                                state = {'model': model_to_save.state_dict(),
                                         'files': [-1] + files,
                                         'rank_0_file_id': to_save}
                                # torch.save(state, output_save_file)

                                start_time = time.time()
                                torch.save(state, output_save_file_tmp)
                                elapsed_time = time.time() - start_time
                                print("save checkpoint time on local /tmp ", elapsed_time, ", rank0_f_id:", to_save)
                                start_time = time.time()
                                shutil.copyfile(output_save_file_tmp, output_save_file_blob)
                                elapsed_time = time.time() - start_time
                                print("save checkpoint time (copy to blob) ", elapsed_time)
                                print("output_save_file_blob: ", output_save_file_blob)
                                most_recent_ckpts_paths.append(output_save_file_blob)


                                # most_recent_ckpts_paths.append(output_save_file)
                                if len(most_recent_ckpts_paths) > 3:
                                    ckpt_to_be_removed = most_recent_ckpts_paths.pop(0)
                                    os.remove(ckpt_to_be_removed)

                        if global_step >= args.max_steps or detector.is_preempted():
                            if is_main_process(args):
                                if detector.is_preempted():
                                    print(datetime.utcnow(), "is_preempted: ", detector.is_preempted(), " save onnx model")
                                print('-----------------------save onnx model-----------------------')
                                if not args.phase2:
                                    # model_to_save.save_as_onnx('{}/phase1_bert.onnx'.format(args.output_dir))
                                    start_time = time.time()
                                    model_to_save.save_as_onnx('/tmp/phase1_bert.onnx')
                                    elapsed_time = time.time() - start_time
                                    print("save onnx model on local /tmp", elapsed_time, " global_step ", global_step)
                                    start_time = time.time()
                                    shutil.copyfile('/tmp/phase1_bert.onnx', '{}/phase1_bert.onnx'.format(args.output_dir))
                                    elapsed_time = time.time() - start_time
                                    print("save onnx model time (copy to blob) ", elapsed_time, " global_step ", global_step)
                                else:
                                    # model_to_save.save_as_onnx('{}/final_bert.onnx'.format(args.output_dir))
                                    start_time = time.time()
                                    model_to_save.save_as_onnx('/tmp/final_bert.onnx')
                                    elapsed_time = time.time() - start_time
                                    print("save onnx model on local /tmp", elapsed_time, " global_step ", global_step)
                                    start_time = time.time()
                                    shutil.copyfile('/tmp/final_bert.onnx', '{}/final_bert.onnx'.format(args.output_dir))
                                    elapsed_time = time.time() - start_time
                                    print("save onnx model time (copy to blob) ", elapsed_time, " global_step ", global_step)
                            del train_dataloader
                            # thread.join()
                            if detector.is_preempted():
                                print("exit training main function after preemption")
                                return args, 0, time.time() - raw_train_start
                            return args, final_loss, train_time_raw

                del train_dataloader
                # thread.join()
                # Make sure pool has finished and switch train_dataloader
                # NOTE: Will block until complete
                if need_load_next:
                    train_dataloader, data_file = dataset_future.result(timeout=None)
                rank0_f_id = next_rank_0_f_id
            epoch += 1
    detector.stop()
    writer.close()


if __name__ == "__main__":
    print("======================in run_pretraining_ort.py==================")
    now = time.time()
    args, final_loss, train_time_raw = main()
    gpu_count = args.n_gpu
    args.max_steps += args.phase1_end_step if (args.phase2 and args.resume_step > 0) else 0
    if args.resume_step == -1:
        args.resume_step = 0
    if torch.distributed.is_initialized():
        gpu_count = torch.distributed.get_world_size()
    if is_main_process(args):
        e2e_time = time.time() - now
        training_perf = args.train_batch_size * args.gradient_accumulation_steps * gpu_count\
                        * (args.max_steps - args.resume_step + skipped_steps) / train_time_raw
        dllogger.log(step=tuple(), data={"e2e_train_time": e2e_time, "training_sequences_per_second": training_perf,
                                         "final_loss": final_loss, "raw_train_time": train_time_raw })
    print("exit training program")
    dllogger.flush()

