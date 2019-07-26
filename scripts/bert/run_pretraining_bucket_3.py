"""
Pre-training Bidirectional Encoder Representations from Transformers
=========================================================================================
This example shows how to pre-train a BERT model with Gluon NLP Toolkit.
@article{devlin2018bert,
  title={BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding},
  author={Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
  journal={arXiv preprint arXiv:1810.04805},
  year={2018}
}
"""

# coding: utf-8

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint:disable=redefined-outer-name,logging-format-interpolation

import os
import logging
import time
import math

import mxnet as mx
import gluonnlp as nlp

from utils import profile
from fp16_utils import FP16Trainer
from pretraining_utils import get_model_loss, get_pretrain_data_npz, get_dummy_dataloader
from pretraining_utils import log, evaluate, forward, split_and_load, get_argparser
from pretraining_utils import save_parameters, save_states

import numpy as np

# arg parser
parser = get_argparser()
parser.add_argument('--gpus', type=str, default='0', help='List of GPUs to use. e.g. 1,3')
parser.add_argument('--bucket_round_len', type=int, default=16, help='round length of padding')
parser.add_argument('--bucket_min_len', type=int, default=-1, help='min length of sequences, for bucketing')  
args = parser.parse_args()

os.environ['MXNET_KVSTORE_USETREE'] = '1'
os.environ['MXNET_GPU_MEM_POOL_TYPE'] = 'Round'

# logging
level = logging.DEBUG if args.verbose else logging.INFO
logging.getLogger().setLevel(level)
logging.info(args)

class ParallelBERT(nlp.utils.Parallelizable):
    """Data parallel BERT model.

    Parameters
    ----------
    model : Block
        The BERT model.
    """
    def __init__(self, model, mlm_loss, nsp_loss, vocab_size, rescale_factor, trainer=None):
        self._model = model
        self._mlm_loss = mlm_loss
        self._nsp_loss = nsp_loss
        self._vocab_size = vocab_size
        self._rescale_factor = rescale_factor
        self._trainer = trainer

    def forward_backward(self, x):
        """forward backward implementation"""
        with mx.autograd.record():
            (ls, next_sentence_label, classified, masked_id, decoded, \
             masked_weight, ls1, ls2, valid_length) = forward(x, self._model, self._mlm_loss,
                                                              self._nsp_loss, self._vocab_size,
                                                              args.dtype)
            ls = ls / self._rescale_factor
        if args.dtype == 'float16':
            self._trainer.backward(ls)
        else:
            ls.backward()
        return ls, next_sentence_label, classified, masked_id, decoded, \
               masked_weight, ls1, ls2, valid_length

def evaluate(dataloader, num_gpus, parallel, bucket_drop_iterations):
    batch_num = 0
    iter_num = 0
    latency_list = []
    gap_list = []
    for _, data_batch in enumerate(dataloader):
        if args.use_avg_len:
            data_list = [[seq.as_in_context(context) for seq in shard]
                            for context, shard in zip(ctx, data_batch)]
        else:
            if data_batch[0].shape[0] < len(ctx):
                continue
            data_list = split_and_load(data_batch, ctx)

        if batch_num == 0:
            # initialize bucket info
            bucket_batch_sizes = dataloader._batch_sampler._bucket_batch_sizes
            bucket_keys = dataloader._batch_sampler._bucket_keys
            bucket_batch_sizes_array = np.array(bucket_batch_sizes, dtype='float32')
            bucket_keys_array = np.array(bucket_keys, dtype='float32')

            # benchmark the ideal case
            max_bucket_key = max(bucket_keys)
            max_bucket_batch_size = bucket_batch_sizes[np.argmax(bucket_keys_array)]
            assert max_bucket_key == bucket_keys[-1]
            assert max_bucket_batch_size == bucket_batch_sizes[-1]

        bucket_idx = np.argmax(bucket_keys_array >= data_list[0][0].shape[1])
        if iter_num > bucket_drop_iterations and \
            data_list[0][0].shape[0] != bucket_batch_sizes[bucket_idx]:
            # ignore abnormal samples
            continue

        mx.nd.waitall()
        batch_begin_time = time.time()

        # parallel forward / backward
        for data in data_list:
            parallel.put(data)
        for _ in range(len(ctx)):
            (_, next_sentence_label, classified, masked_id,
                decoded, masked_weight, ls1, ls2, valid_length) = parallel.get()

        mx.nd.waitall()

        iter_num += 1
        if iter_num <= bucket_drop_iterations:
            continue

        latency = (time.time()-batch_begin_time) * 1000
        latency_list.append(latency)

        if iter_num % num_gpus == num_gpus - 1:
            latency_array = np.array(latency_list)
            min_latency = np.asscalar(np.min(latency_array))
            max_latency = np.asscalar(np.max(latency_array))
            gap = max_latency - min_latency
            gap_list.append(gap)
            gap_array = np.array(gap_list)
            latency_list = []
            # logging.info("iter_num={}, gap={}, avg={}, std={}, min={}, max={}".format(iter_num, gap, np.asscalar(np.mean(gap_array)), np.asscalar(np.std(gap_array)), np.asscalar(np.min(gap_array)), np.asscalar(np.max(gap_array))))
        batch_num += 1
    return gap_array

def train(data_train, model, nsp_loss, mlm_loss, vocab_size, ctx, store):
    """Training function."""
    mlm_metric = nlp.metric.MaskedAccuracy()
    nsp_metric = nlp.metric.MaskedAccuracy()
    mlm_metric.reset()
    nsp_metric.reset()

    lr = args.lr
    optim_params = {'learning_rate': lr, 'epsilon': 1e-6, 'wd': 0.01}
    if args.dtype == 'float16':
        optim_params['multi_precision'] = True

    trainer = mx.gluon.Trainer(model.collect_params(), 'bertadam', optim_params,
                               update_on_kvstore=False, kvstore=store)
    dynamic_loss_scale = args.dtype == 'float16'
    fp16_trainer = FP16Trainer(trainer, dynamic_loss_scale=dynamic_loss_scale)

    if args.start_step:
        state_path = os.path.join(args.ckpt_dir, '%07d.states.%02d'%(args.start_step, 0))
        logging.info('Loading trainer state from %s', state_path)
        nlp.utils.load_states(trainer, state_path)

    accumulate = args.accumulate
    num_train_steps = args.num_steps
    warmup_ratio = args.warmup_ratio
    num_warmup_steps = int(num_train_steps * warmup_ratio)
    params = [p for p in model.collect_params().values() if p.grad_req != 'null']
    param_dict = model.collect_params()

    # Do not apply weight decay on LayerNorm and bias terms
    for _, v in model.collect_params('.*beta|.*gamma|.*bias').items():
        v.wd_mult = 0.0
    if accumulate > 1:
        for p in params:
            p.grad_req = 'add'

    train_begin_time = time.time()
    begin_time = time.time()
    running_mlm_loss = running_nsp_loss = running_num_tks = 0
    batch_num = 0
    step_num = args.start_step

    parallel_model = ParallelBERT(model, mlm_loss, nsp_loss, vocab_size,
                                  store.num_workers * accumulate, trainer=fp16_trainer)
    num_ctxes = len(ctx)
    parallel = nlp.utils.Parallel(num_ctxes if num_ctxes > 1 else 0, parallel_model)

    # the first several iterations are not accurate
    bucket_drop_iterations = 20

    batch_num = 0
    for _, dataloader in enumerate(data_train):
        # create dummy data loader if needed
        if args.dummy_data_len:
            target_shape = (args.batch_size*num_ctxes, args.dummy_data_len)
            dataloader = get_dummy_dataloader(dataloader, target_shape)

        for _, data_batch in enumerate(dataloader):
            if args.use_avg_len:
                data_list = [[seq.as_in_context(context) for seq in shard]
                                for context, shard in zip(ctx, data_batch)]
            else:
                if data_batch[0].shape[0] < len(ctx):
                    continue
                data_list = split_and_load(data_batch, ctx)

            if batch_num == 0:
                # initialize bucket info
                bucket_batch_sizes = dataloader._batch_sampler._bucket_batch_sizes
                bucket_keys = dataloader._batch_sampler._bucket_keys
                bucket_batch_sizes_array = np.array(bucket_batch_sizes, dtype='float32')
                bucket_keys_array = np.array(bucket_keys, dtype='float32')

                # benchmark the ideal case
                max_bucket_key = max(bucket_keys)
                max_bucket_batch_size = bucket_batch_sizes[np.argmax(bucket_keys_array)]
                assert max_bucket_key == bucket_keys[-1]
                assert max_bucket_batch_size == bucket_batch_sizes[-1]

                benchmark_latency_list = [[] for _ in bucket_batch_sizes]

            mx.nd.waitall()
            batch_begin_time = time.time()

            # parallel forward / backward
            for data in data_list:
                parallel.put(data)
            for _ in range(len(ctx)):
                (_, next_sentence_label, classified, masked_id,
                    decoded, masked_weight, ls1, ls2, valid_length) = parallel.get()

            mx.nd.waitall()

            if batch_num > bucket_drop_iterations:
                latency = (time.time()-batch_begin_time) * 1000
                bucket_idx = np.argmax(bucket_keys_array >= data_list[0][0].shape[1])
                if data_list[0][0].shape[0] == bucket_batch_sizes[bucket_idx]:
                    # ignore abnormal samples
                    benchmark_latency_list[bucket_idx].append(latency)
                    logging.info("batch_num={}, batch_size={}, latency={}".format(batch_num, data_list[0][0].shape, latency))
            batch_num += 1

        # original bucket
        print(bucket_batch_sizes)
        gap_array = evaluate(dataloader, 8, parallel, bucket_drop_iterations)
        logging.info('Evaluation: avg gap={}'.format(np.asscalar(np.mean(gap_array))))
        # optimal bucket
        optimal_latency = np.asscalar(np.mean(np.array(benchmark_latency_list[-1], dtype='float32')))
        for bucket_idx in range(len(bucket_batch_sizes)-1):
            current_bucket_latency = np.asscalar(np.mean(np.array(benchmark_latency_list[bucket_idx], dtype='float32')))
            bucket_batch_sizes[bucket_idx] = int(round(optimal_latency / current_bucket_latency * bucket_batch_sizes[bucket_idx]))
        dataloader._batch_sampler._bucket_batch_sizes = bucket_batch_sizes
        print(bucket_batch_sizes)
        gap_array = evaluate(dataloader, 8, parallel, bucket_drop_iterations)
        logging.info('Evaluation: avg gap={}'.format(np.asscalar(np.mean(gap_array))))
        break

if __name__ == '__main__':
    ctx = [mx.cpu()] if args.gpus is None or args.gpus == '' else \
          [mx.gpu(int(x)) for x in args.gpus.split(',')]

    model, nsp_loss, mlm_loss, vocab = get_model_loss(ctx, args.model, args.pretrained,
                                                      args.dataset_name, None, args.dtype,
                                                      ckpt_dir=args.ckpt_dir,
                                                      start_step=args.start_step)

    store = mx.kv.create('device')
    nlp.utils.mkdir(args.ckpt_dir)

    if args.data:
        logging.info('Using training data at {}'.format(args.data))
        num_parts = 1 if args.dummy_data_len else store.num_workers
        part_idx = 0 if args.dummy_data_len else store.rank
        data_train = get_pretrain_data_npz(args.data, args.batch_size, len(ctx), True,
                                           args.use_avg_len, args.num_buckets,
                                           num_parts=num_parts, part_idx=part_idx,
                                           prefetch=not args.dummy_data_len, 
                                           round_len = args.bucket_round_len, 
                                           min_length=args.bucket_min_len)
        train(data_train, model, nsp_loss, mlm_loss, len(vocab), ctx, store)
    # if args.data_eval:
    #     logging.info('Using evaluation data at {}'.format(args.data_eval))
    #     data_eval = get_pretrain_data_npz(args.data_eval, args.batch_size_eval, len(ctx),
    #                                       False, False, 1)
    #     evaluate(data_eval, model, nsp_loss, mlm_loss, len(vocab), ctx,
    #              args.log_interval, args.dtype)
