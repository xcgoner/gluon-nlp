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
parser.add_argument('--bucket_epochs', type=int, default=1000, help='epochs for bucket optimization')
parser.add_argument('--bucket_batchsize', type=int, default=400, help='batch size for bucket optimization')
parser.add_argument('--bucket_lr', type=float, default=0.01, help='learning rate for bucket optimization')
parser.add_argument('--bucket_lr_decay_epoch', type=str, default='40', help='decay epoch for bucket optimization')
parser.add_argument('--bucket_lr_decay_rate', type=str, default='0.1', help='decay rate for bucket optimization')
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

    trainer = mx.gluon.Trainer(model.collect_params(), args.optimizer, optim_params,
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
    benchmark_latency_list = []
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
                if data_list[0][0].shape[0] == max_bucket_batch_size and data_list[0][0].shape[1] == max_bucket_key:
                    benchmark_latency_list.append(latency)
                    benchmark_latency_array = np.array(benchmark_latency_list)
                    min_latency = np.asscalar(np.min(benchmark_latency_array))
                    max_latency = np.asscalar(np.max(benchmark_latency_array))
                    logging.info("batch_num={}, batch_size={}, latency={}, avg={}, std={}, min={}, max={}, gap={}".format(batch_num, data_list[0][0].shape, latency, np.asscalar(np.mean(benchmark_latency_array)), np.asscalar(np.std(benchmark_latency_array)), min_latency, max_latency, max_latency-min_latency))
            batch_num += 1
    
    benchmark_latency = np.asscalar(np.mean(benchmark_latency_array))
    benchmark_latency_gap = max(benchmark_latency_list) - min(benchmark_latency_list)



    bucket_step_num = 0

    for epoch in range(args.epochs):
        batch_num = 0
        iter_num = 0
        latency_list = []
        bucket_grad = np.zeros_like(bucket_batch_sizes_array, dtype='float32')

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

                if iter_num > bucket_drop_iterations and ((data_list[0][0].shape[0] == max_bucket_batch_size and data_list[0][0].shape[1] == max_bucket_key) \
                    or data_list[0][0].shape[0] not in bucket_batch_sizes):
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
                
                # generate quasi gradient
                grad = latency - benchmark_latency
                bucket_idx = np.argmax(bucket_keys_array >= data_list[0][0].shape[1])
                bucket_grad[bucket_idx] += grad

                latency_list.append(latency)
                latency_array = np.array(latency_list)
                min_latency = np.asscalar(np.min(latency_array))
                max_latency = np.asscalar(np.max(latency_array))
                logging.info("batch_num={}, batch_size={}, latency={}, avg={}, std={}, min={}, max={}, gap={}".format(batch_num, data_list[0][0].shape, latency, np.asscalar(np.mean(latency_array)), np.asscalar(np.std(latency_array)), min_latency, max_latency, max_latency-min_latency))

                if batch_num == args.bucket_batchsize - 1:
                    # gradient descent
                    bucket_batch_sizes_array -= (args.bucket_lr * bucket_grad / args.bucket_batchsize)
                    # update dataloader
                    bucket_batch_sizes = bucket_batch_sizes_array.round().astype('int').tolist()
                    dataloader._batch_sampler._bucket_batch_sizes = bucket_batch_sizes
                    break

                batch_num += 1
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
                                           prefetch=not args.dummy_data_len)
        train(data_train, model, nsp_loss, mlm_loss, len(vocab), ctx, store)
    # if args.data_eval:
    #     logging.info('Using evaluation data at {}'.format(args.data_eval))
    #     data_eval = get_pretrain_data_npz(args.data_eval, args.batch_size_eval, len(ctx),
    #                                       False, False, 1)
    #     evaluate(data_eval, model, nsp_loss, mlm_loss, len(vocab), ctx,
    #              args.log_interval, args.dtype)
