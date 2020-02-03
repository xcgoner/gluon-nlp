"""
Transformer
=================================

This example shows how to implement the Transformer model with Gluon NLP Toolkit.

@inproceedings{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones,
          Llion and Gomez, Aidan N and Kaiser, Lukasz and Polosukhin, Illia},
  booktitle={Advances in Neural Information Processing Systems},
  pages={6000--6010},
  year={2017}
}
"""

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

import argparse
import logging
import math
import os
import random
import time

import numpy as np
import mxnet as mx
from mxnet import gluon

import gluonnlp as nlp
from gluonnlp.loss import LabelSmoothing, MaskedSoftmaxCELoss
from gluonnlp.model.transformer import ParallelTransformer, get_transformer_encoder_decoder
from gluonnlp.model.translation import NMTModel
from gluonnlp.utils.parallel import Parallel
import dataprocessor
from bleu import _bpe_to_words, compute_bleu
from translation import BeamSearchTranslator
from utils import logging_config

np.random.seed(100)
random.seed(100)
mx.random.seed(10000)

nlp.utils.check_version('0.9.0')

def init_comm():
    """Init communication for horovod"""
    try:
        import horovod.mxnet as hvd
    except ImportError:
        logging.info('horovod must be installed.')
        exit()
    hvd.init()
    num_workers = hvd.size()
    rank = hvd.rank()
    local_rank = hvd.local_rank()
    is_master_node = rank == local_rank
    ctxs = [mx.gpu(local_rank)] if args.gpu else [mx.cpu()]
    return num_workers, rank, local_rank, is_master_node, ctxs

num_workers, rank, local_rank, is_master_node, ctxs = init_comm()

a = mx.nd.array([rank])
hvd.allreduce_(a, name='loss_denom', average=True)
a_np = np.asscalar(loss_denom_nd.asnumpy())
print(a)

