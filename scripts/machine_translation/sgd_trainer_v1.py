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

# coding: utf-8
# pylint: disable=line-too-long
"""Parameter optimizer."""

from mxnet import optimizer as opt
from mxnet.gluon.parameter import ParameterDict, Parameter

import mxnet as mx
import types
import warnings
import math

import horovod.mxnet as hvd
from horovod.mxnet.mpi_ops import allreduce, allreduce_, size

class SGDTrainerV1(mx.gluon.Trainer):
    def __init__(self, params, optimizer, optimizer_params=None, blocking=False):

        super(SGDTrainerV1, self).__init__(
            params, optimizer, optimizer_params=optimizer_params, kvstore=None)

        # _scale is used to check and set rescale_grad for optimizer in Trainer.step()
        # function. Normalizing it by Horovod size, which is equivalent to performing
        # average in allreduce, has better performance. 
        self._scale /= size()

        self._blocking = blocking

    def _allreduce_grads(self):
        if size() == 1: return
        if self._blocking:
            mx.nd.waitall()
        for i, param in enumerate(self._params):
            if param.grad_req != 'null':
                allreduce_(param.list_grad()[0], average=False,
                           name=param.name, priority=-i)
        if self._blocking:
            mx.nd.waitall()