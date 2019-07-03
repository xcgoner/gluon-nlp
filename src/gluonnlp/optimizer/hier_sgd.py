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
from mxnet.model import _create_kvstore, _create_sparse_kvstore
from mxnet.gluon.parameter import ParameterDict, Parameter

import mxnet as mx
import types
import warnings

import horovod.mxnet as hvd

class DistributedHierTrainer(mx.gluon.Trainer):
    def __init__(self, params, optimizer, optimizer_params=None, kvstore='device'):

        super(DistributedHierTrainer, self).__init__(
            params, optimizer, optimizer_params=optimizer_params, kvstore=kvstore, update_on_kvstore=False)

        # _scale is used to check and set rescale_grad for optimizer in Trainer.step()
        # function. Normalizing it by Horovod size, which is equivalent to performing
        # average in allreduce, has better performance. 
        self._scale /= hvd.size()
        self._allreduce_grads_counter = 0

    def _allreduce_grads(self):
        # hierarchical allreduce, combining local kvstore and hvd
        # if self._allreduce_grads_counter == 10000:
        #     self._allreduce_grads_counter = 0
        # name_base = self._allreduce_grads_counter * len(self._params)
        for i, param in enumerate(self._params):
            if param.grad_req != 'null':
                self._kvstore.push(i, param.list_grad(), priority=-i)
                # TODO(xcong) allreduce the buffer, avoid the extra copy in kvstore.pull
                self._kvstore.pull(i, [param.list_grad()[0]], priority=-i)
                # hvd.allreduce(param.list_grad()[0], average=False, 
                #               name=str(i + name_base), priority=-i)
                hvd.allreduce(param.list_grad()[0], average=False, priority=-i)
                for j in range(1, len(param.list_grad())):
                    param.list_grad()[0].copyto(param.list_grad()[j])

    def broadcast_params(self):
        """For each parameter, broadcast the parameters to different processes and contexts.

        Should be called after the parameters are actually initialized

        """
        for i, param in enumerate(self._params):
            if param.grad_req != 'null':
                hvd.broadcast(param.list_data()[0], root_rank=0, name = str(i))
                for j in range(1, len(param.list_data())):
                    param.list_data()[0].copyto(param.list_data()[j])

