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
import math

import horovod.mxnet as hvd
from horovod.mxnet.mpi_ops import allreduce, allreduce_

class FP16DistributedLocalSGDTrainer(hvd.DistributedTrainer):
    def __init__(self, params, optimizer, optimizer_params=None, 
                local_sgd=1):
        
        # important: only works for bert_adam with fp16 trainer

        super(FP16DistributedLocalSGDTrainer, self).__init__(
            params, optimizer, optimizer_params=optimizer_params)

        # _scale is used to check and set rescale_grad for optimizer in Trainer.step()
        # function. Normalizing it by Horovod size, which is equivalent to performing
        # average in allreduce, has better performance. 
        if local_sgd is None or local_sgd <= 1:
            self._local_sgd = 1
        else:
            self._local_sgd = local_sgd
            self._scale *= hvd.size()
        self._local_sgd_counter = 0

    def _allreduce_grads(self):
        if self._local_sgd == 1:
            for i, param in enumerate(self._params):
                if param.grad_req != 'null':
                    hvd.allreduce(param.list_grad()[0], average=False,
                                  name=str(i), priority=-i)

    # def _update(self, ignore_stale_grad=False):
    #     super(FP16DistributedLocalSGDTrainer, self)._update(ignore_stale_grad=ignore_stale_grad)
    #     if self._local_sgd > 1:
    #         # local sgd
    #         self._local_sgd_counter += 1
    #         if self._local_sgd_counter == self._local_sgd:
    #             self._local_sgd_counter = 0
    #             # synchronization
    #             self._allreduce_params()
    #             # self._allreduce_states()

    def update(self, batch_size, ignore_stale_grad=False):
        """Makes one step of parameter update.
        Should be called after `autograd.backward()` and outside of `record()` scope,
        and after `trainer.update()`.
        For normal parameter updates, `step()` should be used, which internally calls
        `allreduce_grads()` and then `update()`. However, if you need to get the reduced
        gradients to perform certain transformation, such as in gradient clipping, then
        you may want to manually call `allreduce_grads()` and `update()` separately.
        Parameters
        ----------
        batch_size : int
            Batch size of data processed. Gradient will be normalized by `1/batch_size`.
            Set this to 1 if you normalized loss manually with `loss = mean(loss)`.
        ignore_stale_grad : bool, optional, default=False
            If true, ignores Parameters with stale gradient (gradient that has not
            been updated by `backward` after last step) and skip update.
        """
        if not self._kv_initialized:
            self._init_kvstore()
        if self._params_to_init:
            self._init_params()
        assert not (self._kvstore and self._update_on_kvstore), \
                'update() when parameters are updated on kvstore ' \
                'is not supported. Try setting `update_on_kvstore` ' \
                'to False when creating trainer.'

        self._check_and_rescale_grad(self._scale / batch_size)
        self._update(ignore_stale_grad)

        if self._local_sgd > 1:
            # local sgd
            self._local_sgd_counter += 1
            if self._local_sgd_counter == self._local_sgd:
                self._local_sgd_counter = 0
                # synchronization
                # self._allreduce_params()
                self._allreduce_states()

    def _allreduce_params(self):
        # print("_allreduce_params")
        mx.nd.waitall()
        print('_allreduce_params started')
        for i, param in enumerate(self._params):
            if param.grad_req != 'null':
                hvd.allreduce(param.list_data()[0], average=True,
                                name=str(i), priority=-i)
        mx.nd.waitall()
        print('_allreduce_params started')

    def allreduce_states(self):
        """For each parameter, reduce the gradients from different contexts.

        Should be called after `autograd.backward()`, outside of `record()` scope,
        and before `trainer.update()`.

        For normal parameter updates, `step()` should be used, which internally calls
        `allreduce_grads()` and then `update()`. However, if you need to get the reduced
        gradients to perform certain transformation, such as in gradient clipping, then
        you may want to manually call `allreduce_grads()` and `update()` separately.
        """
        
        self._allreduce_states()

    def _allreduce_states(self):
        # print("_allreduce_states")
        # for i, param in enumerate(sorted(self._params, key=lambda p: p.name)):
        # important: only works for bert_adam with fp16 trainer
        # sync params
        # for i, param in enumerate(self._params):
        #     if param.grad_req != 'null':
        #         hvd.allreduce(self._updaters[0].states[i][1], average=True,
        #                         name=str(i), priority=-i)
        #         # copy fp32 weight to fp16 weight, assume using hvd with single GPU per process
        #         self._updaters[0].states[i][1].copyto(param.list_data()[0])
        # sync mean and var
        mx.nd.waitall()
        print('_allreduce_states started')
        for i, param in reversed(list(enumerate(self._params))):
            if param.grad_req != 'null':
                # for j in range(len(self._updaters[0].states[i][0])):
                    j = 0
                    idx = i+len(self._params)*(j+1)
                    hvd.allreduce(self._updaters[0].states[i][0][j], average=True,
                                name=str(idx), priority=-i-len(self._params)*2)

        mx.nd.waitall()
        print('_allreduce_states finished')

