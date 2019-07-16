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

class DistributedLocalSGDTrainer(hvd.DistributedTrainer):
    def __init__(self, params, optimizer, optimizer_params=None, 
                local_sgd=1):

        super(DistributedLocalSGDTrainer, self).__init__(
            params, optimizer, optimizer_params=optimizer_params)

        # _scale is used to check and set rescale_grad for optimizer in Trainer.step()
        # function. Normalizing it by Horovod size, which is equivalent to performing
        # average in allreduce, has better performance. 
        if local_sgd is None or local_sgd <= 1:
            self._local_sgd = 1
        else:
            self._local_sgd = local_sgd
        self._local_sgd_counter = 0


    def step(self, batch_sizes, ignore_stale_grad=False):
        """Makes one step of parameter update. Should be called after
        `autograd.backward()` and outside of `record()` scope.

        For normal parameter updates, `step()` should be used, which internally calls
        `allreduce_grads()` and then `update()`. However, if you need to get the reduced
        gradients to perform certain transformation, such as in gradient clipping, then
        you may want to manually call `allreduce_grads()` and `update()` separately.

        Parameters
        ----------
        batch_sizes : [int]
            Batch size of data processed. Gradient will be normalized by `1/batch_size`.
            Set this to 1 if you normalized loss manually with `loss = mean(loss)`.
        ignore_stale_grad : bool, optional, default=False
            If true, ignores Parameters with stale gradient (gradient that has not
            been updated by `backward` after last step) and skip update.
        """
        rescale_grad = self._scale / batch_size
        self._check_and_rescale_grad(rescale_grad)

        if not self._kv_initialized:
            self._init_kvstore()
        if self._params_to_init:
            self._init_params()

        if self._local_sgd == 1:
            # if not local sgd
            self._allreduce_grads()

        self._update(ignore_stale_grad)

        if self._local_sgd > 1:
            # local sgd
            self._local_sgd_counter += 1
            if self._local_sgd_counter == self._local_sgd:
                self._local_sgd_counter = 0
                # synchronization
                self._allreduce_params()
                self._allreduce_states()
                # indicate that the parameters are synchronized in the current iteration
                return True
            return False
        return True

    def _allreduce_grads(self):
        # sort needed for Python < 3.6 is not guaranteed
        for i, param in enumerate(sorted(self._params, key=lambda p: p.name)):
            if param.grad_req != 'null':
                allreduce_(param.list_grad()[0], average=False,
                name=str(i), priority=-i)

    def allreduce_params(self):
        """For each parameter, reduce the gradients from different contexts.

        Should be called after `autograd.backward()`, outside of `record()` scope,
        and before `trainer.update()`.

        For normal parameter updates, `step()` should be used, which internally calls
        `allreduce_grads()` and then `update()`. However, if you need to get the reduced
        gradients to perform certain transformation, such as in gradient clipping, then
        you may want to manually call `allreduce_grads()` and `update()` separately.
        """
        if self._params_to_init:
            self._init_params()
        
        self._allreduce_params()

    def _allreduce_params(self):
        # print("_allreduce_params")
        for i, param in enumerate(sorted(self._params, key=lambda p: p.name)):
            if param.grad_req != 'null':
                hvd.allreduce_(param.list_data()[0], average=True,
                                name=str(i), priority=-i)

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
        for i, param in reversed(list(enumerate(sorted(self._params, key=lambda p: p.name)))):
            if param.grad_req != 'null':
                if isinstance(self._updaters[0].states[i], (tuple, list)):
                    # for some optimizers, there are multiple states (mean, variance), such as Adam
                    for j in range(len(self._updaters[0].states[i])):
                        idx = i+len(self._params)*(j+1)
                        hvd.allreduce_(self._updaters[0].states[i][j], average=True,
                                    name=str(idx), priority=-i-len(self._params)*2)
                else:
                    idx = i+len(self._params)
                    hvd.allreduce_(self._updaters[0].states[i], average=True,
                                    name=str(idx), priority=-i-len(self._params)*2)

