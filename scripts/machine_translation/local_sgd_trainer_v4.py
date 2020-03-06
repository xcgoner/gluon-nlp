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
from mxnet.ndarray import square
import mxnet.ndarray as nd

import mxnet as mx
import types
import warnings
import math
import logging

import horovod.mxnet as hvd
from horovod.mxnet.mpi_ops import allreduce, allreduce_

class LocalHVDTrainerV4(mx.gluon.Trainer):
    # only works with LocalAdaAlter
    def __init__(self, params, optimizer, optimizer_params=None, local_sgd_interval=0, local_sgd_warmup=800, beta1=0.9, beta2=0.999):
        # if local_sgd_interval == 0, fully sync

        super(LocalHVDTrainerV4, self).__init__(
            params, optimizer, optimizer_params=optimizer_params, kvstore=None, update_on_kvstore = False)
        
        self._update_on_kvstore = False

        self._local_sgd_interval = local_sgd_interval
        self._local_sgd_counter = 0
        self._local_sgd_warmup = local_sgd_warmup
        self._update_counter = 0

        # for adam
        self._beta1 = beta1
        self._beta2 = beta2
        self._coef1 = beta1**local_sgd_interval
        self._coef2 = beta2**local_sgd_interval
        # self._coef2 = beta2

        # print(self._local_sgd_interval)

    def step(self, batch_size, ignore_stale_grad=False):
        """Makes one step of parameter update. Should be called after
        `autograd.backward()` and outside of `record()` scope.
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
        rescale_grad = self._scale / batch_size
        self._check_and_rescale_grad(rescale_grad)

        if not self._kv_initialized:
            self._init_kvstore()
        if self._params_to_init:
            self._init_params()
    
        if self._local_sgd_interval == 0:
            self._allreduce_grads()

        self._update(ignore_stale_grad)

        self._update_counter += 1

        # debug
        g_sum_square = 0
        g_square_sum = 0
        for i, param in reversed(list(enumerate(self._params))):
            if param.grad_req != 'null':
                if param._stype == 'default':
                    g = param.list_grad()[0].copy()
                    g_square = param.list_grad()[0].square()
                    hvd.allreduce_(g, average=True, name=str(i), priority=0)
                    hvd.allreduce_(g_square, average=True, name=str(i), priority=0)
                    g_sum_square += g.square().sum().asnumpy()
                    g_square_sum += g_square.sum().asnumpy()
                else:
                    raise ValueError("Cannot pull row_sparse parameters for local SGD")
        if hvd.rank() == 0:
            logging.info('t:{}, g_sum_square:{}, g_square_sum: {}'.format(self._update_counter, g_sum_square, g_square_sum))

        if self._local_sgd_interval > 1:
            # local sgd
            self._local_sgd_counter += 1
            if self._local_sgd_counter == self._local_sgd_interval:
                self._local_sgd_counter = 0
                # synchronization
                # self.allreduce_params()
                # self.allreduce_states()
                # indicate that the parameters are synchronized in the current iteration
                return True
            return False
        return True

    def _allreduce_grads(self):
        for i, param in enumerate(self._params):
            if param.grad_req != 'null':
                if param.list_grad()[0].stype == 'default':
                    allreduce_(param.list_grad()[0], average=True,
                               name=str(i), priority=-i)
                else:
                    raise ValueError("Cannot pull row_sparse parameters for local SGD")

    def allreduce_params(self):
        """For each parameter, reduce the parameters from different contexts.
        Should be called after `autograd.backward()`, outside of `record()` scope,
        and before `trainer.update()`.
        For normal parameter updates, `step()` should be used, which internally calls
        `allreduce_grads()` and then `update()`. However, if you need to get the reduced
        gradients to perform certain transformation, such as in gradient clipping, then
        you may want to manually call `allreduce_grads()` and `update()` separately.
        """
        for i, param in enumerate(self._params):
            if param.grad_req != 'null':
                hvd.allreduce_(param.list_data()[0], average=True, 
                                       name=str(i), priority=-i)

    def allreduce_states(self):
        # only for Adam

        # debug
        var_sum = 0
        # g_square_sum = 0

        for i, param in reversed(list(enumerate(self._params))):
            if param.grad_req != 'null':
                mean, var, cached_mean, cached_var = self._updaters[0].states[i]
                if param._stype == 'default':
                    hvd.allreduce_(mean, average=True, 
                                   name=str(i+len(self._params)), priority=i-len(self._params)*2)
                    cached_var[:] *= self._coef2
                    cached_var[:] += (1-self._coef2) * square( ( mean - self._coef1*cached_mean ) / (1-self._coef1) )
                    var[:] = cached_var
                    cached_mean[:] = mean

                    # debug
                    var_sum += var.sum().asnumpy()
                    # g_square_sum += g_square.sum().asnumpy()
                else:
                    raise ValueError("Cannot pull row_sparse parameters for local SGD")
        # debug
        if hvd.rank() == 0:
            logging.info('t:{}, var: {}'.format(self._update_counter, var_sum))
            # logging.info('t:{}, var: {}, g_square: {}'.format(self._update_counter, var_sum,g_square_sum))