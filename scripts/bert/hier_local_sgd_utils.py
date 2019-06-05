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

"""Trainer for mixed precision training."""
import warnings
import collections
import numpy as np
import mxnet as mx
from mxnet import nd
from mxnet import gluon

from horovod.mxnet import mpi_ops

from fp16_utils import grad_global_norm, StaticLossScaler, DynamicLossScaler

class HVDLocalSGDTrainer(gluon.Trainer):
    """Applies an `Optimizer` on a set of Parameters. Trainer should
    be used together with `autograd`.

    .. note::

        For the following cases, updates will always happen on kvstore,
        i.e., you cannot set update_on_kvstore=False.

        - dist kvstore with sparse weights or sparse gradients
        - dist async kvstore
        - `optimizer.lr_scheduler` is not None

    Parameters
    ----------
    params : ParameterDict
        The set of parameters to optimize.
    optimizer : str or Optimizer
        The optimizer to use. See
        `help <http://mxnet.io/api/python/optimization/optimization.html#the-mxnet-optimizer-package>`_
        on Optimizer for a list of available optimizers.
    optimizer_params : dict
        Key-word arguments to be passed to optimizer constructor. For example,
        `{'learning_rate': 0.1}`. All optimizers accept learning_rate, wd (weight decay),
        clip_gradient, and lr_scheduler. See each optimizer's
        constructor for a list of additional supported arguments.
    kvstore : str or KVStore
        kvstore type for multi-gpu and distributed training. See help on
        :any:`mxnet.kvstore.create` for more information.
    compression_params : dict
        Specifies type of gradient compression and additional arguments depending
        on the type of compression being used. For example, 2bit compression requires a threshold.
        Arguments would then be {'type':'2bit', 'threshold':0.5}
        See mxnet.KVStore.set_gradient_compression method for more details on gradient compression.
    update_on_kvstore : bool, default None
        Whether to perform parameter updates on kvstore. If None, then trainer will choose the more
        suitable option depending on the type of kvstore. If the `update_on_kvstore` argument is
        provided, environment variable `MXNET_UPDATE_ON_KVSTORE` will be ignored.

    Properties
    ----------
    learning_rate : float
        The current learning rate of the optimizer. Given an Optimizer object
        optimizer, its learning rate can be accessed as optimizer.learning_rate.
    """
    def __init__(self, params, optimizer, optimizer_params=None, kvstore='device',
                 compression_params=None, update_on_kvstore=None, local_sgd=None, local_sgd_regularization=0, local_sgd_regularization_interval=0):

        if local_sgd is None or local_sgd <= 1:
            self._local_sgd = 1
        else:
            self._local_sgd = local_sgd
        self._local_sgd_counter = 0
        # only allows update on workers
        update_on_kvstore = False
        if isinstance(kvstore, str) and 'dist' in kvstore:
            kvstore = 'device'
            warnings.warn('currently, we only consider local kvstore, which is used for hierarchical allreduce')
        self._local_sgd_regularization = local_sgd_regularization
        self._local_sgd_regularization_interval = local_sgd_regularization_interval
        self._local_sgd_regularization_counter = 0
        self._is_states_initialized = False
        super(HVDLocalSGDTrainer, self).__init__(
            params, optimizer, optimizer_params=optimizer_params, kvstore=kvstore,
                 compression_params=compression_params, update_on_kvstore=update_on_kvstore)

    def init_states(self):
        """Initialize states (momentum) in the KVStore, for local sgd
        """
        assert self._kv_initialized, "Cannot initialize states in KVStore " \
                                     "when KVStore is not initialized."
        if self._kvstore and self._is_states_initialized == False:
            for i, param in enumerate(self._params):
                if param.grad_req != 'null':
                    if isinstance(self._updaters[0].states[i], (tuple, list)):
                        # for some optimizers, there are multiple states (mean, variance), such as Adam
                        for j in range(len(self._updaters[0].states[i])):
                            state_arrays = [updater.states[i][j] for updater in self._updaters]
                            self._kvstore.init(i+len(self._params)*(j+1), self._updaters[0].states[i][j])
                    else:
                        state_arrays = [updater.states[i] for updater in self._updaters]
                        self._kvstore.init(i+len(self._params), self._updaters[0].states[i])
            self._is_states_initialized = True

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

        if self._local_sgd == 1:
            # if not local sgd
            self._allreduce_grads()
        # self._allreduce_grads()

        if self._local_sgd > 1 and self._local_sgd_counter == 0 and self._local_sgd_regularization > 0:
            # regularization for local sgd
            self._local_sgd_regularization_params = []
            for i, param in enumerate(self._params):
                if param.grad_req != 'null' and param._stype == 'default':
                    self._local_sgd_regularization_params.append([self._local_sgd_regularization * x.copy() for x in param.list_data()])
                else:
                    self._local_sgd_regularization_params.append([])

        self._update(ignore_stale_grad)

        if self._local_sgd > 1 and self._local_sgd_regularization > 0:
            # regularization for local sgd
            mixing_weight = (1 - self._local_sgd_regularization)
            self._local_sgd_regularization_counter += 1
            if self._local_sgd_regularization_interval == 0 or self._local_sgd_regularization_interval == self._local_sgd_regularization_counter:
                self._local_sgd_regularization_counter = 0
                for i, param in enumerate(self._params):
                    if param.grad_req != 'null' and param._stype == 'default':
                        for j, data in enumerate(param.list_data()):
                            data[:] = self._local_sgd_regularization_params[i][j] + mixing_weight * data

        if self._local_sgd > 1:
            # local sgd
            self._local_sgd_counter += 1
            if self._local_sgd_counter == self._local_sgd:
                self._local_sgd_counter = 0
                # synchronization
                self._allreduce_params()
                if self._is_states_initialized:
                    self._allreduce_states()

    def _hvd_allreduce_grads(self):
        for i, param in enumerate(self._params):
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
        if not self._kv_initialized:
            self._init_kvstore()
        if self._params_to_init:
            self._init_params()
        
        self._allreduce_params()

    def _allreduce_params(self):
        if self._kvstore:
            for i, param in enumerate(self._params):
                if param.grad_req != 'null':
                    self._kvstore.push(i, param.list_data(), priority=-i)
                    if param._stype == 'default':
                        # fetch value from reduced buf
                        # TODO(xcong) more efficient implementation
                        self._kvstore.pull(i, [param.list_data()[0]], priority=-i)
                        # hvd
                        mpi_ops.allreduce_(param.list_data()[0], average=False, name=str(i), priority=-i)
                        # take average
                        # assume that every worker has the same number of gpus/contexts
                        # TODO(xcong) sanity check of the average/normalization factor
                        num_workers = mpi_ops.size() * len(param.list_data())
                        for data in param.list_data():
                            data /= num_workers
                        # local broadcast
                        for data in param.list_data():
                            if data.context is not param.list_data()[0].context:
                                param.list_data()[0].copyto(data)
                    else:
                        raise ValueError("Cannot pull row_sparse parameters for local SGD")

    def allreduce_states(self):
        """For each parameter, reduce the gradients from different contexts.

        Should be called after `autograd.backward()`, outside of `record()` scope,
        and before `trainer.update()`.

        For normal parameter updates, `step()` should be used, which internally calls
        `allreduce_grads()` and then `update()`. However, if you need to get the reduced
        gradients to perform certain transformation, such as in gradient clipping, then
        you may want to manually call `allreduce_grads()` and `update()` separately.
        """
        if not self._kv_initialized:
            self._init_kvstore()
        
        if not self._is_states_initialized:
            raise ValueError("States are not initiallized")
        self._allreduce_states()

    def _allreduce_states(self):
        if self._kvstore:
            # for i, param in enumerate(self._params):
            for i, param in reversed(list(enumerate(self._params))):
                if param.grad_req != 'null':
                    if isinstance(self._updaters[0].states[i], (tuple, list)):
                        # for some optimizers, there are multiple states (mean, variance), such as Adam
                        for j in range(len(self._updaters[0].states[i])):
                            state_arrays = [updater.states[i][j] for updater in self._updaters]
                            idx = i+len(self._params)*(j+1)
                            self._kvstore.push(idx, state_arrays, priority=i-len(self._params)*2)
                            if param._stype == 'default':
                                self._kvstore.pull(idx, state_arrays, priority=i-len(self._params)*2)
                                # take average
                                # assume that every worker has the same number of gpus/contexts
                                num_workers = float(self._kvstore.num_workers * len(state_arrays))
                                for state in state_arrays:
                                    state /= num_workers
                            else:
                                raise ValueError("Cannot pull row_sparse parameters for local SGD")
                    else:
                        state_arrays = [updater.states[i] for updater in self._updaters]
                        idx = i+len(self._params)
                        self._kvstore.push(idx, state_arrays, priority=i-len(self._params)*2)
                        if param._stype == 'default':
                            self._kvstore.pull(idx, state_arrays, priority=i-len(self._params)*2)
                            # take average
                            # assume that every worker has the same number of gpus/contexts
                            num_workers = self._kvstore.num_workers * len(state_arrays)
                            for state in state_arrays:
                                state /= num_workers
                        else:
                            raise ValueError("Cannot pull row_sparse parameters for local SGD")


class LocalSGDFP16Trainer(object):
    """ Trainer for mixed precision training.

    Parameters
    ----------
    trainer: LocalSGDFP32Trainer
      the local fp32 trainer.
    dynamic_loss_scale: bool. Default is True
      whether to use dynamic loss scaling. This is recommended for optimizing model
      parameters using FP16.
    loss_scaler_params : dict
        Key-word arguments to be passed to loss scaler constructor. For example,
        `{"init_scale" : 2.**15, "scale_window" : 2000, "tolerance" : 0.05}`
        for `DynamicLossScaler`.
        See each `LossScaler` for a list of supported arguments'
    """
    def __init__(self, trainer, dynamic_loss_scale=True, loss_scaler_params=None):
        if trainer._kvstore_params['update_on_kvstore'] is not False and trainer._kvstore:
            err = 'Only gluon.Trainer created with update_on_kvstore=False is supported.'
            raise NotImplementedError(err)
        self.fp32_trainer = trainer
        loss_scaler_params = loss_scaler_params if loss_scaler_params else {}
        self._scaler = DynamicLossScaler(**loss_scaler_params) if dynamic_loss_scale \
                       else StaticLossScaler(**loss_scaler_params)
        # if the optimizer supports NaN check, we can always defer the NaN check to the optimizer
        # TODO(haibin) this should be added via registry
        self._support_nan_check = trainer._optimizer.__class__.__name__ == 'BERTAdam'

    def backward(self, loss):
        """backward propagation with loss"""
        with mx.autograd.record():
            if isinstance(loss, (tuple, list)):
                ls = [l * self._scaler.loss_scale for l in loss]
            else:
                ls = loss * self._scaler.loss_scale
        mx.autograd.backward(ls)

    def step(self, batch_size, max_norm=None):
        """Makes one step of parameter update. Should be called after
        `fp16_optimizer.backward()`, and outside of `record()` scope.

        Parameters
        ----------
        batch_size : int
            Batch size of data processed. Gradient will be normalized by `1/batch_size`.
            Set this to 1 if you normalized loss manually with `loss = mean(loss)`.
        max_norm : NDArray, optional, default is None
            max value for global 2-norm of gradients.
        """
        # TODO(xcong) local sgd
        # self.fp32_trainer.allreduce_grads()
        step_size = batch_size * self._scaler.loss_scale
        if max_norm:
            norm, ratio, is_finite = grad_global_norm(self.fp32_trainer._params,
                                                      max_norm * self._scaler.loss_scale)
            step_size = ratio * step_size
            if self._support_nan_check:
                self.fp32_trainer.update(step_size)
                overflow = is_finite.asscalar() < 1
            else:
                overflow = not np.isfinite(norm.asscalar())
                if not overflow:
                    self.fp32_trainer.update(step_size)
        else:
            # TODO(haibin) optimize the performance when max_norm is not present
            # sequentially adding isnan/isinf results may be slow
            if self._support_nan_check:
                self.fp32_trainer.update(step_size)
                overflow = self._scaler.has_overflow(self.fp32_trainer._params)
            else:
                overflow = self._scaler.has_overflow(self.fp32_trainer._params)
                if not overflow:
                    self.fp32_trainer.update(step_size)
        # update scale based on overflow information
        self._scaler.update_scale(overflow)

