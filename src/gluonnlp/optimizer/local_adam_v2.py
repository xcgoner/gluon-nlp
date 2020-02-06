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

"""Local Adam optimizer"""

from mxnet.optimizer import Optimizer, register
from mxnet.ndarray import zeros, NDArray
from mxnet.ndarray import square, power, sqrt, maximum, minimum, clip

__all__ = ['LOCALADAMV2']


@register
class LocalAdamV2(Optimizer):
    """The LocalAdamV2 optimizer.
    This class implements the optimizer described in *Adam: A Method for
    Stochastic Optimization*, available at http://arxiv.org/abs/1412.6980.
    If the storage types of grad is ``row_sparse``, and ``lazy_update`` is True, \
    **lazy updates** at step t are applied by::
        for row in grad.indices:
            rescaled_grad[row] = clip(grad[row] * rescale_grad + wd * weight[row], clip_gradient)
            m[row] = beta1 * m[row] + (1 - beta1) * rescaled_grad[row]
            v[row] = beta2 * v[row] + (1 - beta2) * (rescaled_grad[row]**2)
            lr = learning_rate * sqrt(1 - beta1**t) / (1 - beta2**t)
            w[row] = w[row] - lr * m[row] / (sqrt(v[row]) + epsilon)
    The lazy update only updates the mean and var for the weights whose row_sparse
    gradient indices appear in the current batch, rather than updating it for all indices.
    Compared with the original update, it can provide large improvements in model training
    throughput for some applications. However, it provides slightly different semantics than
    the original update, and may lead to different empirical results.
    Otherwise, **standard updates** at step t are applied by::
        rescaled_grad = clip(grad * rescale_grad + wd * weight, clip_gradient)
        m = beta1 * m + (1 - beta1) * rescaled_grad
        v = beta2 * v + (1 - beta2) * (rescaled_grad**2)
        lr = learning_rate * sqrt(1 - beta1**t) / (1 - beta2**t)
        w = w - lr * m / (sqrt(v) + epsilon)
    This optimizer accepts the following parameters in addition to those accepted
    by :class:`.Optimizer`.
    For details of the update algorithm, see :class:`~mxnet.ndarray.adam_update`.
    Parameters
    ----------
    beta1 : float, optional
        Exponential decay rate for the first moment estimates.
    beta2 : float, optional
        Exponential decay rate for the second moment estimates.
    epsilon : float, optional
        Small value to avoid division by 0.
    lazy_update : bool, optional
       Default is True. If True, lazy updates are applied \
       if the storage types of weight and grad are both ``row_sparse``.
    """
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
                 lazy_update=True, **kwargs):
        super(LocalAdamV2, self).__init__(learning_rate=learning_rate, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.lazy_update = lazy_update

    def create_state(self, index, weight):
        stype = weight.stype if self.lazy_update else 'default'
        return (zeros(weight.shape, weight.context, dtype=weight.dtype,
                      stype=stype),  # mean
                zeros(weight.shape, weight.context, dtype=weight.dtype,
                      stype=stype),  # variance
                zeros(weight.shape, weight.context, dtype=weight.dtype,
                      stype=stype))  # cached variance

    def update(self, index, weight, grad, state):
        assert(isinstance(weight, NDArray))
        assert(isinstance(grad, NDArray))
        self._update_count(index)
        lr = self._get_lr(index)
        wd = self._get_wd(index)

        t = self._index_update_count[index]
        coef1 = 1. - self.beta1**t
        coef2 = 1. - self.beta2**t
        lr *= math.sqrt(coef2)/coef1

        mean, var, cached_var = state

        # preprocess grad
        grad *= self.rescale_grad + wd * weight
        if self.clip_gradient is not None:
            grad = clip(grad, -self.clip_gradient, self.clip_gradient)

        mean[:] = self.beta1 * mean + (1. - self.beta1) * grad
        cached_var[:] = self.beta2 * cached_var + (1. - self.beta2) * square(grad)

        weight[:] -= lr * ( mean / (sqrt(var) + self.epsilon) )



