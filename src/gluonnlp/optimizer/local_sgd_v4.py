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

class LocalSGDTrainerV4(mx.gluon.Trainer):
    def __init__(self, params, optimizer, optimizer_params=None, kvstore='device'):

        super(LocalSGDTrainerV4, self).__init__(
            params, optimizer, optimizer_params=optimizer_params, kvstore=kvstore, update_on_kvstore=False)

    def check_grad_var(self, num_workers_list):
        # print("check_grad_var")

        if not isinstance(num_workers_list, list):
            num_workers_list = [num_workers_list]

        mx.nd.waitall()

        # first average then square
        avg_var_layers_list = [[] for _ in num_workers_list]
        for i, param in enumerate(self._params):
            if param.grad_req != 'null':
                grad_mean = mx.nd.zeros(param.list_grad()[0].shape, param.list_grad()[0].context, dtype=param.list_grad()[0].dtype)
                for j, num_workers in enumerate(num_workers_list):
                    self._kvstore.push(i, param.list_grad()[0:num_workers], priority=-i)
                    self._kvstore.pull(i, [grad_mean], priority=-i)
                    # take average
                    grad_mean /= num_workers
                    avg_var_layers_list[j].append(mx.nd.mean(mx.nd.square(grad_mean)))
        mx.nd.waitall()
        avg_var_scalars_list = [[avg_var.asscalar() for avg_var in avg_var_layers] for avg_var_layers in avg_var_layers_list]

        # first square then average
        var_avg_layers_list = [[] for _ in num_workers_list]
        for i, param in enumerate(self._params):
            if param.grad_req != 'null':
                grad_vars = [mx.nd.zeros(grad.shape, grad.context, dtype=grad.dtype) for grad in param.list_grad()]
                for var, grad in zip(grad_vars, param.list_grad()):
                    mx.nd.square(grad, out=var)
                for j, num_workers in enumerate(num_workers_list):
                    self._kvstore.push(i, grad_vars[0:num_workers], priority=-i)
                    self._kvstore.pull(i, [grad_vars[0]], priority=-i)
                    # take average
                    grad_vars[0] /= num_workers
                    var_avg_layers_list[j].append(mx.nd.mean(grad_vars[0]))
        mx.nd.waitall()
        var_avg_scalars_list = [[var_avg.asscalar() for var_avg in var_avg_layers] for var_avg_layers in var_avg_layers_list]

        return avg_var_scalars_list, var_avg_scalars_list


