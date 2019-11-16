import os
import random
import warnings
import logging
import functools
import time

import mxnet as mx

from horovod.mxnet.mpi_ops import allreduce_

# logging
level = logging.INFO
logging.getLogger().setLevel(level)
os.environ['MXNET_GPU_MEM_POOL_TYPE'] = 'Round'
os.environ['MXNET_SAFE_ACCUMULATION'] = '1'

try:
    import horovod.mxnet as hvd
except ImportError:
    logging.info('horovod must be installed.')
    exit()
hvd.init()
store = None
num_workers = hvd.size()
rank = hvd.rank()
local_rank = hvd.local_rank()
is_master_node = rank == local_rank


def test_local_reduction(ctx):

    # test local reduction
    local_reduction_array = mx.nd.array([float(rank), float(rank), float(rank)]).as_in_context(ctx)
    mx.nd.waitall()
    logging.info('local_reduction_array before 1st local allreduce: {}'.format(local_reduction_array.asnumpy()))

    allreduce_(local_reduction_array, average=True,
                name='local_reduction_array', priority=0, 
                local_reduction = True)
    mx.nd.waitall()
    logging.info('local_reduction_array after 1st local allreduce: {}'.format(local_reduction_array.asnumpy()))

    local_reduction_array[:] = float(rank)+1
    mx.nd.waitall()
    logging.info('local_reduction_array before 2nd local allreduce: {}'.format(local_reduction_array.asnumpy()))

    allreduce_(local_reduction_array, average=True,
                name='local_reduction_array', priority=0, 
                local_reduction = True)
    mx.nd.waitall()
    logging.info('local_reduction_array after 2nd local allreduce: {}'.format(local_reduction_array.asnumpy()))

    local_reduction_array[:] = float(rank)+2
    mx.nd.waitall()
    logging.info('local_reduction_array before 1st cross-only allreduce: {}'.format(local_reduction_array.asnumpy()))

    allreduce_(local_reduction_array, average=True,
                name='local_reduction_array', priority=0, 
                cross_only = True)
    mx.nd.waitall()
    logging.info('local_reduction_array after 1st cross-only allreduce: {}'.format(local_reduction_array.asnumpy()))

    local_reduction_array[:] = float(rank)+3
    mx.nd.waitall()
    logging.info('local_reduction_array before 2nd cross-only allreduce: {}'.format(local_reduction_array.asnumpy()))

    allreduce_(local_reduction_array, average=True,
                name='local_reduction_array', priority=0, 
                cross_only = True)
    mx.nd.waitall()
    logging.info('local_reduction_array after 2nd cross-only allreduce: {}'.format(local_reduction_array.asnumpy()))

    allreduce_(local_reduction_array, average=True,
                name='local_reduction_array', priority=0, 
                local_reduction = False)
    mx.nd.waitall()
    logging.info('local_reduction_array after global allreduce: {}'.format(local_reduction_array.asnumpy()))

    return

if __name__ == '__main__':
    random_seed = random.randint(0, 1000)
    ctx = mx.gpu(local_rank)

    test_local_reduction(ctx)

    