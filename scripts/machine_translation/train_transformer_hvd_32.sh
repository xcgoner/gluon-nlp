#!/bin/bash
#PBS -l select=16:ncpus=112 -lplace=excl

source activate mxnet_hvd
source /homes/cx2/src/mxnet_hvd/MLSL/_install/intel64/bin/mlslvars.sh thread
source /opt/intel/compilers_and_libraries_2017/linux/mpi/bin64/mpivars.sh release_mt

export MLSL_MPI_VERSION_CHECK=0

### OPA FABRIC ###
# export I_MPI_FABRICS=ofi
export I_MPI_FABRICS=shm:tmi
export I_MPI_TMI_PROVIDER=psm2
export HFI_NO_CPUAFFINITY=1
export I_MPI_FALLBACK=0
export OMP_NUM_THREADS=28

### OPA FABRIC ###
export I_MPI_HYDRA_BOOTSTRAP=rsh
export I_MPI_HYDRA_BOOTSTRAP_EXEC=pbs_tmrsh

export KMP_AFFINITY=granularity=fine,compact,1,0

export MXNET_SUBGRAPH_BACKEND=MKLDNN


watchfile=/homes/cx2/src/localadam/uai2020/results/train_transformer_hvd.txt

cat $PBS_NODEFILE | uniq > $PBS_O_WORKDIR/hostfile
cd /homes/cx2/src/localadam/uai2020/gluon-nlp/scripts/machine_translation
mpirun -np 32 -machinefile $PBS_O_WORKDIR/hostfile -ppn 2 -genv I_MPI_PIN_DOMAIN auto:compact \
                       python train_transformer_hvd.py --dataset WMT2014BPE \
                       --src_lang en --tgt_lang de --batch_size 10800 \
                       --optimizer adam --num_accumulated 1 --lr 2.0 --warmup_steps 4000 \
                       --save_dir transformer_en_de_u512 --epochs 30 --scaled \
                       --average_start 5 --num_buckets 20 --bucket_scheme exp --bleu 13a --log_interval 10 2>&1 | tee -a $watchfile
