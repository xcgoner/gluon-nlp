#!/bin/bash
#PBS -l select=5:ncpus=112 -lplace=excl

source /opt/intel/compilers_and_libraries_2017/linux/mpi/bin64/mpivars.sh
source activate mxnet_mkl_latest


### OPA FABRIC ###
export I_MPI_FABRICS=ofi
export I_MPI_TMI_PROVIDER=psm2
export HFI_NO_CPUAFFINITY=1
export I_MPI_FALLBACK=0
export OMP_NUM_THREADS=28

### OPA FABRIC ###
export I_MPI_HYDRA_BOOTSTRAP=rsh
export I_MPI_HYDRA_BOOTSTRAP_EXEC=pbs_tmrsh

export KMP_AFFINITY=granularity=fine,compact,1,0;


watchfile=/homes/cx2/src/localadam/uai2020/results/text_hvd.txt

cat $PBS_NODEFILE | uniq > $PBS_O_WORKDIR/hostfile
cd /homes/cx2/src/localadam/uai2020/gluon-nlp/scripts/machine_translation
mpirun -np 10 -machinefile $PBS_O_WORKDIR/hostfile python test_hvd.py 2>&1 | tee -a $watchfile
