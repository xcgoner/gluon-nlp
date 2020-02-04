#!/bin/bash
#PBS -l select=5:ncpus=112 -lplace=excl

source activate mxnet_hvd
source /homes/cx2/src/mxnet_hvd/MLSL/_install/intel64/bin/mlslvars.sh thread
source /opt/intel/compilers_and_libraries_2017/linux/mpi/bin64/mpivars.sh release_mt

export MLSL_MPI_VERSION_CHECK=0

### OPA FABRIC ###
export I_MPI_FABRICS=ofi
export I_MPI_TMI_PROVIDER=psm2
export HFI_NO_CPUAFFINITY=1
export I_MPI_FALLBACK=0
export OMP_NUM_THREADS=66

### OPA FABRIC ###
export I_MPI_HYDRA_BOOTSTRAP=rsh
export I_MPI_HYDRA_BOOTSTRAP_EXEC=pbs_tmrsh

export KMP_AFFINITY=granularity=fine,compact,1,0;


watchfile=/homes/cx2/src/localadam/uai2020/results/text_hvd.txt

cat $PBS_NODEFILE | uniq > $PBS_O_WORKDIR/hostfile
cd /homes/cx2/src/localadam/uai2020/test_hvd
mpirun -np 5 -machinefile $PBS_O_WORKDIR/hostfile python test_hvd.py 2>&1 | tee -a $watchfile
