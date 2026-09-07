#!/bin/bash
#SBATCH -J test
#SBATCH -p bcores48,bcores48-2,bcores48-3
#SBATCH -n 48
#SBATCH --error=%J.err   
#SBATCH --output=%J.out
ulimit -s unlimited
module load intel/2017u5
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/software/wrf/deps-icc/lib
mpirun -np 4 /data/home/maguolin/WRF/WRFV4.6.0/run/wrf.exe

 