#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --time=72:00:00

module add plgrid/tools/python/2.7.9

cd $HOME/DeterministicSubspace
python script.py $1 $2 $3 $4 $5
