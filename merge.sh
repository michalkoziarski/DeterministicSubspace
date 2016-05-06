#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16384
#SBATCH --time=72:00:00

module add plgrid/tools/python/2.7.9

cd $HOME/DeterministicSubspace
python merge.py $1 $2
