#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --time=72:00:00

module add plgrid/tools/python/2.7.9

cd $HOME/DeterministicSubspace/DeterministicSubspace
python script.py -dataset $1 -fold $2 -classifier $3 -method $4 -measure $5 -k $6 -alpha $7
