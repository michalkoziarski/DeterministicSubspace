#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16384

module add plgrid/tools/python/2.7.9

cd $HOME/DeterministicSubspace
python script.py
