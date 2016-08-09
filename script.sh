#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --time=72:00:00

module add plgrid/tools/python/2.7.9

cd $HOME/DeterministicSubspace/DeterministicSubspace

classifiers=("CART" "kNN" "SVM" "NaiveBayes" "ParzenKDE" "NNKDE" "GMM")
measures=("accuracy" "mutual_information" "correlation")

while read dataset; do
    for fold in $(seq 0 9); do
        for k in $(seq 5 5 50); do
            python script.py -dataset $dataset -fold $fold -classifier RandomForest -k $k

            for classifier in ${classifiers[@]}; do
                python script.py -dataset $dataset -fold $fold -classifier $classifier -method RS -k $k

                for alpha in $(seq 0.0 0.1 1.0); do
                    for measure in ${measures[@]}; do
                        python script.py -dataset $dataset -fold $fold -classifier $classifier -method DS -k $k -measure $measure -alpha $alpha
                    done
                done
            done
        done
    done
done < dataset.txt
