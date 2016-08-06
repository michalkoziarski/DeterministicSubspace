#!/bin/bash

classifiers=("CART" "kNN" "SVM" "NaiveBayes" "ParzenKDE" "NNKDE" "GMM")
measures=("accuracy" "mutual_information" "correlation")

while read dataset; do
    for fold in $(seq 0 9); do
        for k in $(seq 5 5 50); do
            sbatch script.sh $dataset $fold RandomForest - - $k -

            for classifier in ${classifiers[@]}; do
                sbatch script.sh $dataset $fold $classifier RS - $k -

                for alpha in $(seq 0.0 0.1 1.0); do
                    for measure in ${measures[@]}; do
                        sbatch script.sh $dataset $fold $classifier DS $measure $k $alpha
                    done
                done
            done
        done
    done
done < datasets.txt
