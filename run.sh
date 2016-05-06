#!/bin/bash

ks=(5 10 15 20 25 30 35 40 45 50)
clfs=("DecisionTreeClassifier()" "KNeighborsClassifier()" "LinearSVC()" "GaussianNB()")

for i in 'seq 1 5'; do
    for clf in ${clfs[@]}; do
        for k in ${ks[@]}; do
            while read line; do
                sbatch batch.sh $line $k $clf results DeterministicSubspaceClassifier
            done < datasets.txt
        done
    done
done
