#!/bin/bash

ks=(5 10 15 20 25 30 35 40 45 50)

for k in ${ks[@]}; do
    while read line; do
        sbatch batch.sh $line $k
    done < datasets.txt
done
