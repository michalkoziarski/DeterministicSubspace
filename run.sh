#!/bin/bash

while read line; do
    sbatch batch.sh $line
done < datasets.txt
