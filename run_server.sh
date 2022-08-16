#!/bin/bash

# export PATH="${PATH}:/home/ubuntu/miniconda3/bin:/home/ubuntu/miniconda3/condabin:"
source ~/miniconda3/etc/profile.d/conda.sh

BASEDIR=$(dirname "$0")
cd $BASEDIR

conda activate server-gpu-env
conda info | grep "active environment"

[ ! -d "runs" ] && mkdir runs
python server.py 1 > runs/run.log 2>&1

echo "server terminated!"