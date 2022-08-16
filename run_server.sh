#!/bin/bash

# export PATH="${PATH}:/home/ubuntu/miniconda3/bin:/home/ubuntu/miniconda3/condabin:"
source ~/miniconda3/etc/profile.d/conda.sh

BASEDIR=$(dirname "$0")
cd $BASEDIR

conda activate server-gpu-env
conda info | grep "active environment"

if [ -z $1 ]; then 
    SERVER_SCRIPT='server.py'
else 
    SERVER_SCRIPT=$1
fi

[ ! -d "runs" ] && mkdir runs
echo "running script ${SERVER_SCRIPT}"
python $SERVER_SCRIPT 1 > runs/$(date +%y-%d-%m_%T).log 2>&1

echo "server terminated!"