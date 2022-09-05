#!/bin/bash
date="$(date '+%y%m%d_%H:%M')"

source ~/.bashrc

BASEDIR=$(dirname "$0")
cd $BASEDIR

source activate server-gpu-env
conda info | grep "active environment"

if [ -z $1 ]; then 
    SERVER_SCRIPT='server_guide_only.py'
else 
    SERVER_SCRIPT=$1
fi

[ ! -d "runs" ] && mkdir runs
LOG_FILE=runs/fast_${date}.log

echo "running script ${SERVER_SCRIPT}"
python $SERVER_SCRIPT 1 > $LOG_FILE 2>&1 &

tail -F $LOG_FILE

echo "server terminated!"