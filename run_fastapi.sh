#!/bin/bash

source ~/.bashrc

BASEDIR=$(dirname "$0")
cd $BASEDIR

source activate server-gpu-env
conda info | grep "active environment"

if [ -z $1 ]; then 
    SERVER_SCRIPT='server_fast'
else 
    SERVER_SCRIPT=$1
fi

[ ! -d "runs" ] && mkdir runs
LOG_FILE=runs/$(date +%y-%d-%m_%T).log

echo "running script ${SERVER_SCRIPT}"

uvicorn $SERVER_SCRIPT:app --reload --host 0.0.0.0

# python $SERVER_SCRIPT 1 > $LOG_FILE 2>&1 &

# tail -F $LOG_FILE

echo "server terminated!"