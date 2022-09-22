#!/bin/zsh

CURRENT_PID=$(pgrep -f server_fast)
date="$(date '+%y%m%d_%H:%M')"
LOG_DIR=runs
echo process info: ${CURRENT_PID}

source ~/.zshrc

BASEDIR=$(dirname "$0")
cd $BASEDIR

source activate yolov7-env
conda info | grep "active environment"

if [ -z $1 ]; then 
    SERVER_SCRIPT='server_fast'
else 
    SERVER_SCRIPT=$1
fi

if [ -z "$CURRENT_PID" ]; then
    echo "> 현재 구동 중인 애플리케이션이 없으므로 종료하지 않습니다."
else
    echo "> 기존에 구동 중인 서버를 종료합니다."
    kill -15 $CURRENT_PID
    sleep 1
fi

[ ! -d "runs" ] && mkdir runs

pypath=/home/$USER/anaconda3/envs/yolov7-env/bin/python
echo "running script ${SERVER_SCRIPT}"

${pypath} -m uvicorn ${SERVER_SCRIPT}:app --reload --host 0.0.0.0 > ${LOG_DIR}/fast_$date.log 2>&1 &

# python $SERVER_SCRIPT 1 > $LOG_FILE 2>&1 &

tail -F ${LOG_DIR}/fast_$date.log

echo "server terminated!"