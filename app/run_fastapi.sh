#!/bin/zsh

BASEDIR=$(dirname "$0")
cd $BASEDIR/../

date="$(date '+%y%m%d_%H:%M')"
LOG_DIR=app/runs

# source ~/.zshrc
# source activate yolov7-env
conda info | grep "active environment"

if [ -z $1 ]; then 
    PORT=8001
else 
    PORT=$1
fi

CURRENT_PID=$(pgrep -f "port $PORT")
echo process info: ${CURRENT_PID}

if [ -z "$CURRENT_PID" ]; then
    echo "> 현재 구동 중인 애플리케이션이 없으므로 종료하지 않습니다."
else
    echo "> 기존에 구동 중인 서버를 종료합니다."
    kill -15 $CURRENT_PID
    sleep 1
fi

[ ! -d "runs" ] && mkdir runs

LOG_FILE=${LOG_DIR}/fast_${PORT}_${date}.log

nohup uvicorn app.main:app --reload --host 0.0.0.0 --port $PORT > $LOG_FILE 2>&1 &

tail -F $LOG_FILE

echo "server terminated!"