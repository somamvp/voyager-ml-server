#!/bin/zsh
PORT=8002 # ws:8000, dk:8001, jw:8002
BASEDIR=$(dirname "$0")
cd $BASEDIR/../

date="$(date '+%y%m%d_%H:%M')"
LOG_DIR=app/runs

# source ~/.zshrc
# source activate yolov7-env
conda info | grep "active environment"

CURRENT_PID=$(pgrep -f "port $PORT")

echo process info: ${CURRENT_PID}

if [ -z "$CURRENT_PID" ]; then
    echo "> 현재 구동 중인 애플리케이션이 없으므로 종료하지 않습니다."
else
    echo "> 기존에 구동 중인 서버를 종료합니다."
    kill -9 $(lsof -t -i:${PORT})
    sleep 1
fi

[ ! -d "runs" ] && mkdir runs

LOG_FILE=${LOG_DIR}/fast_${PORT}_${date}.log

nohup uvicorn app.main:app --reload --host 0.0.0.0 --port $PORT > $LOG_FILE 2>&1 &

tail -F $LOG_FILE

echo "server terminated!"
