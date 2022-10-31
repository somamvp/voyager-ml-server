#!/bin/bash

# 도커 환경변수 설정
echo export docker=True >> ~/.bashrc
source ~/.bashrc

# KST 타임존 변경
ln -sf /usr/share/zoneinfo/Asia/Seoul /etc/localtime
echo 'Asia/Seoul' > /etc/timezone
date="$(date '+%y%m%d_%H:%M')"

# 로그 파일 경로 설정
#LOG_DIR=app/docker
LOG_DIR=app/runs
LOG_FILE=${LOG_DIR}/docker_${date}.log

# FastAPI 빌드
uvicorn app.main:app --reload --host 0.0.0.0 --port 9898 > $LOG_FILE 2>&1 &

# 로깅 시작
tail -F $LOG_FILE
