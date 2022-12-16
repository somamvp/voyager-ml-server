FROM nvcr.io/nvidia/l4t-pytorch:r32.7.1-pth1.9-py3

WORKDIR /code
#PIP 패키지가 추가되면 여기서 카피해야 함.
# COPY /app /code/app 
RUN apt update && apt upgrade -y
RUN apt install software-properties-common -y && apt install vim -y
RUN add-apt-repository ppa:deadsnakes/ppa -y
# python 3.10 불안정한 것 같아서 (torchvision이랑 안맞는듯)
# RUN apt install python3.9 -y
# RUN apt install python3-pip -y
# RUN pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
RUN apt-get -y install libgl1-mesa-glx

# 도커 캐시 효율을 위해 수정
RUN pip3 install --upgrade pip
COPY ./requirements.txt /code/requirements.txt
RUN pip3 install --no-cache-dir --upgrade -r /code/requirements.txt
# COPY ./run_fastapi_docker.sh /code/run_fastapi_docker.sh
COPY /app /code/app
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9898"]
CMD ["python3", "-m", "app.main"]