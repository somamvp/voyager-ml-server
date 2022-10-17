FROM nvidia/cuda:11.7.0-base-ubuntu20.04

WORKDIR /code

COPY /app /code/app

COPY ./requirements.txt /code/requirements.txt

RUN apt update && apt upgrade -y
RUN apt install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa -y
# python 3.10 불안정한 것 같아서 (torchvision이랑 안맞는듯)
RUN apt install python3.9 -y
RUN apt install python3-pip -y
RUN pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
RUN apt-get -y install libgl1-mesa-glx

# CMD ["uvicorn", "app:main:app", "--host", "0.0.0.0", "--port", "9898"]
