FROM python:3.10

WORKDIR /code

COPY /app /code/app

COPY ./requirements.txt /code/requirements.txt

# COPY . /build
RUN apt-get update
RUN pip3 install --no-cache-dir --upgrade -r /code/requirements.txt
RUN apt-get -y install libgl1-mesa-glx
# COPY . /code/app

# CMD ["uvicorn", "app:main:app", "--host", "0.0.0.0", "--port", "9898"]
