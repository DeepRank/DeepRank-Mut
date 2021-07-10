FROM python:3.8

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app
COPY requirements /usr/src/app/

RUN apt update && apt install -y openmpi-bin openmpi-common openssh-client openssh-server libopenmpi-dev

RUN pip install --no-cache-dir -r requirements

COPY . /usr/src/app
