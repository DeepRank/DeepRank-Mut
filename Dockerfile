FROM python:3.8

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app
COPY requirements /usr/src/app/

RUN pip install --no-cache-dir -r requirements

RUN apt update && apt install openmpi-bin openmpi-common openssh-client openssh-server libopenmpi1.10 libopenmpi-dev

COPY . /usr/src/app
