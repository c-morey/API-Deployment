FROM ubuntu:20.04

RUN apt-get update -y
RUN apt-get install -y python3-pip python-dev build-essential
RUN mkdir -p /app

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

COPY . /app

CMD [ "python3", "app.py"]


