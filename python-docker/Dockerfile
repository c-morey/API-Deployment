FROM ubuntu:20.04

RUN apt-get update -y
RUN apt-get install -y python3-pip python-dev build-essential
RUN mkdir -p /app

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

COPY . .

EXPOSE ${PORT}

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]


