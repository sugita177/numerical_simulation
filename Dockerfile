FROM python:3.13

COPY requirements.txt .

RUN apt-get update
RUN apt-get install -y python3-tk
RUN apt install x11-apps -y
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

WORKDIR /src