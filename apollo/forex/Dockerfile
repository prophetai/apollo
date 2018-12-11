FROM ubuntu:latest
MAINTAINER deds15@gmail.com

RUN apt-get update
RUN apt-get -y install cron
RUN apt-get -y install vim
RUN apt-get -y install python3
RUN alias python=python3
RUN apt-get -y install python-pip
RUN apt install -y python3-pip
RUN alias pip=pip3
RUN apt install -y git
COPY requirements.txt ./
RUN pip3 install -r requirements.txt

RUN export EDITOR=vim

#stablish the working directory
WORKDIR /home/forex

# Copy the files to running the models and the models
COPY models ./models
COPY email ./email
COPY src ./
