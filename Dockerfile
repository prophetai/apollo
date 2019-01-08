FROM ubuntu:latest
MAINTAINER "Dudley Díaz <deds15@gmail.com>"

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

#Install needed libraries and credentials
RUN pip3 install -r requirements.txt

#Assign the editor for the VM
RUN export EDITOR=vim

#Establish the working directory
WORKDIR /home/apollo

# Copy the files to running the models and the models

COPY apollo ./
