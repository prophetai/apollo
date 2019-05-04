FROM ubuntu:latest
LABEL maintainer "Dudley Díaz <deds15@gmail.com>"

ENV LANG C.UTF-8
RUN apt-get update -y && apt-get -y install cron
RUN apt-get -y install vim
RUN apt-get -y install python
RUN apt-get -y install xfonts-75dpi
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install build-essential xorg libssl-dev libxrender-dev wget
RUN wget https://downloads.wkhtmltopdf.org/0.12/0.12.5/wkhtmltox_0.12.5-1.bionic_amd64.deb && \
    dpkg -i wkhtmltox_0.12.5-1.bionic_amd64.deb && \
    apt-get install -f

RUN alias python=python
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
