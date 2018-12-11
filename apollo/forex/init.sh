#!/bin/bash

docker build -t gcr.io/${PROJECT_ID}/forex_update:latest .
docker run -d forex_update:latest
