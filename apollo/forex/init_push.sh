#!/bin/bash

docker build -t gcr.io/${PROJECT_ID}/forex_cron:latest .
gcloud docker -- push gcr.io/${PROJECT_ID}/forex_cron:latest
