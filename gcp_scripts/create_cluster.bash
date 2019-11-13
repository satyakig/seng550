#!/bin/bash

NAME=$1
REGION=$2

gcloud dataproc clusters create ${NAME} \
  --scopes=cloud-platform \
  --region=${REGION} \
  --bucket=pyspark_bucket \
  --max-idle=1h \
  --image-version=1.4 \
  --optional-components=ANACONDA,JUPYTER \
  --num-masters=1 \
  --num-workers=3 \
  --master-machine-type=n1-standard-2 \
  --worker-machine-type=n1-standard-2 \
  --master-boot-disk-size=250GB \
  --worker-boot-disk-size=449GB 
