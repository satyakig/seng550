#!/bin/bash
# ./create_cluster.bash [cluster-name] [region]
# ./create_cluster.bash seng550 us-central1

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
  --num-workers=5 \
  --master-machine-type=n1-highmem-4 \
  --master-boot-disk-type=pd-ssd \
  --master-boot-disk-size=100GB \
  --worker-machine-type=n1-highmem-4 \
  --worker-boot-disk-size=80GB \
  --worker-boot-disk-type=pd-ssd
