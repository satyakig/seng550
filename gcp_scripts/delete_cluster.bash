#!/bin/bash
# ./delete_cluster.bash [cluster-name] [region]
# ./delete_cluster.bash seng550 us-central1

NAME=$1
REGION=$2

gcloud dataproc clusters delete ${NAME} --region=${REGION}
