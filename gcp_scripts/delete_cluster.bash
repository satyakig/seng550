#!/bin/bash
# ./delete_cluster.bash [cluster-name]
# ./delete_cluster.bash seng550

NAME=$1

gcloud dataproc clusters delete ${NAME} --region=us-central1
