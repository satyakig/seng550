#!/bin/bash

NAME=$1
REGION=$2

gcloud dataproc clusters delete ${NAME} --region=${REGION}
