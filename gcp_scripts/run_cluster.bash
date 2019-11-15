#!/bin/bash
# ./run_cluster.bash [python-code] [cluster-name] [region]
# ./run_cluster.bash ../data_cleaning/stats_clean_join.py seng550 us-central1

FILE_NAME=$1
CLUSTER_NAME=$2
CLUSTER_REGION=$3

gcloud dataproc jobs submit pyspark ${FILE_NAME} \
    --cluster ${CLUSTER_NAME} \
    --region ${CLUSTER_REGION} \
    --jars=gs://spark-lib/bigquery/spark-bigquery-latest.jar \
    --driver-log-levels root=FATAL
