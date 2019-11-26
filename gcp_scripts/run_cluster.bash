#!/bin/bash
# ./run_cluster.bash [python-code] [cluster-name]
# ./run_cluster.bash ../data_cleaning/stats_clean_join.py seng550

FILE_NAME=$1
CLUSTER_NAME=$2

gcloud dataproc jobs submit pyspark ${FILE_NAME} \
    --cluster ${CLUSTER_NAME} \
    --region us-central1 \
    --jars=gs://spark-lib/bigquery/spark-bigquery-latest.jar \
    --driver-log-levels root=FATAL
