## How to run on Google Dataproc
1. Open the Google Cloud Shell
2. Run this command:
```
gcloud dataproc jobs submit pyspark --cluster ${cluster_name} \
    --jars gs://spark-lib/bigquery/spark-bigquery-latest.jar \
    --region us-central1 \
    gs://seng550/data_cleaning_code/find_missing_data_stats.py
```