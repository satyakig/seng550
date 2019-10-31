#!/usr/bin/python
"""BigQuery I/O PySpark example."""
from pyspark.sql import SparkSession
from pyspark.sql.functions import isnan, when, count, col


spark = SparkSession \
  .builder \
  .master('yarn') \
  .appName('clean-financial-data') \
  .getOrCreate()

# Use the Cloud Storage bucket for temporary BigQuery export data used
# by the connector. This assumes the Cloud Storage connector for
# Hadoop is configured.
# bucket = spark.sparkContext._jsc.hadoopConfiguration().get(
#     'fs.gs.system.bucket')
# spark.conf.set('temporaryGcsBucket', bucket)

# Load data from BigQuery.
fundamentals = spark.read.format('bigquery') \
  .option('table', 'seng-550:seng_550_data.fundamentals') \
  .load()
fundamentals.createOrReplaceTempView('fundamentals')

# Perform Book Value aggregation.
avg_bvps = spark.sql(
    'SELECT Instrument, AVG(Book_Value_Per_Share) AS avg_BVPS FROM fundamentals GROUP BY Instrument')
avg_bvps.show()
avg_bvps.printSchema()

# Need to add check here that we are not running isnan on Date column,
# since isnan does not work for Timestamp type.
fundamentals.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in fundamentals.columns]).show()


# Saving the data to BigQuery
# word_count.write.format('bigquery') \
#   .option('table', 'wordcount_dataset.wordcount_output') \
#   .save()

# Run with
# --jars=gs://spark-lib/bigquery/spark-bigquery-latest.jar
