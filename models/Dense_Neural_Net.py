"""
Run:  bash ../gcp_scripts/bash_scripts_run_cluster.bash ./Linear_Regressor.py james-cluster us-central1

Checkout put of this job later: [8f3e6d5f78024541bfa107d4fccd54e7]
"""
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

from keras import models
from keras import layers
from elephas.ml_model import ElephasEstimator

# Constants
DATABASE = "seng-550.seng_550_data"
TABLE = "combined_technicals_fundamentals"

# Create the Spark Session on this node
spark = SparkSession.builder.master('yarn').appName(
    'combined_fund').getOrCreate()

# Create a temporary bucket
bucket = spark.sparkContext._jsc.hadoopConfiguration().get('fs.gs.system.bucket')
spark.conf.set('temporaryGcsBucket', bucket)

# Load data from Big Query
combined_data = spark.read.format('bigquery').option('table', "{}.{}".format(DATABASE, TABLE)).load().cache()
combined_data.createOrReplaceTempView('combined_data')

# Start - 2017 rows
# Look at Facebook ordered by year and quarter
training_data = spark.sql(
    'SELECT * FROM combined_data WHERE Instrument="FB.O" AND Year < 2018'
)
print(training_data.count)

# 2018 - now
test_data = spark.sql(
    'SELECT * FROM combined_data WHERE Instrument="FB.O" AND Year >= 2018'
)
print(training_data.count)
