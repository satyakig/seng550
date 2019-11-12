"""
Run:  bash ../gcp_scripts/bash_scripts_run_cluster.bash ./Linear_Regressor.py james-cluster us-central1
"""
from pyspark.sql import SparkSession

# Constants
DATABASE = "seng-550.seng_550_data"
TABLE = "combined_fund_tech_2"

# Create the Spark Session on this node
spark = SparkSession.builder.master('yarn').appName(
    'combined_fund').getOrCreate()

# Create a temporary bucket
bucket = spark.sparkContext._jsc.hadoopConfiguration().get('fs.gs.system.bucket')
spark.conf.set('temporaryGcsBucket', bucket)

# Load data from Big Query
combined_data = spark.read.format('bigquery').option('table', "{}.{}".format(DATABASE, TABLE)).load().cache()
combined_data.createOrReplaceTempView('combined_data')

# Look at Facebook ordered by year and quarter
fb_year = spark.sql(
    'SELECT * FROM combined_data WHERE Instrument="FB.O" ORDER BY Year ASC, QUARTER ASC'
)
fb_year.show()

