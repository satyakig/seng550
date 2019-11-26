'''
Gets the average year over year return of each company broken down by quarter.
Year of year return is found by calculating the average stock price for the
quarter and looking at the average price from the previous quarter a year ago.

This data is then combined with the k_means clustering results and written to
BigQuery. This allows average results of the different clusters to be analyzed.
'''
from pyspark.sql import SparkSession
from pyspark.sql import Window
import pyspark.sql.functions as func

# Constants
DATABASE = 'seng-550.seng_550_data'
TECH_TABLE = 'technicals_cleaned'
KMEANS_TABLE = 'k_means_data'

# Create the Spark Session on this node
spark = SparkSession.builder.master('yarn').appName(
    'binning_k_means').getOrCreate()

# Use the Cloud Storage bucket for temporary BigQuery export data used
# by the connector. This assumes the Cloud Storage connector for
# Hadoop is configured.
bucket = spark.sparkContext._jsc.hadoopConfiguration().get(
    'fs.gs.system.bucket')
spark.conf.set('temporaryGcsBucket', bucket)

# Load data from Big Query
tech_data = spark.read.format('bigquery').option('table', '{}.{}'.format(
    DATABASE, TECH_TABLE)).load().cache()
tech_data.createOrReplaceTempView('tech_data')

print('Read in data: ')
tech_data.show()

# Get average price for each Instrument for a given year and quarter.
#partitioning_cols = ['Instrument', 'Year', 'Quarter']
avg_price_per_quarter = tech_data.select(
  ['Instrument', 'Year', 'Quarter', 'Price_Close'])\
  .groupBy(['Instrument', 'Year', 'Quarter']).avg()

avg_price_per_quarter = avg_price_per_quarter.select(
  'Instrument', 'Year', 'Quarter', func.col('avg(Price_Close)').alias('avg_price_close')
).orderBy('Instrument', 'Year', 'Quarter')

print('Grouped by data: ')
avg_price_per_quarter.show()

# Create window function to get year over year price change.
window = Window.partitionBy('Instrument').orderBy(['Year','Quarter'])

# Get price from previous year. Data is grouped by quarter, so 4 quarters ago
# is equivalent to 1 year ago.
avg_price_per_quarter = avg_price_per_quarter.withColumn(
  'prev_year_price',
  func.lag(col=avg_price_per_quarter['avg_price_close'], count=4).over(window)
)

avg_price_per_quarter = avg_price_per_quarter.withColumn(
  'yearly_return',
  100*(avg_price_per_quarter['avg_price_close'] - \
    avg_price_per_quarter['prev_year_price']) \
    /avg_price_per_quarter['prev_year_price']
)

print('Average year over year return broken down by Instrument and Quarter: ')
avg_price_per_quarter.show()

# Load data from Big Query
kmeans_data = spark.read.format('bigquery').option('table', '{}.{}'.format(
    DATABASE, KMEANS_TABLE)).load().cache()
kmeans_data.createOrReplaceTempView('kmeans_data')

# Join kmeans_data with avg_price_per_quarter table.
binnable_data = kmeans_data.join(avg_price_per_quarter,
  ['Instrument', 'Year', 'Quarter'])

print('Binnable data: ')
binnable_data.show()

# Write data to bigquery for further analysis.
print('Write binnable_data to bigquery.')
binnable_data.write.format('bigquery') \
  .option('table', DATABASE + '.clustered_data_with_historical_returns') \
  .mode('overwrite') \
  .save()
