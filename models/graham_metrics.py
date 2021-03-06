# This script is used to filter fundamentals data to create a new table called test-table
# The new table contains the scores for different companies about the different metrics
# The metrics that are followed are graham metrics

from pyspark.sql import SparkSession
from pyspark.sql.functions import year, month, sum as _sum, abs as _abs, avg, first, last, lit, max as _max

# Constants
DATABASE = 'seng-550.seng_550_data'
FUND_TABLE = 'fundamentals_cleaned'
TECH_TABLE = 'technicals_cleaned'

# Create the Spark Session on this node
spark = SparkSession \
    .builder \
    .master('yarn') \
    .appName('graham_metrics') \
    .config('spark.executor.cores', '16') \
    .config('spark.executor.memory', '71680m') \
    .config('spark.executorEnv.LD_PRELOAD', 'libnvblas.so') \
    .getOrCreate()

# Use the Cloud Storage bucket for temporary BigQuery export data used
# by the connector. This assumes the Cloud Storage connector for
# Hadoop is configured.
bucket = spark.sparkContext._jsc.hadoopConfiguration().get('fs.gs.system.bucket')
spark.conf.set('temporaryGcsBucket', bucket)

# Load data from Big Query
fund_data = spark.read.format('bigquery').option('table', '{}.{}'.format(DATABASE, FUND_TABLE)).load().cache()
fund_data.createOrReplaceTempView('fund_data')

tech_data = spark.read.format('bigquery').option('table', '{}.{}'.format(DATABASE, TECH_TABLE)).load().cache()
tech_data.createOrReplaceTempView('tech_data')


# Metric 1: Companies with revenue over 5 million
revenue_over_5mil = fund_data\
    .groupBy('Instrument', year('Date').alias('actual_year'))\
    .agg(_sum('Total_Revenue').alias('Revenue'))\
    .orderBy('Instrument', 'actual_year')\
    .filter('actual_year == 2019')\
    .filter('Revenue > 5000000')

revenue_over_5mil.show()
revenue_over_5mil = revenue_over_5mil.withColumn('metric1', lit(1))\
    .drop('actual_year', 'Revenue')\
    .orderBy('Instrument')
revenue_over_5mil.show()


# Metric 2: Current asset twice current liabilities
asset_twice_liabilities = fund_data\
    .groupBy('Instrument', year('Date').alias('actual_year'))\
    .agg(avg('Total_Current_Assets').alias('Assets'),
         avg('Total_Current_Liabilities').alias('Liabilities'))\
    .orderBy('Instrument', 'actual_year')\
    .filter('actual_year == 2019')\
    .filter('Assets >= 2 * Liabilities')

asset_twice_liabilities.show()
asset_twice_liabilities = asset_twice_liabilities.withColumn('metric2', lit(1))\
    .drop('actual_year', 'Assets', 'Liabilities')\
    .orderBy('Instrument')
asset_twice_liabilities.show()


# Metric 3: Total working capital is more than long term debt
capital_more_than_debt = fund_data\
    .groupBy('Instrument', year('Date').alias('actual_year'))\
    .agg(avg('Total_Current_Assets').alias('Assets'),
         avg('Total_Current_Liabilities').alias('Liabilities'),
         avg('Total_Long_Term_Debt').alias('Debt'))\
    .orderBy('Instrument', 'actual_year')\
    .filter('actual_year == 2019')\
    .filter('Assets - Liabilities  > Debt')

capital_more_than_debt.show()
capital_more_than_debt = capital_more_than_debt.withColumn('metric3', lit(1))\
    .drop('actual_year', 'Assets', 'Liabilities', 'Debt')\
    .orderBy('Instrument')
capital_more_than_debt.show()


# Metric 4: Positive earning for past 5 years
# Filtering the companies which have positive earnings and then removing
# any company which has negative earnings in last 10 years
negative_earnings = fund_data\
    .groupBy('Instrument', year('Date').alias('actual_year'))\
    .agg(_sum('Normalized_Income_Avail_to_Cmn_Shareholders').alias('Earnings'))\
    .filter('actual_year >= 2015')\
    .filter('Earnings < 0')\
    .select('Instrument').distinct().orderBy('Instrument')

positive_earnings = fund_data\
    .groupBy('Instrument', year('Date').alias('actual_year'))\
    .agg(avg('Normalized_Income_Avail_to_Cmn_Shareholders').alias('Earnings'))\
    .orderBy('Instrument', 'actual_year')\
    .filter('actual_year >= 2015')\
    .filter('Earnings > 0')
positive_earnings = positive_earnings.join(negative_earnings, 'Instrument', 'leftanti')

positive_earnings.show()
positive_earnings = positive_earnings.withColumn('metric4', lit(1))\
    .drop('actual_year', 'Earnings')\
    .groupBy('Instrument')\
    .agg(_max('metric4').alias('metric4'))\
    .orderBy('Instrument')
positive_earnings.show()


# Metric 5: Dividend payment for past 5 years
no_dividend = fund_data\
    .groupBy('Instrument', year('Date').alias('actual_year'))\
    .agg((_sum('Gross_Dividends___Common_Stock') / avg('Total_Common_Shares_Outstanding')).alias('Dividend')) \
    .filter('actual_year >= 2015')\
    .filter('Dividend == 0')\
    .select('Instrument').distinct().orderBy('Instrument')

dividend = fund_data\
    .groupBy('Instrument', year('Date').alias('actual_year'))\
    .agg((_sum('Gross_Dividends___Common_Stock') / avg('Total_Common_Shares_Outstanding')).alias('Dividend')) \
    .orderBy('Instrument', 'actual_year') \
    .filter('actual_year >= 2015')\
    .filter('Dividend > 0')
dividend = dividend.join(no_dividend, 'Instrument', 'leftanti')

dividend.show()
dividend = dividend.withColumn('metric5', lit(1))\
    .drop('actual_year', 'Dividend')\
    .groupBy('Instrument') \
    .agg(_max('metric5').alias('metric5')) \
    .orderBy('Instrument')
dividend.show()


# Metric 6: Earnings increased by 1/3rd in the last 10 years
increased_earnings = fund_data\
    .groupBy('Instrument', year('Date').alias('actual_year'))\
    .agg(_sum('Normalized_Income_Avail_to_Cmn_Shareholders').alias('Earnings'))\
    .filter('actual_year == 2009 or actual_year == 2019')\
    .na.fill(0)\
    .orderBy('Instrument', 'actual_year')\
    .groupBy('Instrument')\
    .agg(((last('Earnings') - first('Earnings')) / _abs(first('Earnings'))).alias('Earnings_increase'))\
    .filter('Earnings_increase >= 0.33')\
    .orderBy('Instrument')

increased_earnings.show()
increased_earnings = increased_earnings.withColumn('metric6', lit(1))\
    .drop('actual_year', 'Earnings_increase')\
    .orderBy('Instrument')
increased_earnings.show()


# Getting the latest price for each instrument
latest_close_price = tech_data\
    .groupBy('Instrument', year('Date').alias('actual_year'), month('Date').alias('month'))\
    .agg(last('Price_Close').alias('Price'))\
    .orderBy('Instrument', 'actual_year', 'month')\
    .filter('actual_year == 2019')\
    .groupBy('Instrument')\
    .agg(last('Price').alias('Price'))\
    .orderBy('Instrument')
latest_close_price.show()


# Merging latest price with fund data
fund_data_with_price = fund_data.join(latest_close_price, 'Instrument', 'full')

# Metric 7: Current Price <= 15 * Earnings Per Share
earnings_15_eps = fund_data_with_price\
    .groupBy('Instrument', year('Date').alias('actual_year'))\
    .agg((_sum('Normalized_Income_Avail_to_Cmn_Shareholders') / avg('Total_Common_Shares_Outstanding')).alias('EPS'),
         last('Price').alias('Price')) \
    .orderBy('Instrument', 'actual_year') \
    .filter('actual_year == 2019')\
    .filter('Price <= 15 * EPS')

earnings_15_eps.show()
earnings_15_eps = earnings_15_eps.withColumn('metric7', lit(1))\
    .drop('actual_year', 'EPS', 'Price')\
    .orderBy('Instrument')
earnings_15_eps.show()


# Metric 8: Current value <= 1.5 Book value
current_value = fund_data_with_price\
    .groupBy('Instrument', year('Date').alias('actual_year'))\
    .agg(avg('Total_Common_Shares_Outstanding').alias('Shares_Outstanding'),
         last('Price').alias('Price'),
         avg('Book_Value_Per_Share').alias('Book_Value_Per_Share')) \
    .orderBy('Instrument', 'actual_year') \
    .filter('actual_year == 2019')\
    .filter('(Price * Shares_Outstanding) <= (1.5 * Book_Value_Per_Share * Shares_Outstanding)')

current_value.show()
current_value = current_value.withColumn('metric8', lit(1))\
    .drop('actual_year', 'Shares_Outstanding', 'Price', 'Book_Value_Per_Share')\
    .orderBy('Instrument')
current_value.show()


# Join all the metrics tables to form one table
joined = revenue_over_5mil.join(asset_twice_liabilities, 'Instrument', 'full')\
    .join(capital_more_than_debt, 'Instrument', 'full')\
    .join(positive_earnings, 'Instrument', 'full')\
    .join(dividend, 'Instrument', 'full')\
    .join(increased_earnings, 'Instrument', 'full')\
    .join(earnings_15_eps, 'Instrument', 'full')\
    .join(current_value, 'Instrument', 'full')\
    .orderBy('Instrument')\
    .na.fill(0)
joined.show()


# Write the joined table to big query
joined.write.format('bigquery') \
  .option('table', DATABASE + '.graham_metrics') \
  .mode('overwrite') \
  .save()
