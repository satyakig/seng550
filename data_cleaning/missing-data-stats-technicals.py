# Directly from BigQuery:
# SELECT SUM(case when int64_field_0 IS NULL then 1 end) as int64_field_0,
# SUM(case when Instrument IS NULL then 1 end) as Instrument_Count,
# SUM(case when Enterprise_Value_To_Sales__Daily_Time_Series_Ratio_ IS NULL then 1 end) as Enterprise_Value_To_Sales__Daily_Time_Series_Ratio_Count,
# SUM(case when P_E__Daily_Time_Series_Ratio_ IS NULL then 1 end) as P_E__Daily_Time_Series_Ratio_Count,
# SUM(case when Price IS NULL then 1 end) as Price_Count,
# SUM(case when Date IS NULL then 1 end) as Date_Count,
# SUM(case when Price_To_Book_Value_Per_Share__Daily_Time_Series_Ratio_ IS NULL then 1 end) as Price_To_Book_Value_Per_Share__Daily_Time_Series_Ratio_Count,
# SUM(case when Price_To_Cash_Flow_Per_Share__Daily_Time_Series_Ratio_ IS NULL then 1 end) as Price_To_Cash_Flow_Per_Share__Daily_Time_Series_Ratio_Count,
# SUM(case when Price_To_Sales_Per_Share__Daily_Time_Series_Ratio_ IS NULL then 1 end) as Price_To_Sales_Per_Share__Daily_Time_Series_Ratio_Count,
# SUM(case when Total_Debt_To_EBITDA__Daily_Time_Series_Ratio_ IS NULL then 1 end) as Total_Debt_To_EBITDA__Daily_Time_Series_Ratio_Count,
# SUM(case when Total_Debt_To_Enterprise_Value__Daily_Time_Series_Ratio_ IS NULL then 1 end) as Total_Debt_To_Enterprise_Value__Daily_Time_Series_Ratio_Count,
# SUM(case when Volume IS NULL then 1 end) as Volume_Count
# FROM seng_550_data.technicals


# gcloud dataproc jobs submit pyspark ./missing-data-stats-technicals.py \
#     --cluster satyaki-dataproc \
#     --jars gs://spark-lib/bigquery/spark-bigquery-latest.jar \
#     --driver-log-levels root=FATAL \
#     --region us-central1

from pyspark.sql import SparkSession

spark = SparkSession.builder.master('yarn').appName(
    'technicals-stats').getOrCreate()
bucket = spark.sparkContext._jsc.hadoopConfiguration().get('fs.gs.system.bucket')
spark.conf.set('temporaryGcsBucket', bucket)


technicals = spark.read.format('bigquery').option('table', 'seng-550.seng_550_data.technicals').load().cache()

technicals.createOrReplaceTempView('technicals')
technicals.printSchema()
num_rows = technicals.count()
print('Total Rows:  {}\n'.format(num_rows))

print('Overall Stats')
for key in technicals.columns:
    results = spark.sql('SELECT COUNT(*) as count FROM technicals WHERE {} IS NULL'.format(key))
    count = results.first()['count']
    print('{}:  {}, {:.5f}% missing'.format(key, count, float(count)/float(num_rows) * 100))

all_cols = '{} IS NULL'.format(technicals.columns[0])
for key in technicals.columns:
    if key != technicals.columns[0]:
        all_cols = all_cols + (' OR {} IS NULL'.format(key))
results = spark.sql('SELECT COUNT(*) as count FROM technicals WHERE {}'.format(all_cols))
count = results.first()['count']
print('One or more cols:  {}, {:.5f}% missing'.format(count, float(count) / float(num_rows) * 100))

results = spark.sql('SELECT COUNT(*) as count FROM technicals WHERE Date IS NULL OR Formatted_Date IS NULL OR Volume IS NULL OR Price IS NULL')
count = results.first()['count']
print('Date, Volume or Price is missing:  {}, {:.5f}% missing'.format(count, float(count) / float(num_rows) * 100))


# 1989 - 1999
total = spark.sql("SELECT * FROM technicals WHERE Formatted_Date >= cast('{}-01-01' as date) AND Formatted_Date <= cast('{}-12-31' as date)".format(1989, 1999))
print('\nYear {} - {}, {} rows, {:.5f}%'.format(1989, 1999, total.count(), float(total.count() / float(num_rows) * 100)))

for key in technicals.columns:
    results = spark.sql("SELECT COUNT(*) as count FROM technicals WHERE {} IS NULL AND Formatted_Date >= cast('{}-01-01' as date) AND Formatted_Date <= cast('{}-12-31' as date)".format(key, 1989, 1999))
    count = results.first()['count']
    print('{}:  {}, {:.5f}% missing'.format(key, count, float(count) / float(total.count()) * 100))

all_cols = '{} IS NULL'.format(technicals.columns[0])
for key in technicals.columns:
    if key != technicals.columns[0]:
        all_cols = all_cols + (' OR {} IS NULL'.format(key))
results = spark.sql("SELECT COUNT(*) as count FROM technicals WHERE {} AND Formatted_Date >= cast('{}-01-01' as date) AND Formatted_Date <= cast('{}-12-31' as date)".format(all_cols, 1989, 1999))
count = results.first()['count']
print('One or more cols:  {}, {:.5f}% missing'.format(count, float(count) / float(total.count()) * 100))

results = spark.sql("SELECT COUNT(*) as count FROM technicals WHERE Date IS NULL OR Formatted_Date IS NULL OR Volume IS NULL OR Price IS NULL AND Formatted_Date >= cast('{}-01-01' as date) AND Formatted_Date <= cast('{}-12-31' as date)".format(1989, 1999))
count = results.first()['count']
print('Date, Volume or Price is missing:  {}, {:.5f}% missing'.format(count, float(count) / float(total.count()) * 100))


# 2000 - 2004
total = spark.sql("SELECT * FROM technicals WHERE Formatted_Date >= cast('{}-01-01' as date) AND Formatted_Date <= cast('{}-12-31' as date)".format(2000, 2004))
print('\nYear {} - {}, {} rows, {:.5f}%'.format(2000, 2004, total.count(), float(total.count() / float(num_rows) * 100)))

for key in technicals.columns:
    results = spark.sql("SELECT COUNT(*) as count FROM technicals WHERE {} IS NULL AND Formatted_Date >= cast('{}-01-01' as date) AND Formatted_Date <= cast('{}-12-31' as date)".format(key, 2000, 2004))
    count = results.first()['count']
    print('{}:  {}, {:.5f}% missing'.format(key, count, float(count) / float(total.count()) * 100))

all_cols = '{} IS NULL'.format(technicals.columns[0])
for key in technicals.columns:
    if key != technicals.columns[0]:
        all_cols = all_cols + (' OR {} IS NULL'.format(key))
results = spark.sql("SELECT COUNT(*) as count FROM technicals WHERE {} AND Formatted_Date >= cast('{}-01-01' as date) AND Formatted_Date <= cast('{}-12-31' as date)".format(all_cols, 2000, 2004))
count = results.first()['count']
print('One or more cols:  {}, {:.5f}% missing'.format(count, float(count) / float(total.count()) * 100))

results = spark.sql("SELECT COUNT(*) as count FROM technicals WHERE Date IS NULL OR Formatted_Date IS NULL OR Volume IS NULL OR Price IS NULL AND Formatted_Date >= cast('{}-01-01' as date) AND Formatted_Date <= cast('{}-12-31' as date)".format(2000, 2004))
count = results.first()['count']
print('Date, Volume or Price is missing:  {}, {:.5f}% missing'.format(count, float(count) / float(total.count()) * 100))


# 2005 - 2009
total = spark.sql("SELECT * FROM technicals WHERE Formatted_Date >= cast('{}-01-01' as date) AND Formatted_Date <= cast('{}-12-31' as date)".format(2005, 2009))
print('\nYear {} - {}, {} rows, {:.5f}%'.format(2005, 2009, total.count(), float(total.count() / float(num_rows) * 100)))

for key in technicals.columns:
    results = spark.sql("SELECT COUNT(*) as count FROM technicals WHERE {} IS NULL AND Formatted_Date >= cast('{}-01-01' as date) AND Formatted_Date <= cast('{}-12-31' as date)".format(key, 2005, 2009))
    count = results.first()['count']
    print('{}:  {}, {:.5f}% missing'.format(key, count, float(count) / float(total.count()) * 100))

all_cols = '{} IS NULL'.format(technicals.columns[0])
for key in technicals.columns:
    if key != technicals.columns[0]:
        all_cols = all_cols + (' OR {} IS NULL'.format(key))
results = spark.sql("SELECT COUNT(*) as count FROM technicals WHERE {} AND Formatted_Date >= cast('{}-01-01' as date) AND Formatted_Date <= cast('{}-12-31' as date)".format(all_cols, 2005, 2009))
count = results.first()['count']
print('One or more cols:  {}, {:.5f}% missing'.format(count, float(count) / float(total.count()) * 100))

results = spark.sql("SELECT COUNT(*) as count FROM technicals WHERE Date IS NULL OR Formatted_Date IS NULL OR Volume IS NULL OR Price IS NULL AND Formatted_Date >= cast('{}-01-01' as date) AND Formatted_Date <= cast('{}-12-31' as date)".format(2005, 2009))
count = results.first()['count']
print('Date, Volume or Price is missing:  {}, {:.5f}% missing'.format(count, float(count) / float(total.count()) * 100))


# 2010 - 2014
total = spark.sql("SELECT * FROM technicals WHERE Formatted_Date >= cast('{}-01-01' as date) AND Formatted_Date <= cast('{}-12-31' as date)".format(2010, 2014))
print('\nYear {} - {}, {} rows, {:.5f}%'.format(2010, 2014, total.count(), float(total.count() / float(num_rows) * 100)))

for key in technicals.columns:
    results = spark.sql("SELECT COUNT(*) as count FROM technicals WHERE {} IS NULL AND Formatted_Date >= cast('{}-01-01' as date) AND Formatted_Date <= cast('{}-12-31' as date)".format(key, 2010, 2014))
    count = results.first()['count']
    print('{}:  {}, {:.5f}% missing'.format(key, count, float(count) / float(total.count()) * 100))

all_cols = '{} IS NULL'.format(technicals.columns[0])
for key in technicals.columns:
    if key != technicals.columns[0]:
        all_cols = all_cols + (' OR {} IS NULL'.format(key))
results = spark.sql("SELECT COUNT(*) as count FROM technicals WHERE {} AND Formatted_Date >= cast('{}-01-01' as date) AND Formatted_Date <= cast('{}-12-31' as date)".format(all_cols, 2010, 2014))
count = results.first()['count']
print('One or more cols:  {}, {:.5f}% missing'.format(count, float(count) / float(total.count()) * 100))

results = spark.sql("SELECT COUNT(*) as count FROM technicals WHERE Date IS NULL OR Formatted_Date IS NULL OR Volume IS NULL OR Price IS NULL AND Formatted_Date >= cast('{}-01-01' as date) AND Formatted_Date <= cast('{}-12-31' as date)".format(2010, 2014))
count = results.first()['count']
print('Date, Volume or Price is missing:  {}, {:.5f}% missing'.format(count, float(count) / float(total.count()) * 100))


# 2015 - 2019
total = spark.sql("SELECT * FROM technicals WHERE Formatted_Date >= cast('{}-01-01' as date) AND Formatted_Date <= cast('{}-12-31' as date)".format(2015, 2019))
print('\nYear {} - {}, {} rows, {:.5f}%'.format(2015, 2019, total.count(), float(total.count() / float(num_rows) * 100)))

for key in technicals.columns:
    results = spark.sql("SELECT COUNT(*) as count FROM technicals WHERE {} IS NULL AND Formatted_Date >= cast('{}-01-01' as date) AND Formatted_Date <= cast('{}-12-31' as date)".format(key, 2015, 2019))
    count = results.first()['count']
    print('{}:  {}, {:.5f}% missing'.format(key, count, float(count) / float(total.count()) * 100))

all_cols = '{} IS NULL'.format(technicals.columns[0])
for key in technicals.columns:
    if key != technicals.columns[0]:
        all_cols = all_cols + (' OR {} IS NULL'.format(key))
results = spark.sql("SELECT COUNT(*) as count FROM technicals WHERE {} AND Formatted_Date >= cast('{}-01-01' as date) AND Formatted_Date <= cast('{}-12-31' as date)".format(all_cols, 2015, 2019))
count = results.first()['count']
print('One or more cols:  {}, {:.5f}% missing'.format(count, float(count) / float(total.count()) * 100))

results = spark.sql("SELECT COUNT(*) as count FROM technicals WHERE Date IS NULL OR Formatted_Date IS NULL OR Volume IS NULL OR Price IS NULL AND Formatted_Date >= cast('{}-01-01' as date) AND Formatted_Date <= cast('{}-12-31' as date)".format(2015, 2019))
count = results.first()['count']
print('Date, Volume or Price is missing:  {}, {:.5f}% missing'.format(count, float(count) / float(total.count()) * 100))
