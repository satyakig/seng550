from pyspark.sql import SparkSession

spark = SparkSession.builder.master('yarn').appName(
    'technicals-stats').getOrCreate()
bucket = spark.sparkContext._jsc.hadoopConfiguration().get(
    'fs.gs.system.bucket')
spark.conf.set('temporaryGcsBucket', bucket)


technicals = spark.read.format('bigquery').option(
    'table', 'seng-550.seng_550_data.technicals').load().cache()

technicals.createOrReplaceTempView('technicals')

num_rows = technicals.count()
print('Total Rows:  {}\n'.format(num_rows))

for key in technicals.columns:
    results = spark.sql(
        'SELECT COUNT(*) as count FROM technicals WHERE {} IS NULL'.format(key))
    count = results.first()['count']
    print('{}:  {}, {:.2f}% missing'.format(
        key, count, float(count)/float(num_rows) * 100))

all_cols = '{} IS NULL'.format(technicals.columns[0])
for key in technicals.columns:
    if key != technicals.columns[0]:
        all_cols = all_cols + (' OR {} IS NULL'.format(key))
results = spark.sql(
    'SELECT COUNT(*) as count FROM technicals WHERE {}'.format(all_cols))
count = results.first()['count']
print('One or more cols:  {}, {:.2f}% missing'.format(
    count, float(count)/float(num_rows) * 100))

# Directly from BigQuery:
# SELECT SUM(case when int64_field_0 IS NULL then 1 end) as int64_field_0,
# SUM(case when Instrument IS NULL then 1 end) as Instrument_Count,
# SUM(case when Enterprise_Value_To_Sales__Daily_Time_Series_Ratio_ IS NULL then 1 end) as Enterprise_Value_To_Sales__Daily_Time_Series_Ratio_Count,
# SUM(case when P_E__Daily_Time_Series_Ratio_ IS NULL then 1 end) as P_E__Daily_Time_Series_Ratio_Count,
# SUM(case when Price_Close IS NULL then 1 end) as Price_Close_Count,
# SUM(case when Date IS NULL then 1 end) as Date_Count,
# SUM(case when Price_To_Book_Value_Per_Share__Daily_Time_Series_Ratio_ IS NULL then 1 end) as Price_To_Book_Value_Per_Share__Daily_Time_Series_Ratio_Count,
# SUM(case when Price_To_Cash_Flow_Per_Share__Daily_Time_Series_Ratio_ IS NULL then 1 end) as Price_To_Cash_Flow_Per_Share__Daily_Time_Series_Ratio_Count,
# SUM(case when Price_To_Sales_Per_Share__Daily_Time_Series_Ratio_ IS NULL then 1 end) as Price_To_Sales_Per_Share__Daily_Time_Series_Ratio_Count,
# SUM(case when Total_Debt_To_EBITDA__Daily_Time_Series_Ratio_ IS NULL then 1 end) as Total_Debt_To_EBITDA__Daily_Time_Series_Ratio_Count,
# SUM(case when Total_Debt_To_Enterprise_Value__Daily_Time_Series_Ratio_ IS NULL then 1 end) as Total_Debt_To_Enterprise_Value__Daily_Time_Series_Ratio_Count,
# SUM(case when Volume IS NULL then 1 end) as Volume_Count
# FROM seng_550_data.technicals
