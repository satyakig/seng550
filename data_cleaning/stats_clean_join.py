from pyspark.sql import SparkSession
from pyspark.sql.functions import to_timestamp, udf, year
from pyspark.sql.types import DoubleType, IntegerType
from json import dumps


# The 'technicals' table has incorrect data types when you create it from the raw csv files for some reason
# Transform technicals columns to proper data types
def transform_technicals(frame):
    frame = frame.withColumn('Date', to_timestamp(frame['Date']))
    frame = frame.withColumn('Enterprise_Value_To_Sales__Daily_Time_Series_Ratio_',
                             frame['Enterprise_Value_To_Sales__Daily_Time_Series_Ratio_'].cast(DoubleType()))
    frame = frame.withColumn('P_E__Daily_Time_Series_Ratio_', frame['P_E__Daily_Time_Series_Ratio_'].cast(DoubleType()))
    frame = frame.withColumn('Price_Close', frame['Price_Close'].cast(DoubleType()))
    frame = frame.withColumn('Price_To_Book_Value_Per_Share__Daily_Time_Series_Ratio_',
                             frame['Price_To_Book_Value_Per_Share__Daily_Time_Series_Ratio_'].cast(DoubleType()))
    frame = frame.withColumn('Price_To_Cash_Flow_Per_Share__Daily_Time_Series_Ratio_',
                             frame['Price_To_Cash_Flow_Per_Share__Daily_Time_Series_Ratio_'].cast(DoubleType()))
    frame = frame.withColumn('Price_To_Sales_Per_Share__Daily_Time_Series_Ratio_',
                             frame['Price_To_Sales_Per_Share__Daily_Time_Series_Ratio_'].cast(DoubleType()))
    frame = frame.withColumn('Total_Debt_To_EBITDA__Daily_Time_Series_Ratio_',
                             frame['Total_Debt_To_EBITDA__Daily_Time_Series_Ratio_'].cast(DoubleType()))
    frame = frame.withColumn('Total_Debt_To_Enterprise_Value__Daily_Time_Series_Ratio_',
                             frame['Total_Debt_To_Enterprise_Value__Daily_Time_Series_Ratio_'].cast(DoubleType()))
    return frame


# Get quarter from a timestamp/date object
def get_quarter(dt):
    month = dt.month
    if 1 <= month <= 3:
        return 1
    elif 4 <= month <= 6:
        return 2
    elif 7 <= month <= 9:
        return 3
    else:
        return 4


TECHNICALS = 'technicals'
FUNDAMENTALS = 'fundamentals'

# Setup the PySpark session
# spark = SparkSession.builder.master('local').appName('stats_clean_join').getOrCreate() # Local run spark session
spark = SparkSession.builder.master('yarn').appName('stats_clean_join').getOrCreate()
bucket = spark.sparkContext._jsc.hadoopConfiguration().get('fs.gs.system.bucket')
spark.conf.set('temporaryGcsBucket', bucket)

# Read the technicals table
technicals = spark.read.format('bigquery').option('table', 'seng-550.seng_550_data.technicals_raw').load().cache()
technicals.createOrReplaceTempView(TECHNICALS)
technicals = transform_technicals(technicals)

# Read the fundamentals table
fundamentals = spark.read.format('bigquery').option('table', 'seng-550.seng_550_data.fundamentals_raw').load().cache()
fundamentals.createOrReplaceTempView(FUNDAMENTALS)

# Dict that contains some values for each table used during computation
# 'df': Stores the DataFrame for the table
# 'or_cols': These cols will be used to compute the 'one or more cols' missing stats
# 'not_required_cols':  Cols that are useless and won't be used
# 'non_null_cols': Cols that cannot be null for our project (NEEDS at least 2 cols, or the query has to be modified)
tables = {
    TECHNICALS: {
        'df': technicals,
        'or_cols': ['Date', 'Instrument', 'Price_Close', 'Volume'],
        'not_required_cols': ['int64_field_0'],
        'non_null_cols': ['Date', 'Instrument', 'Price_Close', 'Volume'],
    },
    FUNDAMENTALS: {
        'df': fundamentals,
        'or_cols': ['Date', 'Instrument', 'Total_Debt', 'Total_Revenue', 'Total_Current_Assets', 'Total_Current_Liabilities'],
        'not_required_cols': [],
        'non_null_cols': ['Date', 'Instrument'],
    }
}

# Find the missing stats for each table
print('\nComputing table stats...')
for table in tables:
    df = tables[table]['df']
    or_cols = tables[table]['or_cols']
    cols = df.columns
    total = df.count()

    print('\n{}'.format(table.upper()))
    df.printSchema()
    print('Total rows: {}'.format(total))
    print('Missing stats:')

    missing_counts = dict()
    for key in cols:
        count = spark.sql('SELECT COUNT(*) as count FROM {} WHERE {} IS NULL'.format(table, key)).first()['count']
        missing_counts[key] = [count, '{:.4f}%'.format(float(count) / float(total) * 100)]
    print(dumps(missing_counts, indent=4))

    all_query = 'SELECT COUNT(*) as count FROM {} WHERE {}'.format(table, ' IS NULL OR '.join(cols) + ' IS NULL')
    count = spark.sql(all_query).first()['count']
    print('One or more cols missing:  {}, {:.4f}%'.format(count, float(count) / float(total) * 100))

    or_query = 'SELECT COUNT(*) as count FROM {} WHERE {}'.format(table,
                                                                  ' IS NULL OR '.join(or_cols) + ' IS NULL' if len(
                                                                      or_cols) > 0 else '')
    count = spark.sql(or_query).first()['count']
    print('{} missing: {}, {:.4f}%'.format(' or '.join(or_cols), count, float(count) / float(total) * 100))
print('\nCompleted table stats\n')

# PySpark UDF functions to create new cols
quarter_udf_fundamental = udf(lambda dt: get_quarter(dt) + 1 if get_quarter(dt) != 4 else 1, IntegerType())
year_udf_fundamental = udf(lambda dt: dt.year if get_quarter(dt) != 4 else dt.year + 1, IntegerType())
quarter_udf = udf(lambda dt: get_quarter(dt), IntegerType())

# Clean up the two tables
print('\nCleaning up tables...')
for table in tables:
    print('\n{}'.format(table.upper()))
    print('Cleaning table'.format(table.upper()))
    not_required_cols = tables[table]['not_required_cols']
    non_null_cols = tables[table]['non_null_cols']
    frame = tables[table]['df']
    required_cols = [x for x in frame.columns if x not in not_required_cols]

    cleaner_query = 'SELECT {} FROM {} WHERE {}'.format(', '.join(required_cols), table,
                                                        ' IS NOT NULL AND '.join(non_null_cols) + ' IS NOT NULL')
    print('Cleaner query: {}'.format(cleaner_query))
    cleaned_table = spark.sql(cleaner_query)

    if table == TECHNICALS:
        cleaned_table = transform_technicals(cleaned_table)
        cleaned_table = cleaned_table.withColumn('Year', year(cleaned_table['Date']))
        cleaned_table = cleaned_table.withColumn('Quarter', quarter_udf('Date'))
    elif table == FUNDAMENTALS:
        cleaned_table = cleaned_table.withColumn('Year', year_udf_fundamental('Date'))
        cleaned_table = cleaned_table.withColumn('Quarter', quarter_udf_fundamental('Date'))
        cleaned_table.createOrReplaceTempView(table)
        fundamental_drop_duplicate = 'SELECT * FROM(SELECT  *, ROW_NUMBER() OVER(PARTITION BY f.Date, f.Instrument ORDER BY f.Instrument, f.Date DESC) rn FROM fundamentals as f) WHERE rn = 1'
        cleaned_table = spark.sql(fundamental_drop_duplicate)
        cleaned_table = cleaned_table.drop('rn')

    cleaned_table.createOrReplaceTempView(table)
    tables[table]['df'] = cleaned_table
    cleaned_table.printSchema()

    print('Saving table to BigQuery')
    cleaned_table.write.format('bigquery').option('table', 'seng-550.seng_550_data.{}_cleaned'.format(table)).mode(
        'overwrite').save()
    print('Newly cleaned table: {} has {} rows'.format(table.upper(), cleaned_table.count()))
