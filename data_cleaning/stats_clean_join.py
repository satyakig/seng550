from pyspark.sql import SparkSession
from pyspark.sql.functions import to_timestamp, udf, year
from pyspark.sql.types import DoubleType, IntegerType


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
def get_quarter_technical(dt):
    month = dt.month

    if 1 < month <= 3:
        return 1
    elif 4 <= month <= 6:
        return 2
    elif 7 <= month <= 9:
        return 3
    else:
        return 4


# Get quarter and year from a timestamp/date object for fundamentals
# Increment the quarter and year where applicable
# This function needs to be updated, because there are still some duplicates in fundamentals
# If Date of the fundamentals report:
#       March - May = Q2, same year
#       June - Aug = Q3, same year
#       September - November = Q4, same year
#       December = Q1, Next year
#       Jan, Feb = Q1, same year
def get_quarter_year_fundamental(dt):
    month = dt.month

    if 3 <= month <= 5:
        return 2, dt.year
    elif 6 <= month <= 8:
        return 3, dt.year
    elif 9 <= month <= 11:
        return 4, dt.year
    else:
        if month == 12:
            return 1, dt.year + 1
        else:
            return 1, dt.year


TECHNICALS = 'technicals'
FUNDAMENTALS = 'fundamentals'

# Setup the PySpark session
# spark = SparkSession.builder.master('local').appName('stats_clean_join').getOrCreate() # Local run spark session
spark = SparkSession \
    .builder \
    .master('yarn') \
    .appName('stats_clean_join') \
    .config('spark.executor.cores', '16') \
    .config('spark.executor.memory', '71680m') \
    .config('spark.executorEnv.LD_PRELOAD', 'libnvblas.so') \
    .getOrCreate()

bucket = spark.sparkContext._jsc.hadoopConfiguration().get('fs.gs.system.bucket')
spark.conf.set('temporaryGcsBucket', bucket)

# Read the technicals table
technicals_df = spark.read.format('bigquery').option('table', 'seng-550.seng_550_data.technicals_raw').load().cache()
technicals_df.createOrReplaceTempView(TECHNICALS)
technicals_df = transform_technicals(technicals_df)

# Read the fundamentals table
fundamentals_df = spark.read.format('bigquery').option('table', 'seng-550.seng_550_data.fundamentals_raw').load().cache()
fundamentals_df.createOrReplaceTempView(FUNDAMENTALS)


# Dict that contains some values for each table used during computation
# 'df': Stores the DataFrame for the table
# 'or_cols': These cols will be used to compute the 'one or more cols' missing stats
# 'not_required_cols':  Cols that are useless and won't be used
# 'non_null_cols': Cols that cannot be null for our project
tables = {
    TECHNICALS: {
        'name': TECHNICALS,
        'df': technicals_df,
        'or_cols': ['Date', 'Instrument', 'Price_Close', 'Volume'],
        'not_required_cols': ['int64_field_0'],
        'non_null_cols': ['Date', 'Instrument', 'Price_Close', 'Volume'],
    },
    FUNDAMENTALS: {
        'name': FUNDAMENTALS,
        'df': fundamentals_df,
        'or_cols': ['Date', 'Instrument', 'Total_Debt', 'Total_Revenue', 'Total_Current_Assets', 'Total_Current_Liabilities'],
        'not_required_cols': [],
        'non_null_cols': ['Date', 'Instrument'],
    }
}


# Find the missing stats for a table
def compute_stats(table_data):
    table_name = table_data['name']
    df = table_data['df']
    or_cols = table_data['or_cols']

    cols = df.columns
    total = df.count()

    print('\n{}'.format(table_name.upper()))
    df.printSchema()
    print('Total rows: {}'.format(total))
    print('Null stats:')

    cols_len = len(cols)
    query = 'SELECT '
    for index, col in enumerate(cols):
        if index == cols_len - 1:
            partial_query = 'SUM(CASE WHEN {} IS NULL then 1 END) as {}'.format(col, col)
        else:
            partial_query = 'SUM(CASE WHEN {} IS NULL then 1 END) as {}, '.format(col, col)
        query = query + partial_query
    query = query + ' FROM {}'.format(table_name)
    null_frame = spark.sql(query)
    null_frame.show()
    for col in cols:
        first = null_frame.first()
        count = first[col] if first[col] is not None else 0
        print('{}: {} nulls or {:.4f}%'.format(col, count, float(count) / float(total) * 100))

    all_query = 'SELECT COUNT(*) as count FROM {} WHERE {}'.format(table_name, ' IS NULL OR '.join(cols) + ' IS NULL')
    count = spark.sql(all_query).first()['count']
    print('One or more cols: {} nulls or {:.4f}%'.format(count, float(count) / float(total) * 100))

    if len(or_cols) >= 2:
        or_query = 'SELECT COUNT(*) as count FROM {} WHERE {}'.format(table_name, ' IS NULL OR '.join(or_cols) + ' IS NULL')
        count = spark.sql(or_query).first()['count']
        print('{}: {} nulls or {:.4f}%'.format(' or '.join(or_cols), count, float(count) / float(total) * 100))


print('\nComputing UNCLEAN table stats...')
for table in tables:
    compute_stats(tables[table])
print('\nCompleted UNCLEAN table stats\n')


# PySpark UDF functions to create new cols
quarter_udf_fundamental = udf(lambda dt: get_quarter_year_fundamental(dt)[0], IntegerType())
year_udf_fundamental = udf(lambda dt: get_quarter_year_fundamental(dt)[1], IntegerType())
quarter_udf_technical = udf(lambda dt: get_quarter_technical(dt), IntegerType())


# Clean up the two tables
print('\nCleaning up tables...')
for table in tables:
    not_required_cols = tables[table]['not_required_cols']
    non_null_cols = tables[table]['non_null_cols']
    df = tables[table]['df']
    required_cols = [x for x in df.columns if x not in not_required_cols]
    cleaned_table = df

    print('\n{}'.format(table.upper()))
    print('Cleaning table'.format(table.upper()))

    if len(non_null_cols) >= 2:
        cleaner_query = 'SELECT {} FROM {} WHERE {}'.format(', '.join(required_cols), table,
                                                            ' IS NOT NULL AND '.join(non_null_cols) + ' IS NOT NULL')
        cleaned_table = spark.sql(cleaner_query)

    if table == TECHNICALS:
        cleaned_table = transform_technicals(cleaned_table)
        cleaned_table = cleaned_table.withColumn('Year', year(cleaned_table['Date']))
        cleaned_table = cleaned_table.withColumn('Quarter', quarter_udf_technical('Date'))
    elif table == FUNDAMENTALS:
        cleaned_table = cleaned_table.withColumn('Year', year_udf_fundamental('Date'))
        cleaned_table = cleaned_table.withColumn('Quarter', quarter_udf_fundamental('Date'))
        cleaned_table.createOrReplaceTempView(table)
        fundamental_drop_duplicate = 'SELECT * FROM(SELECT  *, ROW_NUMBER() OVER(PARTITION BY Date, Instrument ORDER BY Instrument, Date DESC) rn FROM fundamentals) WHERE rn = 1'
        cleaned_table = spark.sql(fundamental_drop_duplicate)
        cleaned_table = cleaned_table.drop('rn')

    cleaned_table.createOrReplaceTempView(table)
    tables[table]['df'] = cleaned_table
    cleaned_table.printSchema()

    print('Saving table to BigQuery')
    cleaned_table.write.format('bigquery').option('table', 'seng-550.seng_550_data.{}_cleaned'.format(table)).mode(
        'overwrite').save()


print('\nComputing CLEAN table stats...')
for table in tables:
    compute_stats(tables[table])
print('\nCompleted CLEAN table stats\n')


# Clear up some memory
spark.catalog.dropTempView(TECHNICALS)
spark.catalog.dropTempView(FUNDAMENTALS)

JOINED_TABLE_NAME = 'combined_technicals_fundamentals'

print('\n\nJoining fundamentals and technicals')
technicals_df = tables[TECHNICALS]['df']
fundamentals_df = tables[FUNDAMENTALS]['df']
joined_df = technicals_df.join(fundamentals_df.drop('Date'), ['Instrument', 'Quarter', 'Year'])

joined_df.createOrReplaceTempView(JOINED_TABLE_NAME)
print('Saving joined table to BigQuery')
joined_df.write.format('bigquery').option('table', 'seng-550.seng_550_data.{}'.format(JOINED_TABLE_NAME)).mode('overwrite').save()

joined_table = {
    'name': JOINED_TABLE_NAME,
    'df': joined_df,
    'or_cols': [],
    'not_required_cols': [],
    'non_null_cols': [],
}
compute_stats(joined_table)
