'''
Run:  bash ../gcp_scripts/bash_scripts_run_cluster.bash ./Linear_Regressor.py james-cluster us-central1

Checkout put of this job later: [8f3e6d5f78024541bfa107d4fccd54e7]
'''
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, array, lit, col
from pyspark.sql.types import StructType, StructField, FloatType, BooleanType, IntegerType, StringType
import pandas as pd

# Constants
DATABASE = 'seng-550.seng_550_data'
TABLE = 'combined_technicals_fundamentals'
VIEW_NAME = 'combined_data'
TARGET = 'Price_Close'
RESULTS_TABLE_NAME = 'Lin_Reg_Results'
TECH_SECTOR = 'Technology'
columns_to_drop = [
    'Date',
    'Company_Common_Name',
    'Exchange_Name',
    'Country_of_Headquarters',
    'Year',
    'TRBC_Economic_Sector_Name',
    'Instrument',
]
companies_to_check = [
    'FB.O',
    'GOOGL.O',
    'AAPL.O',
    'AMZN.O',
    'NFLX.O',
    'TSLA.O',
    'MSFT.O',
    'PYPL.O',
    'AMD.O',
    'INTC.O',
    'NVDA.O',
]

# Create the Spark Session on this node
spark = SparkSession \
    .builder \
    .master('yarn') \
    .appName('lr') \
    .config('spark.executorEnv.LD_PRELOAD', 'libnvblas.so') \
    .getOrCreate()
bucket = spark.sparkContext._jsc.hadoopConfiguration().get('fs.gs.system.bucket')
spark.conf.set('temporaryGcsBucket', bucket)


def load_data(tech_only=False):
    """
    This function pulls the training and test data from big query

    :return: Train and Test data
    """
    # Load data from Big Query
    combined_data = spark.read.format('bigquery').option('table', '{}.{}'.format(DATABASE, TABLE)).load().cache()
    combined_data.createOrReplaceTempView('combined_data')

    end_year = 2018
    if tech_only:
        # Start - 2017 rows
        query = 'SELECT * FROM {} WHERE Year < {} AND TRBC_Economic_Sector_Name="{}"'.format(VIEW_NAME, end_year, TECH_SECTOR)
        training_data = spark.sql(query)

        # 2018 - now
        query = 'SELECT * FROM {} WHERE Year >= {} AND TRBC_Economic_Sector_Name="{}"'.format(VIEW_NAME, end_year, TECH_SECTOR)
        test_data = spark.sql(query)

    else:
        # Start - 2017 rows
        query = 'SELECT * FROM {} WHERE Year < {}'.format(VIEW_NAME, end_year)
        training_data = spark.sql(query)

        # 2018 - now
        query = 'SELECT * FROM {} WHERE Year >= {}'.format(VIEW_NAME, end_year)
        test_data = spark.sql(query)

    return training_data, test_data


def preprocess_data(training_data, test_data):
    """
    This function will pre-process data by dropping the rows we specify and one-hot encoding all categorical data
    :return:
    """
    # Drop columns that we don't want
    train_data = training_data.drop(*columns_to_drop)
    train_data = train_data.select(*(col(c).cast('float').alias(c) for c in train_data.columns))
    train_data = train_data.fillna(0)

    test_dict = dict()
    for test_company in companies_to_check:
        company_data = test_data.filter(test_data.Instrument == test_company).drop(*columns_to_drop)

        if company_data.count() > 0:
            company_data = company_data.select(*(col(c).cast('float').alias(c) for c in company_data.columns))
            company_data = company_data.fillna(0)
            test_dict[test_company] = company_data

    return train_data, test_dict


def vectorize_data(training_data, test_data):
    # Assemble the vectors
    input_columns = training_data.columns.remove(TARGET)
    vector_assembler = VectorAssembler(inputCols=input_columns, outputCol='features')
    train_df = vector_assembler.transform(training_data)

    # Normalize the data using Scalar
    scalar = StandardScaler(inputCol='features', outputCol='scaledFeatures', withStd=True, withMean=True).fit(train_df)
    train_df = scalar.transform(train_df)

    # Select the rows needed
    train_df = train_df.select(['scaledFeatures', TARGET])

    new_test_data = dict()
    for company in test_data:
        company_data = test_data[company]
        test_df = vector_assembler.transform(company_data)
        test_df = scalar.transform(test_df)

        test_df = test_df.select(['scaledFeatures', TARGET])
        new_test_data[company] = test_df

    return train_df, new_test_data


error_pct_udf = udf(lambda arr: (float(abs(arr[0] - arr[1])) * float(100)) / float(arr[0]), FloatType())


def train_and_pred(train, test_data, tech_only=False):
    # train the linear regression model
    lr_model = LinearRegression(featuresCol='scaledFeatures', labelCol=TARGET, maxIter=300, regParam=1, elasticNetParam=1).fit(train)
    print('Coefficients: {}'.format(str(lr_model.coefficients)))
    print('Intercept: {}'.format(str(lr_model.intercept)))

    # summarize the training
    trainingSummary = lr_model.summary
    print('Training r2 = {}'.format(float(trainingSummary.r2)))
    print('Training RMSE = {}\n'.format(float(trainingSummary.rootMeanSquaredError)))

    predictions_dict = dict()
    for company in test_data:
        test_company_data = test_data[company]
        lr_predictions = lr_model.transform(test_company_data)

        # Model Evaluation
        lr_evaluator = RegressionEvaluator(predictionCol='prediction', labelCol=TARGET, metricName='r2')
        test_r2 = lr_evaluator.evaluate(lr_predictions)
        print('{}, testing r2 = {}'.format(company.upper(), test_r2))

        test_result = lr_model.evaluate(test_company_data)
        print('{}, testing RMSE = {}\n'.format(company.upper(), test_result.rootMeanSquaredError))

        new_df = lr_predictions.drop('scaledFeatures').withColumn('Instrument', lit(company))
        new_df = new_df.withColumn('Error_Pct', error_pct_udf(array(TARGET, 'prediction')))
        new_df = new_df.withColumn('Tech_Only_Pred', lit(tech_only))

        predictions_dict[company] = new_df.toPandas().reset_index().rename(columns={'index': 'row_num'})

    return predictions_dict


df_arr = []


train, test = load_data()
train, test = preprocess_data(train, test)
train, test = vectorize_data(train, test)
predictions = train_and_pred(train, test)
for pred in predictions:
    df_arr.append(predictions[pred])

train, test = load_data(True)
train, test = preprocess_data(train, test)
train, test = vectorize_data(train, test)
predictions = train_and_pred(train, test, True)
for pred in predictions:
    df_arr.append(predictions[pred])

combined_output = pd.concat(df_arr, ignore_index=True)
schema = StructType([
    StructField('row_num', IntegerType(), True),
    StructField('Price_Close', FloatType(), True),
    StructField('prediction', FloatType(), True),
    StructField('Instrument', StringType(), True),
    StructField('Error_Pct', FloatType(), True),
    StructField('Tech_Only_Pred', BooleanType(), True),
])

spark_df = spark.createDataFrame(combined_output, schema=schema)

# Write results to bigquery:
spark_df.write.format('bigquery').option('table', 'seng-550.seng_550_data.{}'.format(RESULTS_TABLE_NAME)).mode('overwrite').save()
