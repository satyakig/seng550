"""
Run:  bash ../gcp_scripts/bash_scripts_run_cluster.bash ./Linear_Regressor.py james-cluster us-central1

Checkout put of this job later: [8f3e6d5f78024541bfa107d4fccd54e7]
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

# Constants
DATABASE = "seng-550.seng_550_data"
TABLE = "combined_technicals_fundamentals"
TARGET = "Price_Close"
RESULTS_TABLE_NAME = "Lin_Reg_Results"
columns_to_drop = ["Instrument", "Date", "Company_Common_Name", "TRBC_Economic_Sector_Name",
                   "Country_of_Headquarters", "Exchange_Name", "Year"]

# Create the Spark Session on this node
spark = SparkSession \
    .builder \
    .master('yarn') \
    .appName('lr') \
    .config('spark.executor.cores', '16') \
    .config('spark.executor.memory', '71680m') \
    .config('spark.executorEnv.LD_PRELOAD', 'libnvblas.so') \
    .getOrCreate()

bucket = spark.sparkContext._jsc.hadoopConfiguration().get('fs.gs.system.bucket')
spark.conf.set('temporaryGcsBucket', bucket)


def load_data():
    """
    This function pulls the training and test data from big query

    :return: Train and Test data
    """
    # Load data from Big Query
    combined_data = spark.read.format('bigquery').option('table', "{}.{}".format(DATABASE, TABLE)).load().cache()
    combined_data.createOrReplaceTempView('combined_data')

    # Start - 2017 rows
    training_data = spark.sql(
        'SELECT * FROM combined_data WHERE Year < 2018'
    )
    training_data = training_data.na.fill(0)

    # 2018 - now
    test_data = spark.sql(
        'SELECT * FROM combined_data WHERE Year >= 2018'
    )
    test_data = test_data.na.fill(0)

    return training_data, test_data


def preprocess_data(training_data, test_data):
    """
    This function will pre-process data by dropping the rows we specify and one-hot encoding all categorical data
    :return:
    """
    # Drop columns that we don't want
    train_data = training_data.drop(*columns_to_drop)
    test_data = test_data.drop(*columns_to_drop)

    # One-hot encode each column

    # Convert each column to a float
    for col_name in train_data.columns:
        train_data = train_data.withColumn(col_name, col(col_name).cast('float'))
    for col_name in test_data.columns:
        test_data = test_data.withColumn(col_name, col(col_name).cast('float'))

    return train_data, test_data


def vectorize_data(training_data, test_data):
    # Assemble the vectors
    vector_assembler = VectorAssembler(inputCols=training_data.columns, outputCol='features')
    train_df = vector_assembler.transform(training_data)
    test_df = vector_assembler.transform(test_data)

    # Normalize the data using Scalar
    scalar = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=True).fit(train_df)
    train_df = scalar.transform(train_df)
    test_df = scalar.transform(test_df)

    # Select the rows needed
    train_df = train_df.select(['scaledFeatures', TARGET])
    test_df = test_df.select(['scaledFeatures', TARGET])

    return train_df, test_df


train, test = load_data()
train, test = preprocess_data(train, test)
train, test = vectorize_data(train, test)
train.show(n=5)
test.show(n=5)

# train the linear regression model
lr = LinearRegression(featuresCol='scaledFeatures', labelCol=TARGET,
                      maxIter=600, regParam=1, elasticNetParam=1)
lr_model = lr.fit(train)
print("Coefficients: {}".format(str(lr_model.coefficients)))
print("Intercept: {}".format(str(lr_model.intercept)))
# summarize the training
trainingSummary = lr_model.summary
print("Training RMSE: {}".format(float(trainingSummary.rootMeanSquaredError)))
print("Training r2: {}".format(float(trainingSummary.r2)))

# summarize the test data
lr_predictions = lr_model.transform(test)

# Select example rows to display.
lr_predictions.select("prediction", TARGET).show(5)

# Model Evaluation
lr_evaluator = RegressionEvaluator(predictionCol="prediction",
                                   labelCol=TARGET,
                                   metricName="r2")
test_r2 = lr_evaluator.evaluate(lr_predictions)

# Write this to big query to debug further
print("Testing r2: {}".format(test_r2))
test_result = lr_model.evaluate(test)
print("Training Root Mean Squared Error (RMSE) on test data = {}".format(test_result.rootMeanSquaredError))

# Write results to bigquery:
lr_predictions.select("prediction", TARGET).write.format('bigquery').option('table', 'seng-550.seng_550_data.{}'.format(RESULTS_TABLE_NAME)).mode('overwrite').save()
