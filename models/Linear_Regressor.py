"""
Run:  bash ../gcp_scripts/bash_scripts_run_cluster.bash ./Linear_Regressor.py james-cluster us-central1

Checkout put of this job later: [8f3e6d5f78024541bfa107d4fccd54e7]
"""
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

# Constants
DATABASE = "seng-550.seng_550_data"
TABLE = "combined_technicals_fundamentals"
TARGET = "Price_Close"
FLOAT_COLUMNS = ['Enterprise_Value_To_Sales__Daily_Time_Series_Ratio_', 'P_E__Daily_Time_Series_Ratio_',
                 'Price_Close', 'Price_To_Book_Value_Per_Share__Daily_Time_Series_Ratio_',
                 'Price_To_Cash_Flow_Per_Share__Daily_Time_Series_Ratio_', 'Price_To_Sales_Per_Share__Daily_Time_Series_Ratio_',
                 'Total_Debt_To_EBITDA__Daily_Time_Series_Ratio_', 'Total_Debt_To_Enterprise_Value__Daily_Time_Series_Ratio_',
                 'Volume', 'Book_Value_Per_Share', 'Cash_and_Short_Term_Investments', 'Cost_of_Revenue__Total', 'Current_Ratio',
                 'Diluted_EPS_Excluding_Extraordinary_Items', 'Diluted_EPS_Including_Extraordinary_Items', 'EBIT',
                 'EBIT_Margin__Percent', 'Goodwill__Net', 'Gross_Dividends___Common_Stock', 'Gross_Margin__Percent',
                 'Net_Income_Before_Taxes', 'Normalized_Income_Avail_to_Cmn_Shareholders', 'Operating_Expenses', 'Operating_Income',
                 'Operating_Margin__Percent', 'Property_Plant_Equipment__Total___Net', 'Quick_Ratio', 'ROA_Total_Assets__Percent',
                 'Revenue_Per_Share', 'Tangible_Book_Value_Per_Share', 'Total_Assets__Reported', 'Total_Current_Liabilities',
                 'Total_Current_Assets', 'Total_Debt', 'Total_Equity', 'Total_Inventory', 'Total_Liabilities', 'Total_Long_Term_Debt',
                 'Total_Receivables__Net', 'Total_Revenue', 'Total_Common_Shares_Outstanding', 'Total_Debt_to_Total_Equity__Percent']

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
training_data = training_data.na.fill(0)
vector_assembler = VectorAssembler(inputCols=FLOAT_COLUMNS, outputCol='features')
train_df = vector_assembler.transform(training_data)
train_df = train_df.select(['features', TARGET])

# 2018 - now
test_data = spark.sql(
    'SELECT * FROM combined_data WHERE Instrument="FB.O" AND Year >= 2018'
)
test_data = test_data.na.fill(0)
test_df = vector_assembler.transform(test_data)
test_df = test_df.select(['features', TARGET])

# train the linear regression model
lr = LinearRegression(featuresCol='features', labelCol=TARGET,
                      maxIter=10, regParam=0.3, elasticNetParam=0.8)
lr_model = lr.fit(train_df)
print("Coefficients: {}".format(str(lr_model.coefficients)))
print("Intercept: {}".format(str(lr_model.intercept)))
# summarize the training
trainingSummary = lr_model.summary
print("Training RMSE: {}".format(float(trainingSummary.rootMeanSquaredError)))
print("Training r2: {}".format(float(trainingSummary.r2)))

# summarize the test data
lr_predictions = lr_model.transform(test_df)
# Select example rows to display.
lr_predictions.select("prediction", TARGET, "features").show(5)
lr_evaluator = RegressionEvaluator(predictionCol="prediction",
                                   labelCol=TARGET,
                                   metricName="r2")
test_r2 = lr_evaluator.evaluate(lr_predictions)

# Write this to big query to debug further
print("Testing r2: {}".format(test_r2))
test_result = lr_model.evaluate(test_df)
print("Training Root Mean Squared Error (RMSE) on test data = {}".format(test_result.rootMeanSquaredError))

"""
Model Results:
With just Facebook:
- Training r2: 0.9688
- Test r2: 0.9717
With all companies:
- Training r2: 0.90256
- Test r2: 0.921473
"""
