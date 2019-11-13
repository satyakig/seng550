"""
Run:  bash ../gcp_scripts/bash_scripts_run_cluster.bash ./Linear_Regressor.py james-cluster us-central1
"""
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

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
fb_year.printSchema()
print(fb_year.describe().toPandas().transpose())

"""
Observe which variables correlate the most with the price
Here I will only take variables with over 0.7 correlation to ensure 
that my features all heavily correlate to the label
"""
# float_columns = []
# for i in fb_year.columns:
#     column_name = fb_year.select(i).take(1)[0][0]
#     if isinstance(column_name, float):
#         float_columns.append(i)
#         print("Correlation to Price for ", i, fb_year.stat.corr("Price", i))


float_columns = ['Volume', 'Enterprise_Value_To_Sales_Daily_Time_Series_Ratio', 'P_E_Daily_Time_Series_Ratio',
 'Price_To_Book_Value_Per_Share_Daily_Time_Series_Ratio', 'Price_To_Cash_Flow_Per_Share_Daily_Time_Series_Ratio',
 'Price_To_Sales_Per_Share_Daily_Time_Series_Ratio', 'Total_Debt_To_EBITDA_Daily_Time_Series_Ratio', 'Total_Debt_To_Enterprise_Value_Daily_Time_Series_Ratio',
 'Book_Value_Per_Share', 'Cash_and_Short_Term_Investments', 'Cost_of_Revenue__Total', 'Current_Ratio', 'Diluted_EPS_Excluding_Extraordinary_Items',
 'Diluted_EPS_Including_Extraordinary_Items', 'EBIT', 'EBIT_Margin__Percent', 'Goodwill__Net', 'Gross_Dividends___Common_Stock', 'Gross_Margin__Percent',
 'Net_Income_Before_Taxes', 'Normalized_Income_Avail_to_Cmn_Shareholders', 'Operating_Expenses', 'Operating_Income', 'Operating_Margin__Percent',
 'Property_Plant_Equipment__Total___Net', 'Quick_Ratio', 'ROA_Total_Assets__Percent', 'Revenue_Per_Share', 'Tangible_Book_Value_Per_Share',
 'Total_Assets__Reported', 'Total_Current_Liabilities', 'Total_Current_Assets', 'Total_Debt', 'Total_Equity', 'Total_Liabilities', 'Total_Long_Term_Debt',
 'Total_Receivables__Net', 'Total_Revenue', 'Total_Common_Shares_Outstanding', 'Total_Debt_to_Total_Equity__Percent']

# Prepare the features column and the labels
vector_assembler = VectorAssembler(inputCols=float_columns, outputCol='features')
fb_feature_df = vector_assembler.transform(fb_year.na.fill(0)) # Fills with zeros
fb_feature_df = fb_feature_df.select(['features', 'Price'])
fb_feature_df.show(3)

# Split the dataset
splits = fb_feature_df.randomSplit([0.7, 0.3])
train_df = splits[0]
test_df = splits[1]

# train the linear regression model
lr = LinearRegression(featuresCol='features', labelCol='Price',
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
lr_predictions.select("prediction", "Price", "features").show(5)
lr_evaluator = RegressionEvaluator(predictionCol="prediction",
                                   labelCol="Price",
                                   metricName="r2")
print("Training R Squared (R2) on test data = {}".format(lr_evaluator.evaluate(lr_predictions)))
test_result = lr_model.evaluate(test_df)
print("Training Root Mean Squared Error (RMSE) on test data = {}".format(test_result.rootMeanSquaredError))

"""
Model Results:

Training r2: 0.9688
Test r2: 0.9717
"""
