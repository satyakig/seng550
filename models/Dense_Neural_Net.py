"""
Run:  bash ../gcp_scripts/bash_scripts_run_cluster.bash ./Linear_Regressor.py james-cluster us-central1
"""
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
from keras import models
from keras import layers
from keras.optimizers import RMSprop, serialize

from elephas.ml_model import ElephasEstimator

# Constants
DATABASE = "seng-550.seng_550_data"
TABLE = "combined_technicals_fundamentals"
TARGET = "Price_Close"

# Create the Spark Session on this node
spark = SparkSession.builder.master('yarn').appName(
    'combined_fund').getOrCreate()

# Create a temporary bucket
bucket = spark.sparkContext._jsc.hadoopConfiguration().get('fs.gs.system.bucket')
spark.conf.set('temporaryGcsBucket', bucket)

# Load data from Big Query
combined_data = spark.read.format('bigquery').option('table', "{}.{}".format(DATABASE, TABLE)).load().cache()
combined_data.createOrReplaceTempView('combined_data')

# Training: Start - 2017 rows
training_data = spark.sql(
    'SELECT * FROM combined_data WHERE Instrument="FB.O" AND Year < 2018'
)
training_data = training_data.na.fill(0)
types = [(f.name, f.dataType) for f in training_data.schema.fields]
cat_features = []
num_features = []
for i in types:
    if str(i[1]) in ["LongType", "DoubleType"]:
        num_features.append(i[0])
    else:
        cat_features.append(i[0])

# initialize VectorAssembler
vector_assembler = VectorAssembler(inputCols=num_features, outputCol='features')
train_df = vector_assembler.transform(training_data)
train_df = train_df.select(['features', TARGET])
train_df.show()

# Testing: 2018 - now
test_data = spark.sql(
    'SELECT * FROM combined_data WHERE Instrument="FB.O" AND Year >= 2018'
)
test_data = test_data.na.fill(0)
test_df = vector_assembler.transform(test_data)
test_df = test_df.select(['features', TARGET])
print("This test_df is type {}".format(type(test_df)))
test_df.show()

# Creating a Neural Network
input_dim = len(train_df.select("features").first()[0])
print("There are {} features being used as input".format(input_dim))
model = models.Sequential()
model.add(layers.Dense(64, activation="relu", input_shape=(input_dim,)))
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(1))
model.compile(optimizer="rmsprop", loss="mse")
model.summary()

# Create Elephas Estimator
optimizer_conf = RMSprop(lr=0.01)
opt_conf = serialize(optimizer_conf)

estimator = ElephasEstimator()
estimator.setFeaturesCol("features")
estimator.setLabelCol(TARGET)
estimator.set_keras_model_config(model.to_yaml())
estimator.set_categorical_labels(False)
estimator.set_num_workers(1)
estimator.set_epochs(20)
estimator.set_batch_size(128)
estimator.set_verbosity(1)
estimator.set_mode("synchronous")
estimator.set_loss("mse")
estimator.set_metrics(["mae"])
estimator.set_optimizer_config(opt_conf)
estimator.set_validation_split(0.10)

fitted_model = estimator.fit(train_df)

# Evaluate the model
# summarize the test data
prediction = fitted_model.transform(test_df)
prediction.select("prediction").show()

# Select example rows to display.
print("This result is type {}".format(type(prediction)))
prediction.select(prediction['features'], prediction['prediction']).show()
"""
DataFrame[features: vector, Price_Close: double, prediction: double]
DataFrame[features: vector, Price_Close: double]
"""