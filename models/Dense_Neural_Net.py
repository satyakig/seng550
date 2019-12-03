"""
Run:  bash ../gcp_scripts/run_cluster.bash ./Dense_Neural_Net.py james-cluster us-central1
"""
from pyspark.sql import SparkSession
import systemml
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import r2_score
from pyspark.sql.functions import col
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import VectorAssembler
from keras import models
from keras.layers import Dense, Dropout
from keras import optimizers
import talos as ta
from talos.model.normalizers import lr_normalizer
from talos.model.hidden_layers import hidden_layers
from elephas.utils.rdd_utils import to_simple_rdd
from elephas.spark_model import SparkModel

# TODO: Add distributed training using Elephas

# Constants
DATABASE = "seng-550.seng_550_data"
TABLE = "combined_technicals_fundamentals"
TARGET = "Price_Close"
RESULTS_TABLE_NAME = "DNN_Model_Results"
columns_to_drop = ["Instrument", "Date", "Company_Common_Name", "TRBC_Economic_Sector_Name",
                   "Country_of_Headquarters", "Exchange_Name", "Year"]

# Create the Spark Session on this node
spark = SparkSession \
  .builder \
  .master('yarn') \
  .appName('dnn') \
  .config('spark.executor.cores', '16') \
  .config('spark.executor.memory', '71680m') \
  .config('spark.executorEnv.LD_PRELOAD', 'libnvblas.so') \
  .getOrCreate()

# Create a temporary bucket
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
        'SELECT * FROM combined_data WHERE Instrument="FB.O" AND Year < 2018'
    )
    training_data = training_data.na.fill(0)

    # 2018 - now
    test_data = spark.sql(
        'SELECT * FROM combined_data WHERE Instrument="FB.O" AND Year >= 2018'
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


def create_model(x, y, x_val, y_val, params):

    model = models.Sequential()

    # Input Layer
    sgd = optimizers.Adam(lr=params['lr'])
    model.add(Dense(params["first_neuron"], activation="relu", input_shape=(x.shape[1],)))
    model.add(Dropout(params['dropout']))

    # Hidden layers
    hidden_layers(model, params, 1)

    # output layer
    model.add(Dense(1))
    model.compile(optimizer=sgd, loss="mse", metrics=["mse"])
    # model.summary()

    history = model.fit(x,
                        y,
                        epochs=params['epochs'],
                        batch_size=params['batch_size'],
                        verbose=0,
                        validation_data=(x_val, y_val))

    return history, model


def train_elephas_model(x_train, y_train):

    model = models.Sequential()

    # Input Layer
    sgd = optimizers.Adam(lr=0.1)
    model.add(Dense(128, activation="relu", input_shape=(x_train.shape[1],)))
    model.add(Dropout(0.1))

    # Hidden Layer
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.1))

    # output layer
    model.add(Dense(1))
    model.compile(optimizer=sgd, loss="mse", metrics=["mse"])
    model.summary()

    rdd = to_simple_rdd(sc, x_train, y_train)
    spark_model = SparkModel(model, frequency='epoch', mode='asynchronous')
    spark_model.fit(rdd, epochs=20, batch_size=32, verbose=0, validation_split=0.1)


# ------------- Driver ------------------------------------------------------
train, test = load_data()
train, test = preprocess_data(train, test)
train, test = vectorize_data(train, test)
train = train.toPandas()
test = test.toPandas()

# Training set
X_train = np.array(list(map(lambda x: x.toArray(), train["scaledFeatures"].values)))
y_train = train[TARGET].values
# Test set
X_test = np.array(list(map(lambda x: x.toArray(), test["scaledFeatures"].values)))
y_test = test[TARGET].values

dnn_params = {'lr': [0.01, 0.05, 1],
              'first_neuron': [64, 128, 256],
              'activation': ['relu'],
              'hidden_layers': [1, 2, 3],
              'batch_size': [32, 64, 128],
              'epochs': [10, 15, 25],
              'dropout': [0.01, 0.05, 0.1],
              'shapes': ['brick']}

# Train the model
dnn_model = ta.Scan(
    x=X_train,
    y=y_train,
    model=create_model,
    params=dnn_params,
    experiment_name="fb_stock"
)

results_df = dnn_model.data
results_df = spark.createDataFrame(results_df)
results_df.write.format('bigquery').option('table', 'seng-550.seng_550_data.{}'.format(RESULTS_TABLE_NAME)).mode('overwrite').save()
