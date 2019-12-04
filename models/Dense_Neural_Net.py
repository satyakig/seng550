"""
Run:  bash ../gcp_scripts/run_cluster.bash ./Dense_Neural_Net.py james-cluster us-central1
"""
from pyspark.sql import SparkSession
import numpy as np
from pyspark.sql.functions import col
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import VectorAssembler
from keras import models
from keras.layers import Dense, Dropout
from keras import optimizers
from talos.model.hidden_layers import hidden_layers
from elephas.utils.rdd_utils import to_simple_rdd
from elephas.spark_model import SparkModel
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# Constants
DATABASE = "seng-550.seng_550_data"
TABLE = "combined_technicals_fundamentals"
TARGET = "Price_Close"
VIEW_NAME = 'combined_data'
RESULTS_TABLE_NAME = "DNN_Model_Results"
columns_to_drop = ["Instrument", "Date", "Company_Common_Name", "TRBC_Economic_Sector_Name",
                   "Country_of_Headquarters", "Exchange_Name", "Year"]
TECH_SECTOR = 'Technology'
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
  .appName('dnn') \
  .config('spark.executor.cores', '16') \
  .config('spark.executor.memory', '71680m') \
  .config('spark.executorEnv.LD_PRELOAD', 'libnvblas.so') \
  .getOrCreate()

# Create a temporary bucket
bucket = spark.sparkContext._jsc.hadoopConfiguration().get('fs.gs.system.bucket')
spark.conf.set('temporaryGcsBucket', bucket)
sc = spark.sparkContext


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
        query = 'SELECT * FROM {} WHERE Year >= {}'.format(VIEW_NAME, end_year)
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
    input_columns = training_data.columns
    input_columns.remove(TARGET)
    print("Using these features: {}".format(input_columns))
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


def to_numpy_array(train_array, test_array):
    train_array = train_array.toPandas()

    # Training set
    train_features = np.array(list(map(lambda x: x.toArray(), train_array["scaledFeatures"].values)))
    train_labels = train[TARGET].values

    new_test_data = dict()
    for company in test_array:
        company_data = test_array[company].toPandas()
        test_features = np.array(list(map(lambda x: x.toArray(), company_data["scaledFeatures"].values)))
        test_labels = company_data[TARGET].values

        new_test_data[company] = {'scaledFeatures': test_features, TARGET: test_labels}

    return train_features, train_labels, new_test_data


def train_elephas_model(x, y):
    model = models.Sequential()

    # Input Layer
    sgd = optimizers.Adam(lr=0.01)
    model.add(Dense(128, activation="relu", input_shape=(x.shape[1],)))
    model.add(Dropout(0.1))

    # Hidden Layer
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.1))

    # output layer
    model.add(Dense(1))
    model.compile(optimizer=sgd, loss="mse", metrics=["mse"])
    model.summary()

    rdd = to_simple_rdd(sc, x, y)
    spark_model = SparkModel(model, frequency='epoch', mode='asynchronous')
    spark_model.fit(rdd, epochs=25, batch_size=64, verbose=1, validation_split=0.2)

    return spark_model


def train_and_pred(train_features, train_labels, test_data):
    # train the linear regression model

    dnn = train_elephas_model(train_features, train_labels)

    predictions_dict = dict()
    for company in test_data:
        test_company_data = test_data[company]
        scaled_features = test_company_data["scaledFeatures"]
        labels = test_company_data[TARGET]

        dnn_predictions = dnn.predict(scaled_features)

        r_score = r2_score(labels, dnn_predictions)
        mean_square = mean_squared_error(labels, dnn_predictions)

        print("Testing r2 on {}: {}".format(company, r_score))
        print("MSE on {}  = {}".format(company, mean_square))

    return predictions_dict


df_arr = []

print("_____________________Training on all companies____________________________")

train, test = load_data()
train, test = preprocess_data(train, test)
train, test = vectorize_data(train, test)
X_train, y_train, test = to_numpy_array(train, test)
predictions = train_and_pred(X_train, y_train, test)

print("_____________________Moving onto sector training____________________________")

train, test = load_data(True)
train, test = preprocess_data(train, test)
train, test = vectorize_data(train, test)
X_train, y_train, test = to_numpy_array(train, test)
predictions = train_and_pred(X_train, y_train, test)
