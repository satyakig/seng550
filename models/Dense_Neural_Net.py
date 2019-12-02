"""
Run:  bash ../gcp_scripts/run_cluster.bash ./Dense_Neural_Net.py james-cluster us-central1
"""
from pyspark.sql import SparkSession
#import systemml
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from pyspark.ml.feature import VectorAssembler
from keras import models
from keras import layers
from keras import optimizers

# TODO: Add automatic parameter tuning
# TODO: Add dropout layers
# TODO: Add regularization errors

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


def load_train_test_split(shuffle_data=True):
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
    train_features = np.array(train_df.select('features').collect()).reshape(-1, len(num_features))
    train_labels = np.array(train_df.select(TARGET).collect()).reshape(-1)
    print("Training set has size {} for X and size {} for y".format(train_features.shape, train_labels.shape))
    # train_df.show()

    # Testing: 2018 - now
    test_data = spark.sql(
        'SELECT * FROM combined_data WHERE Instrument="FB.O" AND Year >= 2018'
    )
    test_data = test_data.na.fill(0)
    test_df = vector_assembler.transform(test_data)
    test_features = np.array(test_df.select('features').collect()).reshape(-1, len(num_features))
    test_labels = np.array(test_df.select(TARGET).collect()).reshape(-1)
    print("Test set has size {} for X and size {} for y".format(test_features.shape, test_labels.shape))
    # test_df.show()

    scaler = StandardScaler(copy=False)
    scaler.fit(train_features)

    print(test_features[0])
    scaler.transform(train_features)
    scaler.transform(test_features)
    print(test_features[1])
    # # Shuffle all of the data ?
    if shuffle_data is True:
        shuffled_training = shuffle(train_features, train_labels)
        train_features, train_labels = shuffled_training[0], shuffled_training[1]
        shuffled_test = shuffle(test_features, test_labels)
        test_features, test_labels = shuffled_test[0], shuffled_test[1]

    return train_features, test_features, train_labels, test_labels


def create_model(features):
    sgd = optimizers.Adam(lr=0.1)
    model = models.Sequential()
    model.add(layers.Dense(256, activation="relu", input_shape=(features.shape[1],)))
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dense(1))
    model.compile(optimizer=sgd, loss="mean_squared_error", metrics=["mae"])
    model.summary()

    return model


# ------------- Driver ------------------------------------
X_train, X_test, y_train, y_test = load_train_test_split(shuffle_data=True)
model = create_model(X_train)
model.fit(X_train,
          y_train,
          epochs=40,
          batch_size=128,
          verbose=1,
          validation_data=(X_test, y_test))

# Evaluate the models R2 score here
prediction = model.predict(X_test).reshape(-1,)
print(y_test.shape)
print(y_test)
print(prediction.shape)
print(prediction)
r2 = r2_score(y_test, prediction)
print("r2 score is {}".format(r2))
