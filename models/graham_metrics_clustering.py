# Running k-means clustering on the graham metrics

from pyspark.ml.clustering import KMeans
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import VectorAssembler

# Constants
DATABASE = 'seng-550.seng_550_data'
GRAHAM_TABLE = 'graham_metrics'

# Setup the PySpark session
spark = SparkSession \
    .builder \
    .master('yarn') \
    .config('spark.executor.cores', '16') \
    .config('spark.executor.memory', '71680m') \
    .config('spark.executorEnv.LD_PRELOAD', 'libnvblas.so') \
    .getOrCreate()

bucket = spark.sparkContext._jsc.hadoopConfiguration().get('fs.gs.system.bucket')
spark.conf.set('temporaryGcsBucket', bucket)

# Load data from Big Query
graham_metrics = spark.read.format('bigquery').option('table', '{}.{}'.format(DATABASE, GRAHAM_TABLE)).load().cache()
graham_metrics.createOrReplaceTempView('graham_metrics')

k_values = [2, 8, 15, 25, 45]
scores = []

for i in k_values:
    cols = ['metric1', 'metric2', 'metric3', 'metric4', 'metric5', 'metric6', 'metric7', 'metric8']
    assembler = VectorAssembler(inputCols=cols, outputCol='features')
    spDF = assembler.transform(graham_metrics)

    # Trains a k-means model.
    kmeans = KMeans().setK(i).setSeed(1).setFeaturesCol('features')
    model = kmeans.fit(spDF)

    # Make predictions
    predictions = model.transform(spDF)

    # Evaluate clustering by computing Silhouette score
    evaluator = ClusteringEvaluator()

    silhouette = evaluator.evaluate(predictions)

    scores.append((i, silhouette))
    predictions = predictions.drop('features')
    predictions.write.format('bigquery') \
        .option('table', DATABASE + '.graham_metrics_clustering_' + str(i))\
        .mode('overwrite')\
        .save()

# Shows the result.
for k_value, score in scores:
    print('K Value:', k_value, 'Score:', score)
