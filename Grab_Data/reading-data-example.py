from pyspark.sql.types import StructField, StructType, StringType, IntegerType, FloatType
from pyspark.sql import SparkSession
from py4j.protocol import Py4JJavaError

# Create a SparkSession under the name "technicals". Viewable via the Spark UI
spark = SparkSession.builder.master('yarn').appName(
    "technicals-stats").getOrCreate()


bucket = spark.sparkContext._jsc.hadoopConfiguration().get(
    'fs.gs.system.bucket')
spark.conf.set('temporaryGcsBucket', bucket)


technicals = spark.read.format('bigquery').option(
    'table', 'seng-550.seng_550_data.technicals').load()
technicals.createOrReplaceTempView('technicals')

table = spark.sql(
    'SELECT * from technicals')
table.show()
table.printSchema()
