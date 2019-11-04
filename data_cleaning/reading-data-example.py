from pyspark.sql import SparkSession

# Create a SparkSession under the name "technicals". Viewable via the Spark UI
spark = SparkSession.builder.master('yarn').appName(
    "technicals-stats").getOrCreate()


bucket = spark.sparkContext._jsc.hadoopConfiguration().get(
    'fs.gs.system.bucket')
spark.conf.set('temporaryGcsBucket', bucket)


technicals = spark.read.format('bigquery').option(
    'table', 'seng-550.seng_550_data.technicals').load().cache()

technicals.createOrReplaceTempView('technicals')

technicals.printSchema()
