'''
Run K-means clustering on the fundamental data.
Goal is to get an idea of value by looking company's earnings, rev, etc. clustering
company's with similar performance, and then from there be able to identify what
company's seem to be strong.

PCA is used to determine the 5 most influential fundamental metrics, and those
metrics are used to perform the clustering.
'''

from pyspark.sql.functions import isnan, when, count, col
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.feature import PCA
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler

# Constants
DATABASE = 'seng-550.seng_550_data'
FUND_TABLE = 'fundamentals_cleaned'

# Create the Spark Session on this node
spark = SparkSession \
    .builder \
    .master('yarn') \
    .appName('k_means_fundamentals') \
    .config('spark.executor.cores', '16') \
    .config('spark.executor.memory', '71680m') \
    .config('spark.executorEnv.LD_PRELOAD', 'libnvblas.so') \
    .getOrCreate()

# Use the Cloud Storage bucket for temporary BigQuery export data used
# by the connector. This assumes the Cloud Storage connector for
# Hadoop is configured.
bucket = spark.sparkContext._jsc.hadoopConfiguration().get('fs.gs.system.bucket')
spark.conf.set('temporaryGcsBucket', bucket)

# Load data from Big Query
fund_data = spark.read.format('bigquery').option('table', '{}.{}'.format(
    DATABASE, FUND_TABLE)).load().cache()
fund_data.createOrReplaceTempView('fund_data')

print('Read in data: ')
fund_data.show()

# Fundamental data to use for clustering.
# Columns selected were required to have less than 8% of the column entries be
# null to avoid having too much of the dataset cropped out.
fund_vals_to_normalize = ['Gross_Dividends___Common_Stock',
  'Net_Income_Before_Taxes', 'Normalized_Income_Avail_to_Cmn_Shareholders',
  'Operating_Expenses', 'EBIT', 'Total_Assets__Reported',
  'Total_Debt', 'Total_Equity', 'Total_Liabilities', 'Total_Long_Term_Debt']

normalizer = 'Total_Common_Shares_Outstanding'

existing_per_share_features = ['Revenue_Per_Share', 'Book_Value_Per_Share',
  'Tangible_Book_Value_Per_Share']

# Create new dataframe with normalized data on a per share basis.
fund_data_normalized = fund_data

# Normalize each fundamental value to a per share basis.
for norm in fund_vals_to_normalize:
    fund_data_normalized = fund_data_normalized.withColumn(
      norm + '_per_share', col(norm)/col(normalizer))

print('Data with per share vals added: ')
fund_data_normalized.show()

# Get list of all features on per share basis.
all_features_ps = existing_per_share_features + \
  [norm + '_per_share' for norm in fund_vals_to_normalize]

print('ALL features: ')
print(all_features_ps)

# Need to include Instrument, Year, and Quarter for joining with main tables
# and to look at temporal changes.
data_for_clustering = ['Instrument', 'Year', 'Quarter'] + all_features_ps

fund_data_normalized = fund_data_normalized.select(
  data_for_clustering).na.drop()

print('Preprocessed data: ')
fund_data_normalized.show()

# Put Gross_Dividends___Common_Stock_per_share at end of list so it is never
# the denominator. This is because this column has a lot of 0 entries, which is
# when a company does not give out a dividend.
all_features_ps.remove('Gross_Dividends___Common_Stock_per_share')
all_features_ps.append('Gross_Dividends___Common_Stock_per_share')

N = len(all_features_ps)

clustering_col_names = []

# Compute ratios of all metrics, as this allows all companies to be compared.
for i in range(N-1):
    for j in range(i+1,N):
        cur_name = all_features_ps[j] + '_to_' + all_features_ps[i]
        clustering_col_names.append(cur_name)
        fund_data_normalized = fund_data_normalized.withColumn(
          cur_name, col(all_features_ps[j])/col(all_features_ps[i]))


print('Normalized subset of data for clustering: ')
fund_data_normalized.show()

# Cannot have null values inputted to PCA.
fund_data_normalized = fund_data_normalized.na.drop()

# Pyspark k-means requires vector of features.
vecAssembler = VectorAssembler(inputCols=clustering_col_names,
  outputCol='features')

# Crete featrues vector from fund_data_normalized dataframe.
df_kmeans = vecAssembler.transform(fund_data_normalized).select('Instrument',
  'Year', 'Quarter', 'features')

print('Data in VectorAssembler format: ')
df_kmeans.show()

# *** IMPORTANT ***
# Scale results of each column.
# Scaling is required for PCA to work correctly.
scaler = MinMaxScaler(inputCol='features', outputCol='scaledFeatures')
scalerModel = scaler.fit(df_kmeans)
scaledData = scalerModel.transform(df_kmeans)

# Use PCA to determine 5 features that explain variance the most.
pca = PCA(k=5, inputCol='scaledFeatures', outputCol='pcaFeatures')
model = pca.fit(scaledData)

# Get pcaFeatures for dataset.
result = model.transform(scaledData).select('Instrument','Year', 'Quarter',
  'pcaFeatures')
print('Show PCA results: ')
result.show()

# Perform k-means clustering on top 5 principle components.
k = 5
kmeans = KMeans().setK(k).setSeed(1).setFeaturesCol('pcaFeatures')
model_kmeans = kmeans.fit(result)
centers = model_kmeans.clusterCenters()

print('Cluster Centers: ')
for center in centers:
    print(center)

# Assign clusters to each row.
transformed = model_kmeans.transform(result).select('Instrument', 'Year',
  'Quarter', col('prediction').alias('cluster'))
print('Show cluster each Instrument belongs to: ')
transformed.show()

# Write data to bigquery for further analysis.
transformed.write.format('bigquery') \
  .option('table', DATABASE + '.k_means_data') \
  .mode('overwrite') \
  .save()
