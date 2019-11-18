README

# Overview #
Instructions for how to run each file so that models are properly generated
are described.

## K-means Clustering Model ##

** Files: **
k_means_fundamentals.py, binning_k_means.py

** How to run: **
Run k_means_fundamentals.py first when new data is collected. This is because
k_means_fundamentals.py creates the k_means_data table, which is used in
binning_k_means.py.


** Location of model outputs: **
The results of these models are tables, which are written to BigQuery.
Refer to the files to see the name of the tables created.
