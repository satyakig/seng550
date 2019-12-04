# Overview #
Instructions for how to run each file so that models are properly generated
are described.

## K-means Clustering Model 
##### Files
`k_means_fundamentals.py, binning_k_means_results.py`

##### How to run
Run k_means_fundamentals.py first when new data is collected. This is because
k_means_fundamentals.py creates the k_means_data table, which is used in
binning_k_means.py.

##### Location of model outputs
The results of these models are multiple tables, which are written to BigQuery.
Refer to the files to see the name of the tables created.


## Dense Neural Network
##### Files
`Dense_Neural_Net.py`

##### How to run
Run `Dense_Neural_Net.py` only

##### Location of model outputs
The result of this model is a table, which is written to BigQuery.
The table is called `DNN_Model_Results`


## Graham Metrics
##### Files
`graham_metrics.py`

##### How to run
Run `graham_metrics.py` only

##### Location of model outputs
The result of this model is a table, which is written to BigQuery.
The table is called `DNN_Model_Results`


## Linear Regressor
##### Files
`Linear_Regressor.py`

##### How to run
Run `Linear_Regressor.py` only

##### Location of model outputs
The result of this model is a table, which is written to BigQuery.
The table is called `Lin_Reg_Results`


## LSTM
##### Files
`stock_pred_lstm_on_per_company_basis.py`

##### How to run
Run `stock_pred_lstm_on_per_company_basis.py` only

##### Location of model outputs
The result of this model is a csv, which is written to CloudStorage.
The file name is based on the # epochs and the companies it run on
