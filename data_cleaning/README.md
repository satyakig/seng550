## How to run on Google Dataproc
1. Open the Google Cloud Shell
2. Run this command:
```
gcloud dataproc jobs submit pyspark --cluster ${cluster_name} \
    --jars gs://spark-lib/bigquery/spark-bigquery-latest.jar \
    --region us-central1 \
    gs://seng550/data_cleaning_code/find_missing_data_stats.py
```

## Statistics
### Fundamental: Percentage of Nulls
```
Instrument is 0.00% nulls
Book_Value_Per_Share is 6.80% nulls
Cash_and_Short_Term_Investments is 23.29% nulls
Cost_of_Revenue__Total is 31.75% nulls
Current_Ratio is 28.83% nulls
Diluted_EPS_Excluding_Extraordinary_Items is 2.76% nulls
Diluted_EPS_Including_Extraordinary_Items is 2.71% nulls
EBIT is 1.69% nulls
EBIT_Margin__Percent is 8.46% nulls
Goodwill__Net is 54.49% nulls
Gross_Dividends___Common_Stock is 0.00% nulls
Gross_Margin__Percent is 33.20% nulls
Net_Income_Before_Taxes is 2.35% nulls
Normalized_Income_Avail_to_Cmn_Shareholders is 2.48% nulls
Operating_Expenses is 1.70% nulls
Operating_Income is 17.69% nulls
Operating_Margin__Percent is 26.18% nulls
Property_Plant_Equipment__Total___Net is 11.08% nulls
Quick_Ratio is 28.83% nulls
ROA_Total_Assets__Percent is 8.44% nulls
Revenue_Per_Share is 1.89% nulls
Date is 0.24% nulls
Tangible_Book_Value_Per_Share is 6.50% nulls
Total_Assets__Reported is 6.37% nulls
Total_Current_Liabilities is 28.78% nulls
Total_Current_Assets is 28.76% nulls
Total_Debt is 6.53% nulls
Total_Equity is 6.36% nulls
Total_Inventory is 52.80% nulls
Total_Liabilities is 6.44% nulls
Total_Long_Term_Debt is 6.63% nulls
Total_Receivables__Net is 28.50% nulls
Total_Revenue is 17.53% nulls
Total_Common_Shares_Outstanding is 6.54% nulls
Total_Debt_to_Total_Equity__Percent is 11.33% nulls
Company_Common_Name is 0.00% nulls
Exchange_Name is 0.58% nulls
Country_of_Headquarters is 0.00% nulls
TRBC_Economic_Sector_Name is 0.12% nulls
Has_Div is 0.00% nulls
```

### Technical: Percentage of Nulls
```
Total Rows:  24847880

Overall Stats
int64_field_0:  0, 0.00000% missing
Instrument:  0, 0.00000% missing
Date:  13838935, 55.69463% missing
Volume:  26784, 0.10779% missing
Formatted_Date:  13838935, 55.69463% missing
Enterprise_Value_To_Sales_Daily_Time_Series_Ratio:  16194179, 65.17328% missing
P_E_Daily_Time_Series_Ratio:  19134748, 77.00757% missing
Price:  13921093, 56.02527% missing
Price_To_Book_Value_Per_Share_Daily_Time_Series_Ratio:  15695695, 63.16714% missing
Price_To_Cash_Flow_Per_Share_Daily_Time_Series_Ratio:  18439440, 74.20931% missing
Price_To_Sales_Per_Share_Daily_Time_Series_Ratio:  16194764, 65.17564% missing
Total_Debt_To_EBITDA_Daily_Time_Series_Ratio:  19579413, 78.79712% missing
Total_Debt_To_Enterprise_Value_Daily_Time_Series_Ratio:  15811273, 63.63228% missing
One or more cols:  21096796, 84.90381% missing
Date, Volume or Price is missing:  13939762, 56.10041% missing
```