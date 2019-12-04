## How to run on Google DataProc manually
1. Get the GCloud CLI
2. Login into the GCloud CLI
3. Create a cluster using the `./create_cluster.bash` script
4. Run a job using the `./run_cluster.bash` script
5. Delete the cluster using the `./delete_cluster.bash` script

## Statistics
### Fundamental: Percentage of Nulls
Total rows: 195,636
```
Instrument:	0 nulls or 0.0000%
Book_Value_Per_Share: 13298 nulls or 6.7973%
Cash_and_Short_Term_Investments: 45573 nulls or 23.2948%
Cost_of_Revenue__Total: 62124 nulls or 31.7549%
Current_Ratio: 56408 nulls or 28.8331%
Diluted_EPS_Excluding_Extraordinary_Items: 5395 nulls or 2.7577%
Diluted_EPS_Including_Extraordinary_Items: 5298 nulls or 2.7081%
EBIT: 3304 nulls or 1.6889%
EBIT_Margin__Percent: 16558 nulls or 8.4637%
Goodwill__Net: 106602 nulls or 54.4900%
Gross_Dividends___Common_Stock: None nulls or 0.0000%
Gross_Margin__Percent: 64942 nulls or 33.1953%
Net_Income_Before_Taxes: 4604 nulls or 2.3534%
Normalized_Income_Avail_to_Cmn_Shareholders: 4844 nulls or 2.4760%
Operating_Expenses: 3330 nulls or 1.7021%
Operating_Income: 34613 nulls or 17.6926%
Operating_Margin__Percent: 51219 nulls or 26.1808%
Property_Plant_Equipment__Total___Net: 21672 nulls or 11.0777%
Quick_Ratio: 56408 nulls or 28.8331%
ROA_Total_Assets__Percent: 16511 nulls or 8.4397%
Revenue_Per_Share: 3707 nulls or 1.8948%
Date: 461 nulls or 0.2356%
Tangible_Book_Value_Per_Share: 12717 nulls or 6.5003%
Total_Assets__Reported: 12465 nulls or 6.3715%
Total_Current_Liabilities:56310 nulls or 28.7830%
Total_Current_Assets: 56268 nulls or 28.7616%
Total_Debt: 12768 nulls or 6.5264%
Total_Equity: 12440 nulls or 6.3587%
Total_Inventory: 103296 nulls or 52.8001%
Total_Liabilities: 12592 nulls or 6.4364%
Total_Long_Term_Debt: 12966 nulls or 6.6276%
Total_Receivables__Net:	55761 nulls or 28.5024%
Total_Revenue: 34292 nulls or 17.5285%
Total_Common_Shares_Outstanding: 12789 nulls or 6.5371%
Total_Debt_to_Total_Equity__Percent: 22175 nulls or 11.3348%
Company_Common_Name: None nulls or 0.0000%
Exchange_Name: 1129 nulls or 0.5771%
Country_of_Headquarters: 0 nulls or 0.0000%
TRBC_Economic_Sector_Name: 238 nulls or 0.1217%
Has_Div: 0 nulls or 0.0000%

One or more cols:		151423 nulls or 77.4004%
Date or Instrument or Total_Debt or Total_Revenue or Total_Current_Assets or Total_Current_Liabilities:		57226 nulls or 29.2513%
```

### Technical: Percentage of Nulls
Total rows: 24,847,880
```
int64_field_0: 0 nulls or 0.0000%
Instrument: 0 nulls or 0.0000%
Enterprise_Value_To_Sales__Daily_Time_Series_Ratio_: 16194179 nulls or 65.1733%
P_E__Daily_Time_Series_Ratio_: 19134748 nulls or 77.0076%
Price_Close: 13921093 nulls or 56.0253%
Date: 13838935 nulls or 55.6946%
Price_To_Book_Value_Per_Share__Daily_Time_Series_Ratio_: 15695695 nulls or 63.1671%
Price_To_Cash_Flow_Per_Share__Daily_Time_Series_Ratio_: 18439440 nulls or 74.2093%
Price_To_Sales_Per_Share__Daily_Time_Series_Ratio_: 16194764 nulls or 65.1756%
Total_Debt_To_EBITDA__Daily_Time_Series_Ratio_: 19579413 nulls or 78.7971%
Total_Debt_To_Enterprise_Value__Daily_Time_Series_Ratio_: 15811273 nulls or 63.6323%
Volume: 26784 nulls or 0.1078%

One or more cols: 21096796 nulls or 84.9038%
Date or Instrument or Price_Close or Volume: 13939762 nulls or 56.1004%
```

### Joined: Percentage of Nulls
Total rows: 10,059,333
```
Instrument:	0 nulls or 0.0000%
Quarter: 0 nulls or 0.0000%
Year: 0 nulls or 0.0000%
Enterprise_Value_To_Sales__Daily_Time_Series_Ratio_: 2229652 nulls or 22.1650%
P_E__Daily_Time_Series_Ratio_: 4875187 nulls or 48.4643%
Price_Close: 0 nulls or 0.0000%
Date: 0 nulls or 0.0000%
Price_To_Book_Value_Per_Share__Daily_Time_Series_Ratio_: 1804846 nulls or 17.9420%
Price_To_Cash_Flow_Per_Share__Daily_Time_Series_Ratio_: 4203399 nulls or 41.7861%
Price_To_Sales_Per_Share__Daily_Time_Series_Ratio_: 2230051 nulls or 22.1690%
Total_Debt_To_EBITDA__Daily_Time_Series_Ratio_: 5266478 nulls or 52.3541%
Total_Debt_To_Enterprise_Value__Daily_Time_Series_Ratio_: 1917409 nulls or 19.0610%
Volume: 0 nulls or 0.0000%
Book_Value_Per_Share: 143219 nulls or 1.4237%
Cash_and_Short_Term_Investments: 1879654 nulls or 18.6857%
Cost_of_Revenue__Total: 2914435 nulls or 28.9724%
Current_Ratio: 2348585 nulls or 23.3473%
Diluted_EPS_Excluding_Extraordinary_Items: 69256 nulls or 0.6885%
Diluted_EPS_Including_Extraordinary_Items: 67849 nulls or 0.6745%
EBIT: 42749 nulls or 0.4250%
EBIT_Margin__Percent: 573532 nulls or 5.7015%
Goodwill__Net: 5076309 nulls or 50.4637%
Gross_Dividends___Common_Stock: 0 nulls or 0.0000%
Gross_Margin__Percent: 3034959 nulls or 30.1706%
Net_Income_Before_Taxes: 62300 nulls or 0.6193%
Normalized_Income_Avail_to_Cmn_Shareholders: 63827 nulls or 0.6345%
Operating_Expenses: 43530 nulls or 0.4327%
Operating_Income: 1643459 nulls or 16.3377%
Operating_Margin__Percent: 2336658 nulls or 23.2288%
Property_Plant_Equipment__Total___Net: 479814 nulls or 4.7698%
Quick_Ratio: 2348585 nulls or 23.3473%
ROA_Total_Assets__Percent: 209144 nulls or 2.0791%
Revenue_Per_Share: 43802 nulls or 0.4354%
Tangible_Book_Value_Per_Share: 135324 nulls or 1.3453%
Total_Assets__Reported: 132040 nulls or 1.3126%
Total_Current_Liabilities: 2344888 nulls or 23.3106%
Total_Current_Assets: 2342550 nulls or 23.2873%
Total_Debt: 147459 nulls or 1.4659%
Total_Equity: 130216 nulls or 1.2945%
Total_Inventory: 4840608 nulls or 48.1206%
Total_Liabilities: 136422 nulls or 1.3562%
Total_Long_Term_Debt: 157205 nulls or 1.5628%
Total_Receivables__Net: 2363993 nulls or 23.5005%
Total_Revenue: 1643426 nulls or 16.3373%
Total_Common_Shares_Outstanding: 137709 nulls or 1.3690%
Total_Debt_to_Total_Equity__Percent: 547490 nulls or 5.4426%
Company_Common_Name: 0 nulls or 0.0000%
Exchange_Name: 39958 nulls or 0.3972%
Country_of_Headquarters: 0 nulls or 0.0000%
TRBC_Economic_Sector_Name: 5744 nulls or 0.0571%
Has_Div: 0 nulls or 0.0000%

One or more cols: 8876084 nulls or 88.2373%
```