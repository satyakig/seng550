import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from sklearn.metrics import mean_squared_error


# Constants
DATABASE = "seng-550.seng_550_data"
TABLE = "combined_technicals_fundamentals"
TARGET = "Price_Close"
VIEW_NAME = 'combined_data'
# columns_to_drop = ["Instrument", "Date", "Company_Common_Name", "TRBC_Economic_Sector_Name",
#                    "Country_of_Headquarters", "Exchange_Name", "Year"]
columns_to_drop = []
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
    # train_data = train_data.select(*(col(c).cast('float').alias(c) for c in train_data.columns))
    train_data = train_data.fillna(0)

    test_dict = dict()
    for test_company in companies_to_check:
        company_data = test_data.filter(test_data.Instrument == test_company).drop(*columns_to_drop)

        if company_data.count() > 0:
            # company_data = company_data.select(*(col(c).cast('float').alias(c) for c in company_data.columns))
            company_data = company_data.fillna(0)
            test_dict[test_company] = company_data

    return train_data, test_dict


def year_over_year_model(data):
    '''
    A simple model that uses average year over year growth to predict stock
    price. Predicted stock price is the same every day for a year and equals
    previous year average * (1+ average_growth_rate).

    Args:
        data: scaled data for training.

    Returns:
        prices: actual and predicted prices for each day.
    '''

    prices = data.loc[:, ['Price_Close']]
    prices['year'] = pd.to_datetime(prices.index).year
    avg = prices.groupby(['year']).mean()
    avg_yearly_growth = ((avg[['Price_Close']] - avg.shift()[['Price_Close']]) \
                         / avg[['Price_Close']]).dropna()

    growth_factor = 1 + avg_yearly_growth.mean()[0]
    avg['next_year_price'] = avg[['Price_Close']] * growth_factor
    avg['next_year'] = avg.index + 1
    avg = avg.loc[:, ['next_year_price', 'next_year']]
    avg = avg.rename(columns={'next_year': 'year'})
    # prices = prices.join(avg, on='year')
    prices = prices.merge(avg.reset_index(drop=True), on='year', how='left').dropna()
    # avg.merge(avg_yearly_growth, left_index=True, right_index=True)

    return prices


def clean_up_input_data(df_data):
    '''
    Clean up data read in. This involves subsetting out desired columns,
    reordering columns, and scaling data.

    Args:
        df_data: original read in data.

    Return:
        Cleaned up dataframe.
    '''
    # fund_cols = ['Gross_Dividends___Common_Stock',
    #              'Net_Income_Before_Taxes', 'Normalized_Income_Avail_to_Cmn_Shareholders',
    #              'Operating_Expenses', 'EBIT', 'Total_Assets__Reported',
    #              'Total_Debt', 'Total_Equity', 'Total_Liabilities', 'Total_Long_Term_Debt']
    #
    # tech_cols = ['Date', 'Volume']
    #
    # label_cols = ['Price_Close']
    #
    # all_cols = tech_cols + fund_cols + label_cols

    # df_model_data = df_data.loc[:, all_cols]

    # # # Drop any Null data (should already by done by cleaning)
    # df_model_data = df_data.dropna()

    # Sort by date
    df_model_data = df_data.sort_values(by='Date').reset_index(drop=True)

    # Set date as index (it is not used as feature in training, but rather the
    # sequence of inputs is what is important).
    df_model_data = df_model_data.set_index('Date')

    return df_model_data


# Driver
train, test = load_data(tech_only=True)
train, test = preprocess_data(train, test)
for company in test:
    df = test[company].toPandas()
    df = clean_up_input_data(df)
    df = year_over_year_model(df)

    print(df["Price_Close"].values)
    print(df["Price_Close"].values.shape)
    print(df["next_year_price"].values.shape)
    mean_square = mean_squared_error(df["Price_Close"].values, df["next_year_price"].values)

    print("The mean squarred error for {} is {}".format(company, mean_square))

