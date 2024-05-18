import pandas as pd


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess data, according to methodology in "XGBoost Model and Its Application to Personal Credit Evaluation:
    :param df: source data - pd.DataFrame, credit card loan data from Lending Club, for the year 2018
    :return: adjusted dataframe, according to methodology
    """
    df['issue_d'] = pd.to_datetime(df.issue_d)
    df['month'] = df.issue_d.dt.month
    # apply encoding to loan status. The "good" customers are 'Fully Paid', 'Current', 'In Grace Period'.
    # they are assigned the label 0. The rest is labeled as 1 - bad clients.
    df['target'] = df.loan_status.apply(lambda x: 0 if x in ['Fully Paid', 'Current', 'In Grace Period'] else 1)
    print(f'Source data shape: {df.shape}. \nSource data target distribution:')
    print(df.target.value_counts())
    undersampled_df = pd.DataFrame()
    # run sampling for each month. For each month, we take 4 times as many good clients as bad clients.
    for month in range(1, 13):
        df_month = df[df['month'] == month]
        bad = df_month[df_month['target'] == 1]
        good = df_month[df_month['target'] == 0]
        good = good.sample(n=4 * len(bad), random_state=42)
        print(f'Bads for month {month}: {len(bad)}. Goods for month {month}: {len(good)}.')
        undersampled_df = pd.concat([undersampled_df, good, bad], axis=0)
    print(f'Undersampled data shape: {undersampled_df.shape}. \nUndersampled data target distribution:')
    print(undersampled_df.target.value_counts())
    return undersampled_df
