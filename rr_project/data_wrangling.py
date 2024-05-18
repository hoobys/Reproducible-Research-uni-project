import pandas as pd
from sklearn.preprocessing import LabelEncoder
from config.const import SEED


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the data according to the methodology in "XGBoost Model and Its Application to Personal Credit Evaluation".

    This function converts the issue date to datetime, extracts the month, and applies encoding to the loan status.
    It then performs undersampling for each month, taking 4 times as many "good" clients as "bad" clients.

    Args:
        df (pd.DataFrame): The source data, which is credit card loan data from Lending Club for the year 2018.

    Returns:
        pd.DataFrame: The adjusted dataframe according to the methodology.

    """
    df["issue_d"] = pd.to_datetime(df.issue_d)
    df["month"] = df.issue_d.dt.month
    # apply encoding to loan status. The "good" customers are 'Fully Paid', 'Current', 'In Grace Period'.
    # they are assigned the label 0. The rest is labeled as 1 - bad clients.
    df["target"] = df.loan_status.apply(
        lambda x: 0 if x in ["Fully Paid", "Current", "In Grace Period"] else 1
    )
    print(f"Source data shape: {df.shape}. \nSource data target distribution:")
    print(df.target.value_counts())
    undersampled_df = pd.DataFrame()
    # run sampling for each month. For each month, we take 4 times as many good clients as bad clients.
    for month in range(1, 13):
        df_month = df[df["month"] == month]
        bad = df_month[df_month["target"] == 1]
        good = df_month[df_month["target"] == 0]
        good = good.sample(n=4 * len(bad), random_state=SEED)
        print(
            f"Bads for month {month}: {len(bad)}. Goods for month {month}: {len(good)}."
        )
        undersampled_df = pd.concat([undersampled_df, good, bad], axis=0)
    print(
        f"Undersampled data shape: {undersampled_df.shape}. \nUndersampled data target distribution:"
    )
    print(undersampled_df.target.value_counts())
    return undersampled_df


def missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the proportion of missing values in each column of the dataframe.

    This function first calculates the total number of missing values in each column.
    It then filters out the columns with no missing values.
    The remaining missing values are divided by the total number of rows to get the proportion of missing values.
    Finally, it sorts the columns in descending order based on the proportion of missing values.

    Args:
        df (pd.DataFrame): The dataframe for which to calculate the missing values.

    Returns:
        pd.DataFrame: A dataframe with the proportion of missing values in each column, sorted in descending order.

    """
    missing_values = df.isnull().sum()
    missing_values = missing_values[missing_values > 0]
    missing_values = missing_values / df.shape[0]
    missing_values = missing_values.sort_values(ascending=False)
    return missing_values


def preprocessing_feature_selection(preprocessed_df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs feature selection on the preprocessed dataframe.

    This function first drops the columns listed in `manual_scr` from the dataframe.
    It then calculates the proportion of missing values in each column and drops the columns with more than 50% missing values.

    Args:
        preprocessed_df (pd.DataFrame): The preprocessed dataframe from which to select features.

    Returns:
        pd.DataFrame: The dataframe after feature selection, with unnecessary and high-missing-value columns dropped.

    """
    manual_scr = [
        "id",
        "sub_grade",
        "emp_title",
        "url",
        "addr_state",
        "earliest_cr_line",
        "next_pymnt_d",
        "issue_d",
        "sec_app_earliest_cr_line",
        "recoveries",
        "purpose",
        "last_pymnt_d",
        "last_credit_pull_d",
        "total_rec_int",
        "hardship_start_date",
        "zip_code",
        "settlement_date",
        "hardship_end_date",
        "payment_plan_start_date",
        "debt_settlement_flag_date",
        "initial_list_status",
        "out_prncp",
        "out_prncp_inv",
        "total_pymnt_inv",
        "total_rec_prncp",
        "total_pymnt",
        "total_rec_late_fee",
        "collection_recovery_fee",
        "last_pymnt_amnt",
    ]
    pp_df = preprocessed_df.drop(columns=manual_scr)
    missing_ovr_50_perc_features = missing_values(pp_df)[missing_values(pp_df) > 0.5]
    print(
        f"Features with over 50% missing values = {len(missing_ovr_50_perc_features)}. {missing_ovr_50_perc_features.index}"
    )
    return pp_df.drop(columns=missing_ovr_50_perc_features.index)


def label_encode_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes all object columns of the dataframe using label encoding and prints the encoding dictionary for each column.

    This function uses the LabelEncoder from sklearn.preprocessing to transform the values of all object columns
    into numerical labels. It prints the mapping of original values to encoded values for each column.

    Args:
        df (pd.DataFrame): The dataframe containing the columns to encode.

    Returns:
        pd.DataFrame: The dataframe with all object columns encoded.

    """
    object_cols = df.select_dtypes(include=["object"]).columns
    for column_name in object_cols:
        le = LabelEncoder()
        df[column_name] = le.fit_transform(df[column_name].fillna(0).astype(str))
        # print the encoding dictionary for the column
        encoding_dict = dict(zip(le.classes_, le.transform(le.classes_)))
        print(f"Encoding for column '{column_name}': {encoding_dict}")
    return df


def fill_nulls_with_mean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fills null values in continuous data columns with the average value of non-null values in those columns.

    This function identifies continuous data columns (numeric types) in the dataframe and replaces their null values
    with the mean of the non-null values in the respective columns.

    Args:
        df (pd.DataFrame): The dataframe containing the columns to process.

    Returns:
        pd.DataFrame: The dataframe with null values filled with the mean of the respective columns.
    """
    numeric_cols = df.select_dtypes(include=["number"]).columns
    for column_name in numeric_cols:
        df[column_name] = df[column_name].fillna(df[column_name].mean())
    return df

def wrangle_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Wrangles the data by performing preprocessing, feature selection, label encoding, and filling null values.

    This function combines the following steps:
    1. Preprocess the data.
    2. Perform feature selection.
    3. Label encode all object columns.
    4. Fill null values with the mean of the respective columns.

    Args:
        df (pd.DataFrame): The source data, which is credit card loan data from Lending Club for the year 2018.

    Returns:
        pd.DataFrame: The wrangled dataframe after all the steps have been applied.

    """
    preprocessed_df = preprocess_data(df)
    feature_selected_df = preprocessing_feature_selection(preprocessed_df)
    label_encoded_df = label_encode_all(feature_selected_df)
    wrangled_df = fill_nulls_with_mean(label_encoded_df)
    return wrangled_df


if __name__ == "__main__":
    df = pd.read_csv("../data/credit_card_2018.csv")
    print(wrangle_data(df))