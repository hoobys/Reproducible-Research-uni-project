import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from rr_project.config.const import SEED


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
        "loan_status",
        "last_fico_range_high",
        "last_fico_range_low",
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


def iv_woe(data: pd.DataFrame, target: str, bins=10) -> pd.DataFrame:
    """
    Calculates the Information Value (IV) and Weight of Evidence (WoE) for all independent variables in the data.

    This function first checks if the independent variable is numeric and has more than 10 unique values. If so, it bins the variable into deciles.
    It then calculates the number of events (target = 1) and non-events (target = 0) in each bin.
    The WoE is calculated as the natural logarithm of the division of the percentage of non-events and the percentage of events.
    The IV is calculated as the product of the WoE and the difference between the percentage of events and the percentage of non-events.
    The function prints the IV for each variable and returns a DataFrame with the IV for all variables.

    The code author is Ailurophile, and the original code can be found at
    https://stackoverflow.com/questions/60892714/how-to-get-the-weight-of-evidence-woe-and-information-value-iv-in-python-pan

    Args:
        data (pd.DataFrame): The DataFrame containing the independent and dependent variables.
        target (str): The name of the target variable in the data.
        bins (int, optional): The number of bins to use for numeric variables with more than 10 unique values. Defaults to 10.

    Returns:
        pd.DataFrame: A DataFrame with the IV for all independent variables.
    """
    newDF = pd.DataFrame()

    # Extract Column Names
    cols = data.columns

    # Run WOE and IV on all the independent variables
    for ivars in cols[~cols.isin([target])]:
        if (data[ivars].dtype.kind in "bifc") and (len(np.unique(data[ivars])) > 10):
            binned_x = pd.qcut(data[ivars], bins, duplicates="drop")
            d0 = pd.DataFrame({"x": binned_x, "y": data[target]})
        else:
            d0 = pd.DataFrame({"x": data[ivars], "y": data[target]})

        # Calculate the number of events in each group (bin)
        d = d0.groupby("x", as_index=False).agg({"y": ["count", "sum"]})
        d.columns = ["Cutoff", "N", "Events"]

        # Calculate % of events in each group.
        d["% of Events"] = np.maximum(d["Events"], 0.5) / d["Events"].sum()

        # Calculate the non events in each group.
        d["Non-Events"] = d["N"] - d["Events"]
        # Calculate % of non events in each group.
        d["% of Non-Events"] = np.maximum(d["Non-Events"], 0.5) / d["Non-Events"].sum()

        # Calculate WOE by taking natural log of division of % of non-events and % of events
        d["WoE"] = np.log(d["% of Events"] / d["% of Non-Events"])
        d["IV"] = d["WoE"] * (d["% of Events"] - d["% of Non-Events"])
        d.insert(loc=0, column="Variable", value=ivars)
        temp = pd.DataFrame(
            {"Variable": [ivars], "IV": [d["IV"].sum()]}, columns=["Variable", "IV"]
        )
        newDF = pd.concat([newDF, temp], axis=0)

    return newDF


def iv_selection(data: pd.DataFrame, target: str, threshold=0.02) -> pd.DataFrame:
    """
    Selects the independent variables based on the Information Value (IV) of the variables.

    This function calculates the IV for all independent variables in the data and selects the variables with an IV
    greater than the threshold. It prints the IV for each variable and returns a DataFrame with the IV for all variables.

    Args:
        data (pd.DataFrame): The DataFrame containing the independent and dependent variables.
        target (str): The name of the target variable in the data.
        threshold (float, optional): The threshold value for selecting variables based on IV. Defaults to 0.02.

    Returns:
        pd.DataFrame: A DataFrame with the IV for all independent variables greater than the threshold.
    """
    iv_values = iv_woe(data, target)
    print(iv_values.sort_values("IV", ascending=False))
    print(
        "Variables with IV lower than threshold: ",
        iv_values[iv_values.IV < threshold].Variable.tolist(),
    )
    return data[iv_values[iv_values.IV >= threshold].Variable.tolist() + ["target"]]


def wrangle_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Wrangles the data by performing preprocessing, feature selection, label encoding, and filling null values.

    This function combines the following steps:
    1. Preprocess the data.
    2. Perform feature selection.
    3. Label encode all object columns.
    4. Fill null values with the mean of the respective columns.
    5. Perform Information Value (IV) selection on the data.

    Args:
        df (pd.DataFrame): The source data, which is credit card loan data from Lending Club for the year 2018.

    Returns:
        pd.DataFrame: The wrangled dataframe after all the steps have been applied.

    """
    preprocessed_df = preprocess_data(df)
    print("Preprocessed data shape: ", preprocessed_df.shape)
    feature_selected_df = preprocessing_feature_selection(preprocessed_df)
    print("Feature selected data shape: ", feature_selected_df.shape)
    label_encoded_df = label_encode_all(feature_selected_df)
    print("Label encoded data shape: ", label_encoded_df.shape)
    wrangled_df = fill_nulls_with_mean(label_encoded_df)
    print("Filled nulls data shape: ", wrangled_df.shape)
    df_iv_selection = iv_selection(wrangled_df, "target")
    print("IV selected data shape: ", df_iv_selection.shape)
    return df_iv_selection


if __name__ == "__main__":
    df = pd.read_csv("../data/credit_card_2018.csv")
    print(wrangle_data(df))
