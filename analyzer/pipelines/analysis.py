def analyze_dataset(df):
    """
    Basic analysis of a data.
    :param df: Dataframe
    :return: Rows, columns, column names, average means.
    """
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": list(df.columns),
        "numeric_means": df.describe().loc["mean"].to_dict(),
    }