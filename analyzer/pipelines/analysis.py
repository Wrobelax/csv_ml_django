def analyze_dataset(df):
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": list(df.columns),
        "numeric_means": df.describe().loc["mean"].to_dict(),
    }