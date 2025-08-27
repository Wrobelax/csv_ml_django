import pandas as pd
from .analysis import analyze_dataset
from .regression import run_regression
from .ml_classification import random_forest_classification


def full_pipeline(file_path: str) -> dict:
    """
    Linking all stepsL analysis -> ML

    :param file_path: path to csv file
    :return: dictionary with results.
    """

    df = pd.read_csv(file_path)

    # Basic analysis
    analysis = analyze_dataset(df)

    # Linear regression
    regression = run_regression(df)

    # Random forest classification
    target_col = df.columns[-1]
    if df[target_col].dtype == "object":
        ml_classification = random_forest_classification(df, target_col=target_col)
    else:
        ml_classification = None

    return {
        "analysis": analysis,
        "regression": regression,
        "ml_classification": ml_classification,
    }