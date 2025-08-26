import pandas as pd
from .analysis import analyze_dataset
from .regression import run_regression


def full_pipeline(file_path: str) -> dict:
    """
    Linking all stepsL analysis -> ML

    :param file_path: path to csv file
    :return: dictionary with results.
    """

    df = pd.read_csv(file_path)

    result = {
        "analysis": analyze_dataset(df),
        "regression": run_regression(df)
    }

    return result