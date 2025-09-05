import pandas as pd
from .analysis import analyze_dataset
from .regression import run_regression
from .ml_classification import (
    random_forest_classification,
    logistic_regression_classification,
    knn_classification,
    decision_tree_classification,
    svm_classification
)
import numpy as np


def _nice_name(key: str) -> str:
    """
    Let's sprinkle this boi with cool name.
    """
    return key.replace("_", " ").replace("classification", "").title().strip()


def detect_task_type(df, target_col: str) -> str:
    """
    Automatically checks if data is for regression or classification.
    Rules:
        - Object target (string) -> classification
        - Numerical target (int) with <= 10 unique values -> classification
        - Numerical target (float) -> regression
    """

    target = df[target_col]
    if pd.api.types.is_object_dtype(target) or pd.api.types.is_categorical_dtype(target):
        return "classification"

    # Treat small-cardinary integers as classification (e.g. 0/1)
    if np.issubdtype(target.dtype, np.integer):
        if target.nunique() <= 10:
            return "classification"
        else:
            return "regression"

    if np.issubdtype(target.dtype, np.floating):
        return "regression"

    # Fallback - if only two unique values treat as classification.
    if target.nunique() <= 10:
        return "classification"

    return "regression"


def full_pipeline(file_path: str) -> dict:
    """
    Linking all stepsL analysis -> regression -> ML

    :param file_path: path to csv file
    :return: dictionary with results.
    """

    df = pd.read_csv(file_path)

    # Basic analysis
    try:
        analysis = analyze_dataset(df)
    except TypeError:
        analysis = analyze_dataset(file_path)

    # Protection from empty df
    if df.shape[1] == 0:
        return {
            "analysis": analysis,
            "regression": None,
            "ml_results": None,
        }

    # Setting targeted column (last)
    target_col = df.columns[-1]

    # Detect task type
    task_type = detect_task_type(df, target_col)

    regression = None
    ml_results = None

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    reg_target = None

    # Iterate from the end to pick "last suitable numeric"
    for c in reversed(numeric_cols):
        col = df[c]
        if np.issubdtype(col.dtype, np.floating):
            reg_target = c
            break

        if np.issubdtype(col.dtype, np.integer) and col.nunique() > 10:
            reg_target = c
            break

    if task_type == "regression" or (reg_target is not None and len(numeric_cols) >= 2):
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        try:
            regression = run_regression(df, reg_target)
        except Exception as e:
            regression = {"error": str(e)}

    if task_type == "classification":
        ml_results = {
            "random_forest_classification": random_forest_classification(df, target_col),
            "logistic_regression_classification": logistic_regression_classification(df, target_col),
            "decision_tree_classification": decision_tree_classification(df, target_col),
            "svm_classification": svm_classification(df, target_col),
            "knn_classification": knn_classification(df, target_col),
        }

    # Display name for each model.
    if ml_results:
        for k,v in list(ml_results.items()):
            display = _nice_name(k)
            ml_results[k]["model_name"] = display

    return {
        "analysis": analysis,
        "regression": regression,
        "ml_results": ml_results,
    }