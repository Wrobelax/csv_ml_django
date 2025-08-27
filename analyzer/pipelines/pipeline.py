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


def _nice_name(key: str) -> str:
    """
    Let's sprinkle this boi with cool name.
    """
    return key.replace("_", " ").replace("classification", "").title().strip()


def full_pipeline(file_path: str) -> dict:
    """
    Linking all stepsL analysis -> regression -> ML

    :param file_path: path to csv file
    :return: dictionary with results.
    """

    df = pd.read_csv(file_path)

    # Basic analysis
    analysis = analyze_dataset(df)

    # Protection from empty df
    if df.shape[1] == 0:
        return {
            "analysis": analysis,
            "regression": None,
            "ml_results": None,
        }

    # Setting targeted column (last)
    target_col = df.columns[-1]
    target_dtype = df[target_col].dtype

    regression = None
    ml_results = None

    # If target is numerical -> regression
    if pd.api.types.is_numeric_dtype(target_dtype):
        try:
            regression = run_regression(df)
        except Exception as e:
            regression = {"error": str(e)}


    # If target is categorical -> classification

    if pd.api.types.is_object_dtype(target_dtype) or pd.api.types.is_categorical_dtype(target_dtype):
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
            ml_results[k]["display_name"] = display

    return {
        "analysis": analysis,
        "regression": regression,
        "ml_results": ml_results,
    }