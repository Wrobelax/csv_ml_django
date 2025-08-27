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
    Linking all stepsL analysis -> ML

    :param file_path: path to csv file
    :return: dictionary with results.
    """

    df = pd.read_csv(file_path)

    # Basic analysis
    analysis = analyze_dataset(df)

    # Linear regression
    regression = run_regression(df)

    # ML classification
    ml_results = {}
    target_col = df.columns[-1]

    if df[target_col].dtype == "object":
        ml_results["random_forest_classification"] = random_forest_classification(df, target_col)
        ml_results["logistic_regression_classification"] = logistic_regression_classification(df, target_col)
        ml_results["decision_tree_classification"] = decision_tree_classification(df, target_col)
        ml_results["svm_classification"] = svm_classification(df, target_col)
        ml_results["knn_classification"] = knn_classification(df, target_col)
    else:
        ml_results = None

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