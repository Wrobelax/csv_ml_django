from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd



def run_regression(df, target_col=None):
    """
    Simple linear regression. If target_col not given, it takes last numeric column.
    :param df: Dataframe, target column.
    :return: Coefficients, intercept, r2 score.
    """

    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)


    return {
        "coefficients": model.coef_.tolist(),
        "intercept": float(model.intercept_),
        "r2_score": r2_score(y_test, y_pred),
        "test_values": {
            "y_test": y_test.tolist(),
            "y_pred": y_pred.tolist(),
        }
    }