from sklearn.linear_model import LinearRegression
import numpy as np


def run_regression(df):
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.shape[1] < 2:
        return None

    x = numeric_df.iloc[:,[0]].values
    y = numeric_df.iloc[:,1].values

    model = LinearRegression()
    model.fit(x,y)

    return {
        "coefficients": model.coef_.tolist(),
        "intercept": float(model.intercept_),
        "r2_score": float(model.score(x,y))
    }