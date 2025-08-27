from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def random_forest_classification(df, target_col=None):
    """
    Random forest classification. If target_col not set it becomes last colum from the data.
    :param df: Dataframe
    :param target_col: Target column for the model.
    :return: Accuracy score, report.
    """

    if target_col is None:
        target_col = df.columns[-1]

    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols]
    y = df[target_col]

    # Training and testing division
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "report": classification_report(y_test, y_pred, output_dict=True),
    }