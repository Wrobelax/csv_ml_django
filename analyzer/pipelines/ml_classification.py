from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import numpy as np


def to_python(obj):
    """
    Recursively convert numpy scalar/arrays to python built-in types.
    """
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_python(x) for x in obj]

    return obj


def train_model(model, X, y):
    """
    Basic function for data training and prediction.
    :return: Accuracy, report, confusion matrix, classes, feature importance.
    """

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)


    # Try to get probability estimates
    y_proba = None
    try:
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)
        elif hasattr(model, 'decision_function'):
            # Fallback - decision model may be available depending on model
            y_proba = model.decision_function(X_test)
    except Exception:
        y_proba = None

    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()
    classes = list(np.unique(y_test).astype(str).tolist())


    # Feature importances if available
    feature_importances = None
    if hasattr(model, "feature_importances_"):
        fi_vals = model.feature_importances_.tolist()
        feature_importances = {col: float(val) for col, val in zip(X.columns.tolist(), fi_vals)}
    elif hasattr(model, "coef_"):
        # Logistic regression: coef_ may be (n_classes, n_features)
        coef = model.coef_
        if coef.ndim == 1:
            feature_importances = {col: float(val) for col, val in zip(X.columns.tolist(), coef.tolist())}
        else:
            # Multi-class: sum absolute values across classes
            abs_sum = np.abs(coef).sum(axis=0)
            feature_importances = {col: float(val) for col, val in zip(X.columns.tolist(), abs_sum.tolist())}



    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "report": report,
        "confusion_matrix": cm,
        "classes": classes,
        "feature_importances": feature_importances,
        "y_test": np.asarray(y_test).tolist(),
        "y_pred": np.asarray(y_pred).tolist(),
        "y_proba": None if y_proba is None else (np.asarray(y_proba).tolist()),
    }


def random_forest_classification(df, target_col=None):
    """
    Random forest classification.
    :param df: Dataframe
    :param target_col: Target column for the model.
    :return: Accuracy score, report.
    """

    if target_col is None:
        target_col = df.columns[-1]
    X = df.drop(columns=[target_col]).select_dtypes(include=[np.number])
    y = df[target_col]
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    return train_model(model, X, y)


def logistic_regression_classification(df, target_col=None):
    """
    Logistic regression classification.
    :param df: Dataframe
    :param target_col: Target column for the model.
    :return: Accuracy score, report.
    """

    if target_col is None:
        target_col = df.columns[-1]
    X = df.drop(columns=[target_col]).select_dtypes(include=[np.number])
    y = df[target_col]
    model = LogisticRegression(max_iter=200)

    return train_model(model, X, y)


def knn_classification(df, target_col=None):
    """
    Knearest neighbours classification.
    :param df: Dataframe
    :param target_col: Target column for the model.
    :return: Accuracy score, report.
    """

    if target_col is None:
        target_col = df.columns[-1]
    X = df.drop(columns=[target_col]).select_dtypes(include=[np.number])
    y = df[target_col]
    model = KNeighborsClassifier()

    return train_model(model, X, y)


def decision_tree_classification(df, target_col=None):
    """
    Decision tree classification.
    :param df: Dataframe
    :param target_col: Target column for the model.
    :return: Accuracy score, report.
    """

    if target_col is None:
        target_col = df.columns[-1]
    X = df.drop(columns=[target_col]).select_dtypes(include=[np.number])
    y = df[target_col]
    model = DecisionTreeClassifier(random_state=42)

    return train_model(model, X, y)


def svm_classification(df, target_col=None):
    """
    SVM classification.
    :param df: Dataframe
    :param target_col: Target column for the model.
    :return: Accuracy score, report.
    """

    if target_col is None:
        target_col = df.columns[-1]
    X = df.drop(columns=[target_col]).select_dtypes(include=[np.number])
    y = df[target_col]
    model = SVC(probability=True)

    return train_model(model, X, y)