from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


def train_model(df, model_class, target_col= None):
    """
    Basic function for data training.
    :param df: Dataframe
    :param model_class: Model class
    :param target_col: Column targeted for training. If none -> Last column from the data.
    :return: Accuracy, report.
    """

    if target_col is None:
        target_col = df.columns[-1]

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = model_class()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "report": classification_report(y_test, y_pred, output_dict=True),
    }



def random_forest_classification(df, target_col=None):
    """
    Random forest classification.
    :param df: Dataframe
    :param target_col: Target column for the model.
    :return: Accuracy score, report.
    """
    return train_model(df, RandomForestClassifier, target_col)


def logistic_regression_classification(df, target_col=None):
    """
    Logistic regression classification.
    :param df: Dataframe
    :param target_col: Target column for the model.
    :return: Accuracy score, report.
    """
    return train_model(df, LogisticRegression, target_col)


def knn_classification(df, target_col=None):
    """
    Knearest neighbours classification.
    :param df: Dataframe
    :param target_col: Target column for the model.
    :return: Accuracy score, report.
    """
    return train_model(df, KNeighborsClassifier, target_col)

def decision_tree_classification(df, target_col=None):
    """
    Decision tree classification.
    :param df: Dataframe
    :param target_col: Target column for the model.
    :return: Accuracy score, report.
    """
    return train_model(df, DecisionTreeClassifier, target_col)


def svm_classification(df, target_col=None):
    """
    SVM classification.
    :param df: Dataframe
    :param target_col: Target column for the model.
    :return: Accuracy score, report.
    """
    return train_model(df, SVC, target_col)