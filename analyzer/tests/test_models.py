import pytest
import pandas as pd
from django.contrib.auth.models import User
from analyzer.models import UploadedDataset
from analyzer.pipelines.ml_classification import (
    knn_classification,
    decision_tree_classification,
    logistic_regression_classification,
    svm_classification
)
from analyzer.pipelines.regression import run_regression


@pytest.mark.django_db
def test_create_dataset_with_owner():
    user = User.objects.create_user(username='testuser', password='pass123')
    dataset = UploadedDataset.objects.create(
        name="test.csv",
        file="test.csv",
        owner=user,
    )

    assert dataset.owner == user
    assert dataset.name == "test.csv"


def test_knn_classification_returns_metrics():
    df = pd.DataFrame({
        'feature1': range(1,11),
        'feature2': [i * 10 for i in range(1,11)],
        'target': [0,1,0,1,0,1,0,1,0,1]
    })

    result = knn_classification(df, 'target')

    assert isinstance(result, dict)
    assert "accuracy" in result
    assert "confusion_matrix" in result
    assert "classes" in result
    assert 0.0 <= result["accuracy"] <= 1.0


def test_decision_tree_classification_returns_metrics():
    df = pd.DataFrame({
        'x1': [0,1,0,1,0,1,0,1,0,1],
        'x2': [i for i in range(1,11)],
        'target': [0,1,0,1,0,1,0,1,0,1]
    })

    result = decision_tree_classification(df, 'target')

    assert "accuracy" in result
    assert 0.0 <= result["accuracy"] <= 1.0


def test_logistic_regression_runs():
    df =pd.DataFrame({
        'a': range(20),
        'b': [i % 2 for i in range(20)],
        'target': [0] * 10 + [1] * 10
    })

    result = logistic_regression_classification(df, 'target')

    assert "accuracy" in result
    assert 0.0 <= result["accuracy"] <= 1.0


def test_svm_classification_returns_metrics():
    df = pd.DataFrame({
        'feature1': range(1,11),
        'feature2': [i * 10 for i in range(1,11)],
        'target': [0,1,0,1,0,1,0,1,0,1]
    })

    result = svm_classification(df, 'target')

    assert "accuracy" in result
    assert "confusion_matrix" in result
    assert "classes" in result
    assert 0.0 <= result["accuracy"] <= 1.0


def test_linear_regression_metrics():
    df = pd.DataFrame({
        'x': range(1,21),
        'y': [i * 4.20 for i in range(1,21)]
    })

    result = run_regression(df)

    assert "r2_score" in result
    assert "coefficients" in result
    assert "intercept" in result
    assert result["r2_score"] > 0.9