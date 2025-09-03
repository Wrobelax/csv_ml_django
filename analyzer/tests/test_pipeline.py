import pandas as pd
import pytest
from analyzer.pipelines.pipeline import full_pipeline


def test_pipeline_with_simple_data(tmp_path):
    # Creating CSV
    csv_file = tmp_path / 'test.csv'
    df = pd.DataFrame({
        'feature1': range(1,11),
        'feature2': [i * 10 for i in range(1,11)],
        'target': [i % 2 for i in range(1,11)]
    })
    df.to_csv(csv_file, index=False)

    result = full_pipeline(str(csv_file))
    assert "analysis" in result
    assert "regression" in result


@pytest.mark.django_db
def test_pipeline_regression(tmp_path):
    csv_file = tmp_path / 'regression.csv'
    df = pd.DataFrame({'x': range(1,51), 'y': [i * 2.5 for i in range(1,51)]})
    df.to_csv(csv_file, index=False)

    result = full_pipeline(str(csv_file))

    assert "regression" in result
    assert "regression" is not None
    assert "r2_score" in result["regression"]


@pytest.mark.django_db
def test_pipeline_classification(tmp_path):
    csv_file = tmp_path / 'classification.csv'
    df = pd.DataFrame({'feature1': range(1,11), 'target': [i % 2 for i in range(1,11)]})
    df.to_csv(csv_file, index=False)

    result = full_pipeline(str(csv_file))

    assert "ml_results" in result
    assert "knn_classification" in result["ml_results"]
    assert "accuracy" in result["ml_results"]["knn_classification"]


def test_pipeline_without_target(tmp_path):
    csv_file = tmp_path / "no_target.csv"
    df = pd.DataFrame({'a': [1,2,3], 'b': [4,5,6]})
    df.to_csv(csv_file, index=False)

    with pytest.raises(Exception):
        full_pipeline(str(csv_file))


def test_pipeline_empty_csv(tmp_path):
    csv_file = tmp_path / 'empty.csv'
    pd.DataFrame().to_csv(csv_file, index=False)

    with pytest.raises(ValueError):
        full_pipeline(str(csv_file))


def test_pipeline_wrong_format(tmp_path):
    txt_file = tmp_path / 'wrong_format.txt'
    txt_file.write_text('not,a,csv,file,xd')

    with pytest.raises(Exception):
        full_pipeline(str(txt_file))


def test_pipeline_knn_too_few_samples(tmp_path):
    csv_file = tmp_path / 'knn_too_few_samples.csv'
    df = pd.DataFrame({'feature1': [1,2], 'target': [1,0]})
    df.to_csv(csv_file, index=False)

    with pytest.raises(ValueError):
        full_pipeline(str(csv_file))


def test_pipeline_full_models(tmp_path):
    csv_file = tmp_path / 'full_models.csv'
    df = pd.DataFrame({
        'feature1': range(1,51),
        'feature2': [i * 10 for i in range(1,51)],
        'target': [i % 2 for i in range(1,51)],
    })
    df.to_csv(csv_file, index=False)

    result = full_pipeline(str(csv_file))

    assert "ml_results" in result
    for model in [
        "knn_classification",
        "decision_tree_classification",
        "random_forest_classification",
        "logistic_regression_classification",
    ]:
        assert model in result["ml_results"]