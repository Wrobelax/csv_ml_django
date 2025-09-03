import pandas as pd
from analyzer.pipelines.pipeline import full_pipeline


def test_pipeline_with_simple_data(tmp_path):
    # Creating CSV
    csv_file = tmp_path / 'test.csv'
    df = pd.DataFrame({
        'feature1': [1,2,3,4,5,6,7,8,9,10],
        'feature2': [10,20,30,40,50,60,70,80,90,100],
        'target': [1,0,1,0,1,0,1,0,1,0]
    })

    df.to_csv(csv_file, index=False)

    result = full_pipeline(str(csv_file))
    assert "analysis" in result
    assert "regression" in result