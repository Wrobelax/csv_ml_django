import pytest
from django.contrib.auth.models import User
from analyzer.models import UploadedDataset


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