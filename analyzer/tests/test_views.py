import pytest
from django.urls import reverse
from django.core.files.uploadedfile import SimpleUploadedFile
from django.contrib.auth.models import User
from analyzer.models import UploadedDataset


@pytest.mark.django_db
def test_upload_redirects_if_not_logged_in(client):
    url = reverse("upload_dataset")
    response = client.get(url)

    assert response.status_code == 302
    assert response.url.startswith("/accounts/login/")


@pytest.mark.django_db
def test_upload_creates_dataset(client):
    user = User.objects.create_user(username='testuser', password='pass123')
    client.login(username='testuser', password='pass123')

    url = reverse('upload_dataset')
    file = SimpleUploadedFile('test.csv',
                              b'feature,target\n1,0\n2,1\n3,0\n4,1\n5,0\n6,1\n7,0\n8,1\n9,0\n10,1',
                              content_type='text/csv'
    )

    response = client.post(url, {'name': 'data.csv', 'file': file}, follow=True)

    print(response.content)
    assert response.status_code == 200
    assert UploadedDataset.objects.count() == 1
    dataset = UploadedDataset.objects.first()
    assert dataset.owner == user


@pytest.mark.django_db
def test_analysis_results_requires_login(client):
    url = reverse('dataset_detail', args=[1])
    response = client.get(url)

    assert response.status_code == 302
    assert "/login" in response.url


@pytest.mark.django_db
def test_analysis_results_not_found(client):
    user = User.objects.create_user(username='testuser', password='pass123')
    client.login(username='testuser', password='pass123')

    url = reverse('dataset_detail', args=[999])
    response = client.get(url)

    assert response.status_code == 404


# === Dashboard tests ===
@pytest.mark.django_db
def test_dashboard_requires_login(client):
    url = reverse('my_datasets')
    response = client.get(url)

    assert response.status_code == 302
    assert "/login" in response.url


@pytest.mark.django_db
def test_dashboard_empty_for_new_user(client):
    user = User.objects.create_user(username='testuser', password='pass123')
    client.login(username='testuser', password='pass123')

    url = reverse('my_datasets')
    response = client.get(url)

    assert response.status_code == 200
    content = response.content.decode()
    assert "No dataset" in content


@pytest.mark.django_db
def test_dashboard_shows_only_user_datasets(client):
    user1 = User.objects.create_user(username='user1', password='pass123')
    user2 = User.objects.create_user(username='user2', password='pass456')

    UploadedDataset.objects.create(name='user1_file', file='file1.csv', owner=user1)
    UploadedDataset.objects.create(name='user2_file', file='file2.csv', owner=user2)

    client.login(username='user1',password='pass123')
    url = reverse('my_datasets')
    response = client.get(url)

    assert response.status_code == 200
    content = response.content.decode()

    assert "user1" in content
    assert "user2" not in content