from django.test import TestCase
from django.core.files.uploadedfile import SimpleUploadedFile
from django.urls import reverse
from .models import UploadedDataset

class UploadTest(TestCase):
    def test_upload_csv_creates_dataset_and_analyze(self):
        csv_content = b"col1,col2\n1,2\n3,4\n"
        f = SimpleUploadedFile('test.csv', csv_content, content_type='text/csv')
        response = self.client.post(reverse('upload_dataset'), {'file': f})
        self.assertEqual(response.status_code, 302)
        ds = UploadedDataset.objects.first()
        self.assertIsNotNone(ds)
        self.assertEqual(ds.n_rows, 2)
        self.assertEqual(ds.n_cols, 2)
        self.assertEqual(ds.status, 'done')