from django.db import models
from django.contrib.auth import get_user_model
import pandas as pd



class UploadedDataset(models.Model):
    uploaded_at = models.DateTimeField(auto_now_add=True)
    owner = models.ForeignKey(get_user_model(), null=True, blank=True, on_delete=models.SET_NULL)
    original_filename = models.CharField(max_length=255)
    file = models.FileField(upload_to='datasets/')
    n_rows = models.IntegerField(null=True, blank=True)
    n_cols = models.IntegerField(null=True, blank=True)
    summary = models.JSONField(null=True, blank=True)
    status = models.CharField(max_length=32, default='uploaded')
    error_message = models.TextField(null=True, blank=True)

    def __str__(self):
        return f"{self.original_filename} {self.uploaded_at:%Y-%m-%d %H:%M}"


    def analyze(self):
        """
        Reads files via pandas, saves n_rows, n_cols and summary (describe).
        Method saves model and sets status.
        """

        try:
            df = pd.read_csv(self.file.path)
            self.n_rows, self.n_cols = df.shape
            self.summary = df.describe(include='all').to_dict()
            self.status = 'done'
            self.error_message = None
            self.save()

        except Exception as e:
            self.status = 'error'
            self.error_message = str(e)
            self.save()
            raise


