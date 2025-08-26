from django.contrib import admin
from .models import UploadedDataset

@admin.register(UploadedDataset)
class UploadedDatasetAdmin(admin.ModelAdmin):
    list_display = ('original_filename', 'uploaded_at', 'n_rows', 'n_cols', 'status')
    readonly_fields = ('uploaded_at',)
