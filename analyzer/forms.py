from django import forms
from .models import UploadedDataset

class UploadedDatasetForm(forms.ModelForm):
    class Meta:
        model = UploadedDataset
        fields = ['file']


        def clean_file(self):
            f =self.cleaned_data['file']

            # Simple validator for file type and file size.
            if not f.name.lower().endswith('.csv'):
                raise forms.ValidationError('File must end with .csv')

            max_size = 10 * 1024 * 1024
            if f.size > max_size:
                raise forms.ValidationError('File too big, max 10 mb.')
            return f
