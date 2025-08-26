import pandas as pd
from django.shortcuts import render, redirect, get_object_or_404
from django.urls import reverse
from .forms import UploadedDatasetForm
from .models import UploadedDataset


def upload_dataset(request):
    """
    GET: Shows form.
    POST: Saves file, runs analysis (synchronous), redirect to detail.
    """

    if request.method == 'POST':
        form = UploadedDatasetForm(request.POST, request.FILES)
        if form.is_valid():
            dataset = form.save(commit=False)
            dataset.original_filename = dataset.file.name
            if request.user.is_authenticated:
                dataset.owner = request.user
            dataset.status = 'processing'
            dataset.save()

            # Synchronous analysis for small files (up to 10mb).
            try:
                dataset.analyze()

            except Exception:
                # If analysis did not work - dataset.status set as 'error' in analyze()
                pass

            return redirect('detail', pk=dataset.pk)

        else:
            form = UploadedDatasetForm()
        return render(request, 'analyzer/upload.html', {'form': form})


def dataset_detail(request, pk):
    """
    Details view of upload: metadata, first 5 rows and summary.
    """

    dataset = get_object_or_404(UploadedDataset, pk=pk)
    preview_html = ""

    try:
        df = pd.read_csv(dataset.file.path)
        preview_html = df.head().to_html(index=False, classes='table table-sm')

    except Exception:
        preview_html = "<p>Can't load file view.</p>"


    return render(request, 'analyzer/detail.html', {
        'dataset': dataset,
        'preview_html': preview_html,
    })
