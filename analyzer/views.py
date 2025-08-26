import pandas as pd
from django.shortcuts import render, redirect, get_object_or_404
from django.urls import reverse
from django.http import JsonResponse
from .forms import UploadedDatasetForm
from .models import UploadedDataset


def upload_dataset(request):
    """
    GET: Shows form.
    POST: Saves file, runs analysis (synchronous), redirect to detail.
    """

    if request.method == 'POST' and request.FILES['file']:

        form = UploadedDatasetForm(request.POST, request.FILES)

        if form.is_valid():
            dataset = form.save(commit=False)
            dataset.original_filename = dataset.file.name
            if request.user.is_authenticated:
                dataset.owner = request.user
            dataset.status = 'processing'
            dataset.save()

            # Load Pandas CSV
            try:
                df = pd.read_csv(dataset.file.path)
                analysis = {
                    "rows": df.shape[0],
                    "columns": df.shape[1],
                    "column_names": list(df.columns),
                    "numeric_means": df.describe().loc["mean"].to_dict()
                }
                dataset.status = "done"
                dataset.save()

            except Exception:
                # If analysis did not work - dataset.status set as 'error' in analyze()
                dataset.status = "error"
                dataset.save()
                return JsonResponse({"error": "Failed to analyze dataset."}, status=400)

            return JsonResponse({
                "id": dataset.pk,
                "name": dataset.original_filename,
                "analysis": analysis
            })


        # If GET -> show form.
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
