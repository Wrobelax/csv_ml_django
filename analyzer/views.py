import pandas as pd
from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from .forms import UploadedDatasetForm
from .models import UploadedDataset
from .pipelines.pipeline import full_pipeline


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

            # Load Pandas CSV + full pipeline
            try:
                result = full_pipeline(dataset.file.path)
                dataset.analysis = result["analysis"]
                dataset.regression = result["regression"]

                dataset.status = 'done'
                dataset.save()

                # Load preview for template\
                df = pd.read_csv(dataset.file.path)
                preview = df.head().values.tolist()
                columns = df.columns.tolist()
                preview_html = df.head().to_html(index=False, classes='table table-bordered table-striped')


            except Exception as e:
                # If analysis did not work - dataset.status set as 'error' in analyze()
                dataset.status = "error"
                dataset.save()
                return JsonResponse({"error": str(e)},status=400)

            return render(request, 'analyzer/detail.html',{
                "dataset": dataset,
                "analysis": dataset.analysis,
                "regression": dataset.regression,
                "preview": preview,
                "preview_html": preview_html,
                "columns": columns,
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
    analysis = dataset.analysis
    regression = dataset.regression


    try:
        df = pd.read_csv(dataset.file.path)
        preview = df.head().values.tolist()
        columns = df.columns.tolist()
        preview_html = df.head().to_html(index=False, classes='table table-bordered table-striped')

    except Exception:
        preview = []
        columns = []
        preview_html = []

    return render(request, 'analyzer/detail.html', {
        "dataset": dataset,
        "analysis": analysis,
        "regression": regression,
        "preview": preview,
        "columns": columns,
        "preview_html": preview_html,
    })
