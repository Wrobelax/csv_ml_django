import pandas as pd
from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from .forms import UploadedDatasetForm
from .models import UploadedDataset
from .pipelines.pipeline import full_pipeline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import io, base64
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('ascii')


def plot_prediction_distribution(y_pred):
    if y_pred is None:
        return None

    arr = np.array(y_pred, dtype=object)
    classes, counts = np.unique(arr, return_counts=True)
    fig, ax = plt.subplots(figsize=(5,3))
    ax.bar(classes.astype(str), counts)
    ax.set_title("Prediction distribution")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Count")
    return fig_to_base64(fig)


def plot_roc_curve(y_test, y_proba, classes):
    try:
        y_true = np.array(y_test)
        y_score = np.array(y_proba)

        # If y_score is 1D (decision_function single score) convert to 2D
        if y_score.ndim == 1:
            y_score = np.vstack([1- y_score, y_score]).T

        # Binarize y_true to shape(n_samples, n_classes)
        classes_unique = list(classes)
        y_true_bin = label_binarize(y_true, classes=classes_unique)
        n_classes = y_true_bin.shape[1]

        fig, ax = plt.subplots(figsize=(5,4))
        for i in range(n_classes):
            try:
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, lw=2, label=f"{classes_unique[i]} (AUC={roc_auc:.2f})")
            except Exception:
                continue

        ax.plot([0, 1], [0, 1], 'k--', lw=1)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title("ROC curve (one-vs-rest)")
        ax.legend(loc="lower right", fontsize="small")
        return fig_to_base64(fig)

    except Exception:
        return None


def plot_confusion_matrix(cm, classes):
    fig, ax = plt.subplots(figsize=(4,3))
    cm_arr = np.array(cm)
    im = ax.imshow(cm_arr, interpolation='nearest', aspect='auto')
    ax.set_title("Confusion matrix")
    ax.set_xticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_yticks(np.arange(len(classes)))
    ax.set_yticklabels(classes)

    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            val = int(cm_arr[i, j])
            ax.text(
                j, i, str(val),
                ha='center',
                va='center',
                color='white' if cm_arr[i, j] > cm_arr.max()/2 else 'black'
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    return fig_to_base64(fig)


def plot_feature_importances(feature_importances: dict):
    if not feature_importances:
        return None

    items = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
    names = [i[0] for i in items]
    vals = [i[1] for i in items]
    fig, ax = plt.subplots(figsize=(6, max(2, len(names)*0.4)))
    y_pos = range(len(names))
    ax.barh(y_pos, vals)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title("Feature importances")

    return fig_to_base64(fig)


def plot_regression_pred_actual(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    fig, ax = plt.subplots(figsize=(5,4))
    ax.scatter(y_true, y_pred, alpha=0.7)
    mn = min(min(y_true), min(y_pred))
    mx= max(max(y_true), max(y_pred))
    ax.plot([mn,mx], [mn,mx], "--", linewidth=1)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs. Predicted")

    return fig_to_base64(fig)


def upload_dataset(request):
    """
    GET: Shows form.
    POST: Saves file, runs analysis (synchronous), redirect to detail.
    """

    reg_plot = None

    if request.method == 'POST' and request.FILES.get('file'):

        form = UploadedDatasetForm(request.POST, request.FILES)
        if not form.is_valid():
            return render(request, 'analyzer/upload.html', {'form': form})

        dataset = form.save(commit=False)
        dataset.original_filename = dataset.file.name
        if request.user.is_authenticated:
            dataset.owner = request.user
        dataset.status = 'processing'
        dataset.save()

        # Run pipeline (returns Python-serializable dicts/lists)
        try:
            result = full_pipeline(dataset.file.path)
            dataset.analysis = result.get("analysis")
            dataset.regression = result.get("regression")
            ml_results = result.get("ml_results")

            dataset.status = 'done'
            dataset.save()

            # Load preview for template
            df = pd.read_csv(dataset.file.path)
            preview = df.head().values.tolist()
            columns = df.columns.tolist()
            preview_html = df.head().to_html(index=False, classes='table table-bordered table-striped')

            # Generating charts for ML
            if ml_results:
                for model_key, model_data in ml_results.items():
                    if not model_data:
                        continue

                    plots = {}


                    # Confusion matrix plot
                    cm = model_data.get("confusion_matrix")
                    classes = model_data.get("classes")
                    fi = model_data.get("feature_importances")

                    if cm and classes:
                        plots["confusion_matrix"] = plot_confusion_matrix(cm, classes)
                    else:
                        plots["confusion_matrix"] = None

                    # Feature importances plot
                    plots["feature_importances"] = plot_feature_importances(fi) if fi else None

                    model_data["plots"] = plots


                    # Prediction distribution
                    y_test = model_data.get("y_test")
                    y_pred = model_data.get("y_pred")
                    y_proba = model_data.get("y_proba")
                    plots["prediction_dist"] = plot_prediction_distribution(y_pred) if y_pred is not None else None

                    # Roc curve if probabilities available
                    if y_proba is not None and classes:
                        plots["roc_curve"] = plot_roc_curve(y_test, y_proba, classes)
                    else:
                        plots["roc_curve"] = None

                    model_data["plots"] = plots


                # Regression plot for actual vs. predicted
                reg = dataset.regression

                if isinstance(reg, dict):
                    y_test = reg.get("y_test")
                    y_pred = reg.get("y_pred")

                    if y_test and y_pred and len(y_test) > 0 and len(y_pred) > 0:
                        reg_plot = plot_regression_pred_actual(y_test, y_pred)
                        dataset.regression_plot = reg_plot
                dataset.status = 'done'
                dataset.save()


        except Exception as e:
            dataset.status = 'error'
            dataset.save()
            return JsonResponse({"error": str(e)},status=400)



        return render(request, 'analyzer/detail.html',{
            "dataset": dataset,
            "analysis": dataset.analysis,
            "regression": dataset.regression,
            "ml_results": ml_results,
            "preview": preview,
            "preview_html": preview_html,
            "columns": columns,
            "regression_plot": reg_plot,
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
    ml_results = getattr(dataset, 'ml_results', None)
    reg_plot = dataset.regression_plot


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
        "ml_results": ml_results,
        "preview": preview,
        "columns": columns,
        "preview_html": preview_html,
        "regression_plot": reg_plot,
    })
