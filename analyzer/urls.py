from django.urls import path
from .views import upload_dataset, dataset_detail

urlpatterns = [
    path('upload/', upload_dataset, name='upload_dataset'),
    path('<int:pk>/', dataset_detail, name='dataset_detail'),
]