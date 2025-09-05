# *CSV ML Django*
___
![Status](https://img.shields.io/badge/project_status-complete\closed-darkgreen)

![Status](https://img.shields.io/badge/testing-done-darkgreen)

A Django web application for analyzing CSV datasets using machine learning models. Users can upload datasets, run ML analyses, and view results on a dashboard.

___

## *Table of Contents*
- [Project Summary](#project-summary)
- [Features](#features)
- [Technologies](#technologies)
- [Installation](#installation)
- [Running the Project]()
- [Running Tests](#running-tests)
- [Project Structure](#project-structure)
- [Example Usage](#example-usage)
- [Conclusion](#conclusion)

___

## Project Summary
This Django project provides an interface for uploading CSV datasets and running machine learning analyses. Key points:
- Combines **Django web framework** with **ML pipelines** using **scikit-learn.**
- Fully **user-specific:** each user has their own datasets and results.
- Provides **visualization of results** through charts and tables.
- Includes **tests (Pytest)** for models, views, and ML pipelines.

### Screenshots
**1. Login/Registration page**
**2. Dataset upload page**
![upload.png](ml_data/screens/upload.png)
**3. Dataset detail page** showing available analyses
![datasets.png](ml_data/screens/datasets.png)
**4. ML result page** with metrics and plots
![ml_results1.png](ml_data/screens/ml_results1.png)
![ml_results2.png](ml_data/screens/ml_results2.png)
![ml_results3.png](ml_data/screens/ml_results3.png)
**5. Zoomed chart**(auto-zoom)
![ml_results3.png](ml_data/screens/ml_results_zoomed.png)
___

## Features
- Upload CSV datasets through a web interface.
- Automatic dataset creation associated with the logged-in-user.
- Run multiple ML algorithms: KNN, Decision Tree, Logistic Regression, SVM, Linear Regression.
- View results, metrics, and plots for each analysis.
- Dashboard showing only the dataset of the logged-in user.

___

## *Technologies*
- Python 3.13
- Django 5.2
- Pandas, NumPy
- Scikit-learn
- Pytest
- SQLite

___

## *Installation*

### 1.Clone the repository:
```bash
git clone https://github.com/Wrobelax/csv_ml_django
cd csv_ml_django
```

### 2.Create and activate a virtual environment:
```bash
python -m venv.venv
.venv\Script\activate # Windows
source .venv/bin/activate # Linux / macOS
```

### 3. Install dependencies:
```bash
pip install -r requirements.txt
```

## *Running the Project*

### 1. Apply database migrations:
```bash
python manage.py migrate
```

### 2. Create a superuser (optional):
```bash
python manage.py createsuperuser
```

### 3. Run the server:
```bash
python manage.py runserver
```

### 4. Open the browser and access the project via http://127.0.0.1:8000/datasets/upload

___

## *Running Tests*
The project includes unit tests for models, views, and ML pipelines.
Run all tests with:
```bash
pytest -v
```
Tests include:
- Dataset creation and validation
- Running ML models and checking results
- Access control for views (login required)

___

## *Project Structure*
```
csv_ml_django/
|
├── accounts/
├── analyzer/
|   ├── migrations/
|   ├── models.py                  # Django models
|   ├── views.py                   # Views
|   ├── urls.py                    # URL patterns
|   ├── pipelines/                 # Machine learning functions
|   |   ├── analysis.py
|   |   ├── ml_classification.py
|   |   ├── pipelines.p
|   |   └── regression.py
|   ├── templates/
|   ├── tests/                     # Unit tests
|   |   ├── test_models.py
|   |   ├── test_pipeline.py
|   └── └── test_views.py
├── backend/                      # Django project settings
|   ├── settings.py
|   └── urls.py
├── ml_data/                      # Example datasets for machine learning
|   ├── iris_classification.csv   # Sample data with iris classification
|   └── diabetes.csv              # Sample data with diabetes classification
├── manage.py
├── requirements.txt
└── README.md
```
___

## *Example Usage*

### 1. Register and login
Navigate to the registration page, create a user and login into dashboard.

### 2. Upload CSV
Go to the upload page (/datasets/upload), select you CSV file (you can use one of example data from ml_data folder), and submit.

### 3. Run analysis
From the dataset detail page choose a ML algorithm:
- KNN
- Decision Tree
- SVM
- Logistic Regression
- Linear Regression

### 4. View results on the dashboard
The dashboard (/dashboard/) shows:
- Uploaded datasets
- ML results
- Confusion matrices, accuracy, and regression metrics

___

## *Conclusion*
This project demonstrates a web-based platform that allows users to upload datasets, run machine learning algorithms, and visualize results. It combines Django web development, data processing, and ML integration in a single application.

Key takeaways:
- Seamless **dataset management** and user authentication.
- Implementation of **KNN, Decision Tree, Logistic Regression, SVM, and Linear Regression** pipelines.
- Clear **data visualization** and performance metrics.
- Application of **testing** for reliability and maintainability.