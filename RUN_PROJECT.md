# Project Run Link - Easy Method

Project URL:

```text
http://localhost:8501
```

## How to Run

### Method 1: Batch File

Double-click:

```text
RUN_PROJECT.bat
```

It will create `.venv`, install packages from `requirements.txt`, and start the app.

If someone directly runs `streamlit run app.py`, the app also tries to install missing dependencies from `requirements.txt`, but `RUN_PROJECT.bat` is still the cleanest method.

### Method 2: Command Prompt

```bat
cd customer-churn-prediction
RUN_PROJECT.bat
```

Open the browser at:

```text
http://localhost:8501
```

## Features

- Dashboard - overview and metrics
- Predict Churn - predict customer churn
- Analytics - customer analytics
- Customer Insights - customer segmentation
- Retention Actions - retention strategies
- Bulk Analysis - upload large datasets

## If You Face Issues

### Module not found

Run:

```bat
RUN_PROJECT.bat
```

This project needs dependencies such as `sqlalchemy`, `streamlit`, `pandas`, and `scikit-learn`. They are listed in `requirements.txt`.

### Model files missing

```bat
python model_training.py
```

### Port already in use

Use the new URL shown in the terminal, or run:

```bat
.venv\Scripts\python.exe -m streamlit run app.py --server.port 8502
```
