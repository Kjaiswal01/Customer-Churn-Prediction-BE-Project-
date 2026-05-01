# Quick Start - Run Your Project

## Project Run Link

Once you start the application, access it at:

```text
http://localhost:8501
```

## Easiest Windows Method

Double-click:

```text
RUN_PROJECT.bat
```

This creates `.venv`, installs `requirements.txt`, and starts the Streamlit app.

## Command Prompt Method

```bat
cd customer-churn-prediction
RUN_PROJECT.bat
```

## Before Sharing ZIP

Do not include these folders/files in the ZIP:

- `.venv/`
- `__pycache__/`
- `.git/`
- `.env`
- `*.log`

## Troubleshooting

### If modules are missing

Run:

```bat
RUN_PROJECT.bat
```

The script installs packages automatically. The `sqlalchemy` package is already included in `requirements.txt`.

### If Python is missing

Install Python 3.11 or 3.12 and tick `Add python.exe to PATH`.

### If port 8501 is already in use

Run:

```bat
.venv\Scripts\python.exe -m streamlit run app.py --server.port 8502
```

