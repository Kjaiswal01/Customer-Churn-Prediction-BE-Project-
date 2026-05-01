from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


REQUIRED_MODULES = (
    "flask",
    "flask_cors",
    "imblearn",
    "joblib",
    "matplotlib",
    "numpy",
    "openpyxl",
    "pandas",
    "plotly",
    "pymysql",
    "seaborn",
    "shap",
    "sklearn",
    "sqlalchemy",
    "streamlit",
    "textblob",
    "uvicorn",
    "werkzeug",
    "xgboost",
)


def _missing_modules() -> list[str]:
    return [module for module in REQUIRED_MODULES if importlib.util.find_spec(module) is None]


def ensure_dependencies() -> None:
    missing = _missing_modules()
    if not missing:
        return

    requirements_path = Path(__file__).resolve().parent / "requirements.txt"
    if not requirements_path.exists():
        missing_text = ", ".join(missing)
        raise RuntimeError(
            f"Missing Python packages: {missing_text}. requirements.txt was not found."
        )

    print("Missing Python packages detected:", ", ".join(missing))
    print("Installing project dependencies from requirements.txt...")

    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(requirements_path)],
        check=False,
    )
    if result.returncode != 0:
        missing_text = ", ".join(missing)
        raise RuntimeError(
            "Dependency installation failed. Run this command in the project folder: "
            f'"{sys.executable}" -m pip install -r requirements.txt. '
            f"Still missing: {missing_text}"
        )

    still_missing = _missing_modules()
    if still_missing:
        raise RuntimeError(
            "Some dependencies are still missing after installation: "
            + ", ".join(still_missing)
        )

