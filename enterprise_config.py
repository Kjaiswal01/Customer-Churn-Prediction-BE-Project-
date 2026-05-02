from __future__ import annotations

import os
import secrets
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
MODEL_VERSION_DIR = MODEL_DIR / "versions"
INSTANCE_DIR = BASE_DIR / "instance"
UPLOAD_DIR = DATA_DIR / "uploads"

INSTANCE_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MODEL_VERSION_DIR.mkdir(parents=True, exist_ok=True)

APP_ENV = os.getenv("APP_ENV", "development")
IS_PRODUCTION = APP_ENV.lower() == "production"
ENABLE_DEMO_MODE = os.getenv("ENABLE_DEMO_MODE", "true" if not IS_PRODUCTION else "false").lower() == "true"
BOOTSTRAP_DEMO_ON_STARTUP = os.getenv("BOOTSTRAP_DEMO_ON_STARTUP", "false").lower() == "true"
SEED_DEFAULT_USERS = os.getenv("SEED_DEFAULT_USERS", "true" if not IS_PRODUCTION else "false").lower() == "true"
MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_DB = os.getenv("MYSQL_DB", "retention_platform")
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}",
)
DATABASE_FALLBACK_URL = os.getenv(
    "DATABASE_FALLBACK_URL",
    f"sqlite:///{(INSTANCE_DIR / 'retention_platform.db').as_posix()}",
)
USE_SQLITE_FALLBACK = os.getenv("USE_SQLITE_FALLBACK", "false" if IS_PRODUCTION else "true").lower() == "true"

SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USERNAME = os.getenv("SMTP_USERNAME", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
SMTP_FROM_EMAIL = os.getenv("SMTP_FROM_EMAIL", "noreply@retentionos.local")
SMTP_USE_TLS = os.getenv("SMTP_USE_TLS", "true").lower() == "true"

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", "8501"))

ARTIFACT_PATH = MODEL_DIR / "enterprise_artifacts.joblib"
JWT_SECRET = os.getenv("JWT_SECRET", "")
if not JWT_SECRET:
    if IS_PRODUCTION:
        raise RuntimeError("JWT_SECRET environment variable is required in production.")
    JWT_SECRET = secrets.token_urlsafe(32)
