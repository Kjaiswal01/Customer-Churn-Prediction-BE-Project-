# Deployment Guide

## Local Run

```bash
pip install -r requirements.txt
streamlit run app.py
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

## Docker Deployment

```bash
docker compose up --build
```

Services:

- Dashboard: `http://localhost:8501`
- FastAPI Docs: `http://localhost:8000/docs`
- MySQL: `localhost:3306`
- Default dashboard login: `admin@retentionos.local` / `admin123`

## Cloud Deployment Suggestions

- Streamlit dashboard: Render, Azure Web App, EC2, or ECS
- FastAPI backend: Render, Railway, Azure App Service, ECS, or Cloud Run
- MySQL: Azure Database for MySQL, Amazon RDS, PlanetScale, or Railway MySQL

## Minimum Production Checklist

- Set real `DATABASE_URL`
- Set SMTP credentials
- Turn off `USE_SQLITE_FALLBACK`
- Use managed MySQL
- Add reverse proxy / HTTPS
- Add secrets through platform environment variables
- Add monitoring and logging
