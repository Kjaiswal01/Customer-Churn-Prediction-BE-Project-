# Retention Intelligence Platform Blueprint

## Product Direction

This project is no longer just a churn prediction demo. It is now positioned as a **Retention Intelligence Platform** for telecom, subscription, SaaS, banking, and similar customer-heavy businesses.

## What Makes It Unique

- Multi-company workspaces
- Dataset upload plus automatic schema normalization
- Stored customer intelligence in a database
- Per-customer churn reasons
- Auto-generated retention offer and next-best-action
- Retention action desk and campaign queue
- Prediction history for future audit and model monitoring

## Core Data Stored

- `companies`
- `dataset_uploads`
- `customers`
- `prediction_history`
- `retention_actions`

## Current Stage 2 Upgrade Included

- Database-backed Flask platform with SQLite
- Company dataset upload API for CSV and Excel
- Automatic upload storage
- Stored customer profiles and analysis history
- Company analytics endpoints
- Retention queue generation from customer risk
- New dashboard, analytics, customer studio, retention desk, and campaign studio

## Next Stage 3 Additions Recommended

- Authentication and role-based access
- PostgreSQL migration for production
- Background jobs for long dataset processing
- WhatsApp, email, and SMS integrations
- Admin approval workflow for offers
- Model explainability with SHAP
- Drift detection and retraining pipeline
- CRM integrations
- Exportable PDF and Excel reports
- Docker and CI/CD deployment

## Recommended Production Stack

- Frontend: Flask templates now, React later if needed
- Backend: Flask service layer
- Database: PostgreSQL
- Queue: Redis plus Celery
- Storage: S3 or cloud blob storage
- Deployment: Docker plus Render/Azure/AWS
- Monitoring: Sentry plus structured logs

## Pitch Line

“An AI-powered retention intelligence platform that not only predicts churn, but also explains why a customer is at risk, stores full history, and recommends the best intervention before the customer leaves.”
