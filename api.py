from __future__ import annotations

from typing import Any

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

from enterprise_auth import authenticate_user, create_access_token, create_user, decode_token, seed_default_user
from enterprise_database import Company, CustomerRecord, User, init_database, get_session
from enterprise_service import (
    churn_by_dimension,
    customer_timeline,
    dashboard_stats,
    execute_assign_action,
    execute_call_action,
    execute_email_action,
    execute_offer_action,
    forecast_churn,
    high_risk_customers,
    log_audit_event,
    nlp_issue_insights,
    predict_records,
    score_dataset_and_store,
    send_retention_emails_for_company,
    simulate_business_impact,
    train_models_from_upload,
    workflow_overview,
)

init_database()
seed_default_user()

app = FastAPI(title="Retention Intelligence API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def audit_request(request: Request, call_next):
    response = await call_next(request)
    session = next(get_session())
    try:
        log_audit_event(
            session,
            event_type="api_call",
            entity_type="route",
            entity_id=request.url.path,
            details={"method": request.method, "status_code": response.status_code},
        )
        session.commit()
    finally:
        session.close()
    return response


class PredictionRequest(BaseModel):
    records: list[dict[str, Any]]


class LoginRequest(BaseModel):
    email: str
    password: str


class RegisterRequest(BaseModel):
    full_name: str
    email: str
    password: str
    role: str = "manager"


class ActionRequest(BaseModel):
    customer_record_id: int
    assigned_agent: str | None = None


class WhatIfRequest(BaseModel):
    record: dict[str, Any]
    reduce_price_pct: float = 0.0
    improve_support: bool = False


def get_current_user(authorization: str | None = Header(default=None), session: Session = Depends(get_session)) -> User:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization.split(" ", 1)[1]
    try:
        payload = decode_token(token)
    except Exception as exc:
        raise HTTPException(status_code=401, detail=str(exc))
    user = session.query(User).filter(User.id == int(payload["sub"]), User.is_active.is_(True)).one_or_none()
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/auth/register")
def register(request: RegisterRequest):
    user = create_user(request.full_name, request.email, request.password, request.role)
    token = create_access_token(user)
    return {"access_token": token, "user": {"id": user.id, "email": user.email, "role": user.role}}


@app.post("/auth/login")
def login(request: LoginRequest):
    user = authenticate_user(request.email, request.password)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token(user)
    return {"access_token": token, "user": {"id": user.id, "email": user.email, "role": user.role}}


@app.post("/train")
async def train_api(
    company_name: str = Form(...),
    industry: str = Form("Telecom"),
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    try:
        result = train_models_from_upload(session, company_name, industry, file)
        return result
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/datasets/{company_id}/score")
async def score_dataset_api(
    company_id: int,
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    try:
        return score_dataset_and_store(session, company_id, file)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/predict")
def predict_api(request: PredictionRequest, session: Session = Depends(get_session), current_user: User = Depends(get_current_user)):
    try:
        return {"results": predict_records(session, request.records)}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/segmentation/{company_id}")
def segmentation_api(company_id: int, session: Session = Depends(get_session), current_user: User = Depends(get_current_user)):
    rows = session.query(CustomerRecord).filter(CustomerRecord.company_id == company_id).all()
    return {
        "results": [
            {
                "customer_id": row.external_customer_id,
                "segment": row.churn_segment,
                "churn_probability": row.churn_probability,
                "customer_value": row.customer_value,
            }
            for row in rows
        ]
    }


@app.get("/recommendations/{company_id}")
def recommendations_api(company_id: int, session: Session = Depends(get_session), current_user: User = Depends(get_current_user)):
    rows = session.query(CustomerRecord).filter(CustomerRecord.company_id == company_id).all()
    return {
        "results": [
            {
                "customer_id": row.external_customer_id,
                "offer": row.recommended_offer,
                "action": row.recommended_action,
                "explanation": row.explanation,
            }
            for row in rows
        ]
    }


@app.post("/emails/{company_id}/send")
def send_emails_api(company_id: int, session: Session = Depends(get_session), current_user: User = Depends(get_current_user)):
    try:
        return send_retention_emails_for_company(session, company_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/companies")
def companies_api(session: Session = Depends(get_session), current_user: User = Depends(get_current_user)):
    rows = session.query(Company).all()
    return {"results": [{"id": row.id, "name": row.name, "industry": row.industry} for row in rows]}


@app.get("/dashboard/stats")
def dashboard_stats_api(company_id: int | None = None, session: Session = Depends(get_session), current_user: User = Depends(get_current_user)):
    return dashboard_stats(session, company_id)


@app.get("/analytics/churn-by-plan")
def churn_by_plan_api(company_id: int | None = None, session: Session = Depends(get_session), current_user: User = Depends(get_current_user)):
    return {"results": churn_by_dimension(session, "Subscription_Type", company_id)}


@app.get("/analytics/churn-by-gender")
def churn_by_gender_api(company_id: int | None = None, session: Session = Depends(get_session), current_user: User = Depends(get_current_user)):
    return {"results": churn_by_dimension(session, "Gender", company_id)}


@app.get("/action-center")
def action_center_api(company_id: int | None = None, session: Session = Depends(get_session), current_user: User = Depends(get_current_user)):
    return {"results": high_risk_customers(session, company_id)}


@app.post("/action/call")
def action_call_api(request: ActionRequest, session: Session = Depends(get_session), current_user: User = Depends(get_current_user)):
    return execute_call_action(session, request.customer_record_id, current_user.email)


@app.post("/action/email")
def action_email_api(request: ActionRequest, session: Session = Depends(get_session), current_user: User = Depends(get_current_user)):
    return execute_email_action(session, request.customer_record_id, current_user.email)


@app.post("/action/offer")
def action_offer_api(request: ActionRequest, session: Session = Depends(get_session), current_user: User = Depends(get_current_user)):
    return execute_offer_action(session, request.customer_record_id, current_user.email)


@app.post("/action/assign")
def action_assign_api(request: ActionRequest, session: Session = Depends(get_session), current_user: User = Depends(get_current_user)):
    if not request.assigned_agent:
        raise HTTPException(status_code=400, detail="assigned_agent is required")
    return execute_assign_action(session, request.customer_record_id, request.assigned_agent, current_user.email)


@app.post("/what-if")
def what_if_api(request: WhatIfRequest, current_user: User = Depends(get_current_user)):
    return simulate_business_impact(request.record, request.reduce_price_pct, request.improve_support)


@app.get("/nlp/insights")
def nlp_insights_api(company_id: int | None = None, session: Session = Depends(get_session), current_user: User = Depends(get_current_user)):
    return nlp_issue_insights(session, company_id)


@app.get("/timeline/{customer_record_id}")
def timeline_api(customer_record_id: int, session: Session = Depends(get_session), current_user: User = Depends(get_current_user)):
    return {"results": customer_timeline(session, customer_record_id)}


@app.get("/forecast/churn")
def forecast_api(company_id: int | None = None, session: Session = Depends(get_session), current_user: User = Depends(get_current_user)):
    return forecast_churn(session, company_id)


@app.get("/workflow")
def workflow_api(company_id: int | None = None, session: Session = Depends(get_session), current_user: User = Depends(get_current_user)):
    return workflow_overview(session, company_id)
