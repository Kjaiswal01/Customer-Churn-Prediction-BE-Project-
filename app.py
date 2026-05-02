from __future__ import annotations

from dependency_bootstrap import ensure_dependencies

ensure_dependencies()

import io
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from data_pipeline import classify_issue, ensure_dataset_exists, simple_sentiment_score
from enterprise_auth import authenticate_user, create_access_token, create_user, decode_token, seed_default_user
from enterprise_config import BOOTSTRAP_DEMO_ON_STARTUP, ENABLE_DEMO_MODE, IS_PRODUCTION, SEED_DEFAULT_USERS, USE_SQLITE_FALLBACK
from enterprise_database import Company, CustomerRecord, DatasetUpload, RetentionAction, SessionLocal, init_database
from enterprise_ml import (
    customer_value_score,
    heuristic_churn_probability,
    load_artifacts,
    load_dataset_from_upload,
    personalized_offer,
    retention_strategy,
    score_customers,
)
from enterprise_service import (
    bootstrap_demo_environment,
    extract_keywords,
    customer_timeline,
    execute_assign_action,
    execute_call_action,
    execute_email_action,
    execute_offer_action,
    forecast_churn,
    high_risk_customers,
    nlp_issue_insights,
    predict_records,
    predictor_assistant_response,
    prepare_company_dataset,
    score_dataframe_and_store,
    score_dataset_and_store,
    send_retention_emails_for_company,
    simulate_business_impact,
    train_models_from_dataframe,
    workflow_overview,
    infer_missing_business_signals,
)

MAX_DASHBOARD_ROWS = 750
logger = logging.getLogger(__name__)

ensure_dataset_exists()
init_database()
seed_default_user()

st.set_page_config(
    page_title="Retention Intelligence Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.2rem; padding-bottom: 2.4rem; max-width: 1400px;}
    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(circle at top left, rgba(9,63,180,0.08), transparent 25%),
            radial-gradient(circle at top right, rgba(13,148,136,0.10), transparent 30%),
            linear-gradient(180deg, #f7fbff 0%, #eef4fb 52%, #ffffff 100%);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #082f49 0%, #0f766e 100%);
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] span {
        color: #f8fafc !important;
    }
    input, textarea, [data-baseweb="select"] > div, [data-baseweb="base-input"] {
        color: #0f172a !important;
    }
    [data-baseweb="input"] input, [data-baseweb="textarea"] textarea {
        background: #ffffff !important;
        color: #0f172a !important;
    }
    .hero {
        background:
            radial-gradient(circle at top left, rgba(255,255,255,0.25), transparent 30%),
            radial-gradient(circle at bottom right, rgba(255,255,255,0.14), transparent 35%),
            linear-gradient(135deg, #0b3b8f 0%, #0d9488 58%, #ea580c 100%);
        padding: 2.25rem;
        border-radius: 28px;
        color: white;
        margin-bottom: 1.2rem;
        box-shadow: 0 24px 50px rgba(15, 23, 42, 0.18);
    }
    .soft {
        background: linear-gradient(180deg, #f9fbff 0%, #eef5ff 100%);
        border: 1px solid #dce9ff;
        border-radius: 18px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
    }
    .offer {
        background: #f4fff9;
        border-left: 5px solid #0d9488;
        border-radius: 14px;
        padding: 1rem;
    }
    .hero-kicker {
        text-transform: uppercase;
        letter-spacing: 0.22rem;
        font-size: 0.78rem;
        opacity: 0.9;
        margin-bottom: 0.8rem;
        font-weight: 700;
    }
    .insight-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 1rem;
        margin: 1rem 0 1.2rem 0;
    }
    .insight-card {
        background: rgba(255,255,255,0.92);
        border: 1px solid rgba(255,255,255,0.7);
        border-radius: 22px;
        padding: 1rem 1.1rem;
        box-shadow: 0 14px 30px rgba(15, 23, 42, 0.09);
    }
    .insight-label {
        color: #475569;
        font-size: 0.84rem;
        text-transform: uppercase;
        letter-spacing: 0.08rem;
        margin-bottom: 0.4rem;
    }
    .insight-value {
        color: #0f172a;
        font-size: 2rem;
        font-weight: 800;
        line-height: 1.1;
    }
    .insight-note {
        color: #64748b;
        font-size: 0.92rem;
        margin-top: 0.35rem;
    }
    .spotlight {
        background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
        border: 1px solid #d7e6ff;
        border-radius: 24px;
        padding: 1.2rem 1.3rem;
        box-shadow: 0 18px 36px rgba(15, 23, 42, 0.08);
        margin-bottom: 1rem;
    }
    .pill-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.7rem;
        margin-top: 1rem;
    }
    .pill {
        background: rgba(255,255,255,0.16);
        border: 1px solid rgba(255,255,255,0.18);
        border-radius: 999px;
        padding: 0.5rem 0.9rem;
        font-size: 0.92rem;
    }
    .version-chip {
        display: inline-block;
        background: #fff7ed;
        color: #9a3412;
        border: 1px solid #fdba74;
        border-radius: 999px;
        padding: 0.32rem 0.8rem;
        font-size: 0.82rem;
        font-weight: 700;
        margin-bottom: 0.9rem;
    }
    .login-shell {
        max-width: 860px;
        margin: 2rem auto 0 auto;
    }
    .login-card {
        background: rgba(255,255,255,0.96);
        border: 1px solid #d7e6ff;
        border-radius: 24px;
        padding: 1.4rem 1.5rem;
        box-shadow: 0 24px 45px rgba(15, 23, 42, 0.1);
    }
    .decision-banner {
        background: linear-gradient(135deg, #082f49 0%, #0f766e 100%);
        color: #ffffff;
        border-radius: 24px;
        padding: 1.35rem 1.5rem;
        margin: 0.8rem 0 1rem 0;
        box-shadow: 0 20px 44px rgba(8, 47, 73, 0.18);
    }
    .decision-title {
        font-size: 0.8rem;
        letter-spacing: 0.12rem;
        text-transform: uppercase;
        opacity: 0.82;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }
    .decision-text {
        font-size: 1.5rem;
        font-weight: 800;
        line-height: 1.25;
    }
    .action-box {
        background: linear-gradient(180deg, #fffaf0 0%, #ffffff 100%);
        border: 1px solid #fed7aa;
        border-radius: 18px;
        padding: 1rem 1.1rem;
        margin-bottom: 0.9rem;
    }
    .action-box h4 {
        margin: 0 0 0.35rem 0;
        color: #9a3412;
    }
    .action-box p {
        margin: 0.2rem 0;
        color: #475569;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_session():
    return SessionLocal()


if BOOTSTRAP_DEMO_ON_STARTUP and ENABLE_DEMO_MODE and not st.session_state.get("_bootstrap_done", False):
    session_for_bootstrap = get_session()
    try:
        bootstrap_demo_environment(session_for_bootstrap)
    except Exception as exc:
        logger.warning("Demo bootstrap skipped after startup issue: %s", exc)
    finally:
        session_for_bootstrap.close()
        st.session_state["_bootstrap_done"] = True


def fetch_companies():
    session = get_session()
    try:
        return session.query(Company).order_by(Company.name).all()
    finally:
        session.close()


def fetch_customers(company_id=None, limit=200):
    session = get_session()
    try:
        query = session.query(CustomerRecord)
        if company_id:
            query = query.filter(CustomerRecord.company_id == company_id)
        return query.order_by(CustomerRecord.created_at.desc()).limit(limit).all()
    finally:
        session.close()


def fetch_actions(company_id=None, limit=200):
    session = get_session()
    try:
        query = session.query(RetentionAction)
        if company_id:
            query = query.filter(RetentionAction.company_id == company_id)
        return query.order_by(RetentionAction.created_at.desc()).limit(limit).all()
    finally:
        session.close()


def fetch_uploads(company_id=None, limit=20, user_email=None, dataset_role=None):
    session = get_session()
    try:
        query = session.query(DatasetUpload)
        if company_id:
            query = query.filter(DatasetUpload.company_id == company_id)
        if user_email:
            query = query.filter(DatasetUpload.uploaded_by_email == user_email)
        if dataset_role:
            query = query.filter(DatasetUpload.dataset_role == dataset_role)
        return query.order_by(DatasetUpload.created_at.desc()).limit(limit).all()
    finally:
        session.close()


@st.cache_data(show_spinner=False)
def load_uploaded_dataset_cached(stored_path: str) -> pd.DataFrame:
    path = Path(str(stored_path))
    if not path.exists():
        return pd.DataFrame()
    if path.suffix.lower() in {".xlsx", ".xls"}:
        raw_dataset = pd.read_excel(path)
    else:
        raw_dataset = pd.read_csv(path)
    prepared_dataset, schema_mapping = prepare_company_dataset(raw_dataset)
    enriched_dataset, _ = infer_missing_business_signals(prepared_dataset, schema_mapping)
    return enriched_dataset


def load_uploaded_dataset(upload: DatasetUpload | None) -> pd.DataFrame:
    if upload is None:
        return pd.DataFrame()
    return load_uploaded_dataset_cached(str(upload.stored_path))


@st.cache_resource(show_spinner=False)
def get_cached_artifacts():
    return load_artifacts()


def safe_dataset_churn_rate(df: pd.DataFrame) -> float:
    churn_column = next((column for column in ["Churn", "churn", "Churn Label", "churn_label"] if column in df.columns), None)
    if not churn_column or df.empty:
        return 0.0
    values = df[churn_column].astype(str).str.strip().str.lower()
    positive = values.isin({"yes", "1", "true", "y", "churned"})
    return round(float(positive.mean() * 100), 1)


def series_from_frame(df: pd.DataFrame, column_name: str, default_value) -> pd.Series:
    if column_name in df.columns:
        return df[column_name]
    return pd.Series([default_value] * len(df), index=df.index)


def delete_dataset_upload(upload_id: int, user_email: str) -> bool:
    session = get_session()
    try:
        upload = (
            session.query(DatasetUpload)
            .filter(DatasetUpload.id == upload_id, DatasetUpload.uploaded_by_email == user_email)
            .one_or_none()
        )
        if upload is None:
            return False
        file_path = Path(str(upload.stored_path or ""))
        session.delete(upload)
        session.commit()
        if file_path.exists():
            try:
                file_path.unlink()
            except OSError:
                pass
        return True
    finally:
        session.close()


def company_summary(company_id=None):
    customers = fetch_customers(company_id, limit=1000)
    actions = fetch_actions(company_id, limit=1000)
    revenue_at_risk = round(
        sum((row.customer_value or 0) for row in customers if (row.churn_probability or 0) >= 0.65),
        2,
    )
    return {
        "companies": len(fetch_companies()),
        "customers": len(customers),
        "high_risk": sum(1 for row in customers if (row.churn_probability or 0) >= 0.65),
        "avg_churn": round((sum((row.churn_probability or 0) for row in customers) / len(customers) * 100), 2) if customers else 0,
        "actions": len(actions),
        "revenue_at_risk": revenue_at_risk,
        "potential_recovery": round(revenue_at_risk * 0.6, 2),
    }


def build_manager_narrative(data: pd.DataFrame, action_rows: list[dict], forecast_data: dict) -> dict:
    if data.empty:
        return {
            "headline": "No scored customers available yet.",
            "subline": "Upload or score customer records to generate churn decisions and action priorities.",
            "top_issue": "No issue data",
            "top_segment": "No segment data",
            "top_action": "Score a dataset first",
        }

    avg_churn = float(data["Churn Probability"].mean())
    high_risk_count = int((data["Churn Probability"] >= 65).sum())
    total_customers = int(len(data))
    high_risk_share = round((high_risk_count / total_customers) * 100, 1) if total_customers else 0.0
    top_issue = (
        data["Issue Category"].fillna("General Feedback").astype(str).value_counts().idxmax()
        if "Issue Category" in data.columns and not data["Issue Category"].dropna().empty
        else "General Feedback"
    )
    top_segment = (
        data["Segment"].fillna("Growth Opportunity").astype(str).value_counts().idxmax()
        if "Segment" in data.columns and not data["Segment"].dropna().empty
        else "Growth Opportunity"
    )
    next_month = float(forecast_data.get("next_month_churn_pct", 0))
    top_action = "Launch proactive retention outreach"
    if action_rows:
        top_action = str(action_rows[0].get("recommended_actions", ["Launch proactive retention outreach"])[0])

    return {
        "headline": f"{high_risk_count} customers need immediate attention; churn exposure is {avg_churn:.1f}% on average.",
        "subline": f"{high_risk_share}% of monitored customers are already in high-risk bands, mainly driven by {top_issue.lower()} issues. Next-month forecast is {next_month:.1f}%.",
        "top_issue": top_issue,
        "top_segment": top_segment,
        "top_action": top_action,
    }


def top_action_table(action_rows: list[dict]) -> pd.DataFrame:
    rows = []
    for item in action_rows[:8]:
        next_action = item.get("recommended_actions", ["Review account"])[0]
        rows.append(
            {
                "Customer": item.get("customer_name") or item.get("customer_id"),
                "Risk": item.get("risk_level", "Medium"),
                "Churn %": round(float(item.get("churn_probability", 0)) * 100, 1),
                "Problem": str(item.get("reason", ""))[:110],
                "Next Action": next_action,
                "Owner": item.get("assigned_agent") or "Retention Team",
                "Status": item.get("status", "pending").title(),
            }
        )
    return pd.DataFrame(rows)


def customer_rows(customers):
    rows = []
    for customer in customers:
        rows.append(
            {
                "Customer ID": customer.external_customer_id,
                "Customer Name": customer.customer_name,
                "Email": customer.email,
                "Churn Probability": round((customer.churn_probability or 0) * 100, 2),
                "Segment": customer.churn_segment,
                "Customer Value": round(customer.customer_value or 0, 2),
                "Offer": customer.recommended_offer,
                "Action": customer.recommended_action,
            }
        )
    return rows


def build_customer_analytics_frame(customers, fallback_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for customer in customers:
        payload = customer.raw_payload or {}
        rows.append(
            {
                "Customer ID": customer.external_customer_id,
                "Customer Name": customer.customer_name,
                "Email": customer.email,
                "Gender": payload.get("Gender", payload.get("gender", "Unknown")),
                "Age": payload.get("Age", payload.get("age")),
                "Tenure": payload.get("Tenure", payload.get("tenure")),
                "Subscription Type": payload.get("Subscription_Type", payload.get("subscription_type", "Unknown")),
                "Contract Type": payload.get("Contract_Type", payload.get("contract_type", "Unknown")),
                "Payment Mode": payload.get("Payment_Mode", payload.get("payment_mode", "Unknown")),
                "Monthly Charges": payload.get("Monthly_Charges", payload.get("monthly_charges")),
                "Usage Score": payload.get("Usage_Score", payload.get("usage_score")),
                "Engagement Score": payload.get("Engagement_Score", payload.get("engagement_score")),
                "Issue Category": payload.get("Issue_Category", payload.get("issue_category", "General Feedback")),
                "Sentiment": payload.get("Sentiment", payload.get("sentiment", 0)),
                "Churn Probability": round((customer.churn_probability or 0) * 100, 2),
                "Risk Level": risk_level_from_probability(customer.churn_probability or 0),
                "Segment": customer.churn_segment,
                "Customer Value": round(customer.customer_value or 0, 2),
                "Offer": customer.recommended_offer,
                "Action": customer.recommended_action,
            }
        )
    if rows:
        return pd.DataFrame(rows)

    fallback = heuristic_dataset_scoring(fallback_df.copy())
    fallback["Churn Probability"] = (pd.to_numeric(fallback["churn_probability"], errors="coerce").fillna(0) * 100).round(2)
    fallback["Risk Level"] = fallback["Churn Probability"].apply(lambda value: risk_level_from_probability(value / 100))
    fallback["Segment"] = fallback["customer_segment"]
    fallback["Customer Value"] = pd.to_numeric(fallback["customer_value"], errors="coerce").fillna(0).round(2)
    fallback["Offer"] = fallback["personalized_offer"].apply(lambda item: item.get("offer_name") if isinstance(item, dict) else "Retention Offer")
    fallback["Action"] = fallback["retention_strategy"].apply(lambda item: item.get("action") if isinstance(item, dict) else "Review customer manually")
    if "Customer_Name" not in fallback.columns:
        fallback["Customer_Name"] = fallback["Customer_ID"]
    return fallback.rename(
        columns={
            "Subscription_Type": "Subscription Type",
            "Contract_Type": "Contract Type",
            "Payment_Mode": "Payment Mode",
            "Monthly_Charges": "Monthly Charges",
            "Usage_Score": "Usage Score",
            "Engagement_Score": "Engagement Score",
            "Issue_Category": "Issue Category",
            "Customer_ID": "Customer ID",
            "Customer_Name": "Customer Name",
        }
    )[
        [
            "Customer ID",
            "Customer Name",
            "Email",
            "Gender",
            "Age",
            "Tenure",
            "Subscription Type",
            "Contract Type",
            "Payment Mode",
            "Monthly Charges",
            "Usage Score",
            "Engagement Score",
            "Issue Category",
            "Sentiment",
            "Churn Probability",
            "Risk Level",
            "Segment",
            "Customer Value",
            "Offer",
            "Action",
        ]
    ]


def customer_payload_frame(customers) -> pd.DataFrame:
    rows = []
    for customer in customers:
        payload = dict(customer.raw_payload or {})
        payload["customer_id"] = customer.external_customer_id
        payload["customer_name"] = customer.customer_name
        payload["churn_probability"] = float(customer.churn_probability or 0)
        rows.append(payload)
    return pd.DataFrame(rows)


def is_scalar_like(value) -> bool:
    return not isinstance(value, (dict, list, set, tuple))


def safe_categorical_candidates(df: pd.DataFrame) -> list[str]:
    candidates = []
    for column in df.columns:
        series = df[column].dropna()
        if series.empty:
            continue
        if not series.map(is_scalar_like).all():
            continue
        if df[column].dtype == "object" and series.astype(str).nunique(dropna=True) <= 20:
            candidates.append(column)
    return candidates


def risk_level_from_probability(probability: float) -> str:
    if probability < 0.3:
        return "Low"
    if probability < 0.7:
        return "Medium"
    if probability < 0.85:
        return "High"
    return "Critical"


def prediction_input_frame(record: dict) -> pd.DataFrame:
    frame = pd.DataFrame([record])
    if "Sentiment" not in frame.columns:
        frame["Sentiment"] = frame["Feedback"].apply(simple_sentiment_score)
    if "Issue_Category" not in frame.columns:
        frame["Issue_Category"] = frame["Feedback"].apply(classify_issue)
    if "Engagement_Score" not in frame.columns:
        frame["Engagement_Score"] = (
            frame["Usage_Score"].astype(float) * 0.55
            + (100 - frame["Last_Interaction_Days"].astype(float).clip(0, 100)) * 0.25
            + frame["Tenure"].astype(float).clip(0, 60) * 0.35
        ).clip(1, 100).round(2)
    if "Risk_Indicator" not in frame.columns:
        frame["Risk_Indicator"] = (
            frame["Monthly_Charges"].astype(float) * 0.34
            + frame["Last_Interaction_Days"].astype(float) * 2.1
            + frame["Support_Tickets"].astype(float) * 16
            + frame["Payment_Delay_Days"].astype(float) * 1.8
            - frame["Usage_Score"].astype(float) * 0.45
            - frame["Tenure"].astype(float) * 0.25
        ).round(2)
    return frame


def explain_prediction(input_frame: pd.DataFrame, result: dict) -> str:
    row = input_frame.iloc[0]
    reasons = []
    if float(row["Monthly_Charges"]) >= 1000:
        reasons.append("high monthly charges")
    if float(row["Usage_Score"]) <= 35:
        reasons.append("low product usage")
    if float(row["Last_Interaction_Days"]) >= 30:
        reasons.append("long gap since last interaction")
    if float(row["Sentiment"]) < -0.1:
        reasons.append("negative customer feedback")
    if float(row["Tenure"]) <= 6:
        reasons.append("early-stage onboarding risk")
    issue = str(row["Issue_Category"])
    if issue and issue != "General Feedback":
        reasons.append(issue.lower())

    top_features = [item["feature"] for item in result.get("shap_explanation", [])[:3]]
    if top_features:
        reasons.append("model drivers: " + ", ".join(top_features))

    if not reasons:
        reasons.append("stable engagement and healthy customer behavior")

    return f"Customer is {risk_level_from_probability(result['churn_probability']).lower()} churn risk due to " + ", ".join(reasons) + "."


def prediction_signal_frame(input_frame: pd.DataFrame, result: dict) -> pd.DataFrame:
    row = input_frame.iloc[0]
    probability = float(result["churn_probability"])
    risk_level = risk_level_from_probability(probability)
    usage = float(row.get("Usage_Score", 0) or 0)
    last_interaction = float(row.get("Last_Interaction_Days", 0) or 0)
    support_tickets = float(row.get("Support_Tickets", 0) or 0)
    payment_delay = float(row.get("Payment_Delay_Days", 0) or 0)
    sentiment = float(result.get("sentiment_score", 0) or 0)
    issue = str(row.get("Issue_Category", "General Feedback"))
    monthly = float(row.get("Monthly_Charges", 0) or 0)
    tenure = float(row.get("Tenure", 0) or 0)
    customer_value = float(result.get("customer_value", 0) or 0)
    signals = [
        {
            "Signal": "Overall Risk",
            "Current Value": f"{probability * 100:.1f}% ({risk_level})",
            "Interpretation": "This is the final churn probability after combining model and business signals.",
            "Recommended Response": "Use the priority level to decide whether to monitor, engage, or escalate immediately.",
        },
        {
            "Signal": "Usage",
            "Current Value": f"{usage:.1f}/100",
            "Interpretation": "Low usage means the customer may not be receiving enough value.",
            "Recommended Response": "Offer usage coaching, feature education, or a usage booster pack." if usage < 45 else "Maintain value-based engagement.",
        },
        {
            "Signal": "Recent Interaction",
            "Current Value": f"{last_interaction:.0f} days",
            "Interpretation": "A long gap since the last interaction increases disengagement risk.",
            "Recommended Response": "Send a proactive check-in and schedule follow-up." if last_interaction >= 30 else "Keep the customer in normal nurture communication.",
        },
        {
            "Signal": "Support Load",
            "Current Value": f"{support_tickets:.0f} tickets",
            "Interpretation": "More tickets can indicate unresolved service friction.",
            "Recommended Response": "Assign priority support and close the issue loop." if support_tickets >= 3 else "Monitor support satisfaction.",
        },
        {
            "Signal": "Payment Delay",
            "Current Value": f"{payment_delay:.0f} days",
            "Interpretation": "Payment delay can signal affordability or billing friction.",
            "Recommended Response": "Offer billing help, payment reminders, or plan correction." if payment_delay >= 10 else "No urgent payment intervention required.",
        },
        {
            "Signal": "Feedback Issue",
            "Current Value": issue,
            "Interpretation": f"The main issue detected from feedback is {issue.lower()}.",
            "Recommended Response": "Address this exact issue in the retention message.",
        },
        {
            "Signal": "Sentiment",
            "Current Value": f"{result.get('sentiment_label', 'Neutral')} ({sentiment:+.2f})",
            "Interpretation": "Negative sentiment needs service recovery; neutral sentiment needs clearer value communication.",
            "Recommended Response": "Use an empathetic message and avoid a generic campaign.",
        },
        {
            "Signal": "Commercial Value",
            "Current Value": f"{customer_value:,.2f}",
            "Interpretation": f"Tenure is {tenure:.0f} months and monthly charges are {monthly:,.2f}.",
            "Recommended Response": "Prioritize faster action when value is high or pricing pressure is detected.",
        },
    ]
    return pd.DataFrame(signals)


def prediction_policy_frame(input_frame: pd.DataFrame, result: dict) -> pd.DataFrame:
    probability = float(result["churn_probability"])
    risk_level = risk_level_from_probability(probability)
    priority = result.get("retention_strategy", {}).get("priority", "Medium")
    rows = [
        {"Decision Area": "Risk Policy", "Recommendation": f"{risk_level} risk requires {priority.lower()} priority handling."},
        {"Decision Area": "Primary Strategy", "Recommendation": result.get("retention_strategy", {}).get("strategy", "Review customer manually.")},
        {"Decision Area": "Immediate Action", "Recommendation": result.get("retention_strategy", {}).get("action", "Review customer manually.")},
        {"Decision Area": "Offer Policy", "Recommendation": result.get("personalized_offer", {}).get("offer_details", "Use a targeted retention offer.")},
        {"Decision Area": "Follow-up Rule", "Recommendation": "Track response within 72 hours and re-score if usage, payment, or sentiment changes."},
    ]
    if probability >= 0.85:
        rows.append({"Decision Area": "Escalation", "Recommendation": "Escalate to a senior retention owner immediately."})
    elif probability >= 0.65:
        rows.append({"Decision Area": "Escalation", "Recommendation": "Assign to the retention team today and confirm contact completion."})
    else:
        rows.append({"Decision Area": "Escalation", "Recommendation": "Keep in nurture flow and monitor behavior changes."})
    return pd.DataFrame(rows)


def prediction_driver_frame(input_frame: pd.DataFrame, result: dict) -> pd.DataFrame:
    drivers = []
    for item in result.get("shap_explanation", [])[:5]:
        drivers.append(
            {
                "Driver": str(item.get("feature", "")).replace("num__", "").replace("cat__", ""),
                "Impact": round(abs(float(item.get("contribution", 0) or 0)), 4),
                "Direction": item.get("direction", "Increase Risk"),
            }
        )

    business_signals = [
        ("Raw Model", float(result.get("raw_model_probability", result["churn_probability"]))),
        ("Hybrid Risk", float(result.get("heuristic_probability", result["churn_probability"]))),
        ("Final Probability", float(result["churn_probability"])),
        ("Retention Success", float(result.get("retention_success_probability", 0))),
    ]
    for label, score in business_signals:
        drivers.append({"Driver": label, "Impact": round(score, 4), "Direction": "Reference"})
    return pd.DataFrame(drivers)


def customer_health_frame(input_frame: pd.DataFrame, result: dict) -> pd.DataFrame:
    row = input_frame.iloc[0]
    return pd.DataFrame(
        [
            {"Metric": "Usage Score", "Value": float(row["Usage_Score"])},
            {"Metric": "Sentiment", "Value": float(result.get("sentiment_score", 0)) * 100},
            {"Metric": "Recency Health", "Value": max(0.0, 100 - float(row["Last_Interaction_Days"]))},
            {"Metric": "Payment Discipline", "Value": max(0.0, 100 - float(row["Payment_Delay_Days"]) * 2)},
            {"Metric": "Support Stability", "Value": max(0.0, 100 - float(row["Support_Tickets"]) * 12)},
        ]
    )


def build_offer_options(input_frame: pd.DataFrame, result: dict) -> pd.DataFrame:
    row = input_frame.iloc[0]
    probability = float(result["churn_probability"])
    issue = str(row.get("Issue_Category", "General Feedback"))
    monthly = float(row.get("Monthly_Charges", 0) or 0)
    usage = float(row.get("Usage_Score", 0) or 0)
    sentiment = float(result.get("sentiment_score", 0) or 0)
    last_interaction = float(row.get("Last_Interaction_Days", 0) or 0)
    support_tickets = float(row.get("Support_Tickets", 0) or 0)
    offers = [
        {
            "Offer": result.get("personalized_offer", {}).get("offer_name", "Recommended Retention Offer"),
            "Best For": "Selected customer profile",
            "Details": result.get("personalized_offer", {}).get("offer_details", "Use the recommended retention offer."),
            "Priority": result.get("retention_strategy", {}).get("priority", "Medium"),
            "Why This Fits": "This is the model-selected offer for the current profile.",
            "Expected Impact": "Highest contextual fit",
        }
    ]
    offers.extend(
        [
            {
                "Offer": "Price Protection Plan",
                "Best For": "Price-sensitive customers",
                "Details": "Offer temporary bill relief, bundle correction, or price-lock protection.",
                "Priority": "High" if issue == "Pricing Issue" or monthly >= 800 else "Medium",
                "Why This Fits": "Useful when monthly charges or pricing feedback may be creating churn pressure.",
                "Expected Impact": "Reduce price objection",
            },
            {
                "Offer": "Usage Booster Pack",
                "Best For": "Low-engagement customers",
                "Details": "Provide extra usage credits, onboarding support, and feature education.",
                "Priority": "High" if usage < 40 else "Medium",
                "Why This Fits": "Useful when the customer is not using enough product value.",
                "Expected Impact": "Increase engagement",
            },
            {
                "Offer": "Service Recovery Credit",
                "Best For": "Service or complaint-led customers",
                "Details": "Give a small service credit with a priority support callback and resolution timeline.",
                "Priority": "High" if issue in {"Service Issue", "Network Issue"} or sentiment < -0.1 or support_tickets >= 3 else "Medium",
                "Why This Fits": "Useful when support friction, service feedback, or negative sentiment is present.",
                "Expected Impact": "Recover trust",
            },
            {
                "Offer": "Priority Retention Save",
                "Best For": "High or critical risk customers",
                "Details": "Assign a retention specialist, call within 24 hours, and approve a custom save offer.",
                "Priority": "Critical" if probability >= 0.85 else "High" if probability >= 0.65 else "Medium",
                "Why This Fits": "Useful when churn probability is high enough to justify direct human intervention.",
                "Expected Impact": "Prevent immediate churn",
            },
            {
                "Offer": "Re-Engagement Plan",
                "Best For": "Inactive customers",
                "Details": "Send a goal-based check-in, product education, and a 72-hour follow-up reminder.",
                "Priority": "High" if last_interaction >= 30 else "Medium",
                "Why This Fits": "Useful when the customer has not interacted recently.",
                "Expected Impact": "Restart contact",
            },
            {
                "Offer": "Loyalty Reward",
                "Best For": "Stable or medium-risk customers",
                "Details": "Give loyalty points, referral benefits, and a personalized appreciation message.",
                "Priority": "Low" if probability < 0.4 else "Medium",
                "Why This Fits": "Useful for keeping stable customers satisfied without over-discounting.",
                "Expected Impact": "Maintain loyalty",
            },
        ]
    )
    return pd.DataFrame(offers).drop_duplicates(subset=["Offer"], keep="first")


def build_contact_plan(input_frame: pd.DataFrame, result: dict) -> pd.DataFrame:
    probability = float(result["churn_probability"])
    row = input_frame.iloc[0]
    issue = str(row.get("Issue_Category", "General Feedback"))
    email = str(row.get("Email", "customer@example.com"))
    customer_id = str(row.get("Customer_ID", "Customer"))
    return pd.DataFrame(
        [
            {
                "Channel": "Email",
                "When To Use": "Default follow-up and offer confirmation",
                "Message": f"Send a personalized retention email to {email} addressing {issue.lower()} and explaining the selected offer.",
            },
            {
                "Channel": "Phone Call",
                "When To Use": "Use immediately for high or critical risk",
                "Message": f"Call customer {customer_id} with a retention specialist if churn probability is above 65%.",
            },
            {
                "Channel": "SMS / WhatsApp",
                "When To Use": "Use for quick reminders and response tracking",
                "Message": "Send a short message with the offer summary and a callback link.",
            },
            {
                "Channel": "Action Center Assignment",
                "When To Use": "Use when a retention owner must track the case",
                "Message": "Assign the customer to a retention agent and track response within 72 hours.",
            },
        ]
    )


def prediction_storage_key(dataset_name: str, customer_id: str) -> str:
    return f"{dataset_name}::{customer_id}"


def save_prediction_history(dataset_name: str, selected_label: str, input_frame: pd.DataFrame, result: dict) -> None:
    history = st.session_state.setdefault("prediction_history", {})
    row = input_frame.iloc[0]
    key = prediction_storage_key(dataset_name, str(row["Customer_ID"]))
    history[key] = {
        "dataset": dataset_name,
        "selected_label": selected_label,
        "customer_id": str(row["Customer_ID"]),
        "customer_name": str(row.get("Customer_Name", row.get("Customer_ID", "Customer"))),
        "email": str(row.get("Email", "")),
        "risk_level": risk_level_from_probability(float(result["churn_probability"])),
        "churn_probability": float(result["churn_probability"]),
        "customer_value": float(result.get("customer_value", 0)),
        "issue_category": str(row.get("Issue_Category", "General Feedback")),
        "offer": result.get("personalized_offer", {}).get("offer_name", "Retention Offer"),
        "action": result.get("retention_strategy", {}).get("action", "Review customer manually."),
        "strategy": result.get("retention_strategy", {}).get("strategy", "Review"),
        "priority": result.get("retention_strategy", {}).get("priority", "Medium"),
        "predicted_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def prediction_history_frame(dataset_name: str | None = None) -> pd.DataFrame:
    history = list(st.session_state.get("prediction_history", {}).values())
    if dataset_name:
        history = [item for item in history if item.get("dataset") == dataset_name]
    if not history:
        return pd.DataFrame()
    frame = pd.DataFrame(history)
    frame["Churn Probability"] = (pd.to_numeric(frame["churn_probability"], errors="coerce").fillna(0) * 100).round(1).astype(str) + "%"
    return frame.rename(
        columns={
            "customer_id": "Customer ID",
            "customer_name": "Customer Name",
            "risk_level": "Risk Level",
            "issue_category": "Issue Category",
            "offer": "Offer",
            "action": "Next Action",
            "priority": "Priority",
            "predicted_at": "Predicted At",
        }
    )


def mark_action_status(customer_id: str, action_name: str, status: str = "completed") -> None:
    statuses = st.session_state.setdefault("retention_action_statuses", {})
    customer_status = statuses.setdefault(str(customer_id), {})
    customer_status[action_name] = status


def dataset_profile(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in df.columns:
        rows.append(
            {
                "Column": column,
                "Type": str(df[column].dtype),
                "Missing Values": int(df[column].isna().sum()),
                "Unique Values": int(df[column].nunique(dropna=True)),
                "Sample": str(df[column].dropna().iloc[0]) if df[column].dropna().shape[0] else "N/A",
            }
        )
    return pd.DataFrame(rows)


DEMO_DATASETS = {
    "Native Sample Dataset": Path("data/customer_churn.csv"),
    "IBM Benchmark Sample": Path("data/ibm_telco_benchmark_sample.csv"),
}


@st.cache_data(show_spinner=False)
def load_demo_dataset(label: str) -> pd.DataFrame:
    path = DEMO_DATASETS.get(label, DEMO_DATASETS["Native Sample Dataset"])
    dataset = pd.read_csv(path)
    rename_map = {
        "CustomerID": "Customer_ID",
        "Gender": "Gender",
        "Tenure": "Tenure",
        "Contract": "Contract_Type",
        "PaymentMethod": "Payment_Mode",
        "Monthly Charges": "Monthly_Charges",
        "Total Charges": "Total_Spend",
        "Feedback": "Feedback",
        "Sentiment": "Sentiment",
        "Issue Category": "Issue_Category",
        "Churn Label": "Churn",
    }
    normalized = dataset.rename(columns={key: value for key, value in rename_map.items() if key in dataset.columns}).copy()
    return normalized


def demo_customer_options(df: pd.DataFrame) -> list[str]:
    if "Customer_ID" not in df.columns:
        return []
    labels = []
    for _, row in df.iterrows():
        customer_id = str(row.get("Customer_ID", ""))
        issue = str(row.get("Issue_Category", row.get("Issue Category", "General Feedback")))
        customer_name = str(row.get("Customer_Name", row.get("Customer Name", customer_id)))
        labels.append(f"{customer_id} | {customer_name} | {issue}")
    return labels


def dataset_sentiment_label(value: float) -> str:
    if value <= -0.15:
        return "Negative"
    if value >= 0.15:
        return "Positive"
    return "Neutral"


@st.cache_data(show_spinner=False)
def cached_dataset_views(df_json: str, artifact_signature: str) -> tuple[pd.DataFrame, dict, dict]:
    dataset = pd.read_json(io.StringIO(df_json), orient="split")
    cached_artifacts = get_cached_artifacts() if artifact_signature != "no-model" else None
    return build_dataset_driven_views(dataset, cached_artifacts)


def prepare_dashboard_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    if len(df) <= MAX_DASHBOARD_ROWS:
        return df.copy(), False
    sampled = df.sample(n=MAX_DASHBOARD_ROWS, random_state=42).sort_index()
    return sampled.reset_index(drop=True), True


def heuristic_dataset_scoring(df: pd.DataFrame) -> pd.DataFrame:
    scored = df.copy()
    if scored.empty:
        return scored
    probabilities = scored.apply(lambda row: heuristic_churn_probability(row), axis=1)
    scored["raw_model_probability"] = probabilities
    scored["heuristic_probability"] = probabilities
    scored["churn_probability"] = probabilities.round(4)
    scored["churn_prediction"] = scored["churn_probability"] >= 0.55
    scored["customer_segment"] = np.where(
        scored["churn_probability"] >= 0.8,
        "Critical Churn",
        np.where(scored["churn_probability"] >= 0.6, "High Attention", np.where(scored["churn_probability"] >= 0.35, "Growth Opportunity", "Loyal Base")),
    )
    scored["customer_value"] = scored.apply(customer_value_score, axis=1)
    scored["retention_strategy"] = scored.apply(
        lambda row: retention_strategy(row, float(row["churn_probability"]), float(row["customer_value"])),
        axis=1,
    )
    scored["personalized_offer"] = scored.apply(
        lambda row: personalized_offer(row, float(row["churn_probability"])),
        axis=1,
    )
    scored["reason"] = scored.apply(
        lambda row: (
            f"Customer risk is driven by {str(row.get('Issue_Category', 'general behavior')).lower()}, "
            f"tenure {int(float(row.get('Tenure', 0) or 0))}, usage score {float(row.get('Usage_Score', 50) or 50):.1f}, "
            f"and sentiment {float(row.get('Sentiment', 0) or 0):+.2f}."
        ),
        axis=1,
    )
    return scored


def build_dataset_driven_views(df: pd.DataFrame, artifacts) -> tuple[pd.DataFrame, dict, dict]:
    working = df.copy()
    try:
        if artifacts is not None:
            scored = score_customers(working, artifacts)
        else:
            scored = heuristic_dataset_scoring(working)
            if "Churn" in scored.columns:
                churn_override = np.where(
                    scored["Churn"].astype(str).str.lower().isin({"yes", "1", "true", "y", "churned"}),
                    np.maximum(scored["churn_probability"], 0.78),
                    np.minimum(scored["churn_probability"], 0.22),
                )
                scored["churn_probability"] = churn_override.round(4)
                scored["raw_model_probability"] = scored["churn_probability"]
                scored["heuristic_probability"] = scored["churn_probability"]
                scored["churn_prediction"] = scored["churn_probability"] >= 0.55
    except (ValueError, TypeError, AttributeError) as exc:
        logger.warning("Falling back to heuristic dataset scoring in dashboard views: %s", exc)
        scored = heuristic_dataset_scoring(working)

    analytics = scored.copy()
    analytics["Customer ID"] = analytics.get("Customer_ID", analytics.get("CustomerID"))
    analytics["Customer Name"] = analytics.get("Customer_Name", analytics["Customer ID"])
    analytics["Email"] = series_from_frame(analytics, "Email", "")
    analytics["Gender"] = series_from_frame(analytics, "Gender", "Unknown")
    analytics["Age"] = pd.to_numeric(series_from_frame(analytics, "Age", np.nan), errors="coerce")
    analytics["Tenure"] = pd.to_numeric(series_from_frame(analytics, "Tenure", np.nan), errors="coerce")
    analytics["Subscription Type"] = (
        analytics["Subscription_Type"] if "Subscription_Type" in analytics.columns
        else series_from_frame(analytics, "Subscription Type", "Unknown")
    )
    analytics["Contract Type"] = (
        analytics["Contract_Type"] if "Contract_Type" in analytics.columns
        else series_from_frame(analytics, "Contract", "Unknown")
    )
    analytics["Payment Mode"] = (
        analytics["Payment_Mode"] if "Payment_Mode" in analytics.columns
        else series_from_frame(analytics, "PaymentMethod", "Unknown")
    )
    monthly_charge_source = analytics["Monthly_Charges"] if "Monthly_Charges" in analytics.columns else series_from_frame(analytics, "Monthly Charges", np.nan)
    usage_score_source = analytics["Usage_Score"] if "Usage_Score" in analytics.columns else series_from_frame(analytics, "Usage Score", 50)
    engagement_source = analytics["Engagement_Score"] if "Engagement_Score" in analytics.columns else series_from_frame(analytics, "Engagement Score", np.nan)
    issue_source = analytics["Issue_Category"] if "Issue_Category" in analytics.columns else series_from_frame(analytics, "Issue Category", "General Feedback")
    sentiment_source = series_from_frame(analytics, "Sentiment", 0.0)
    churn_source = series_from_frame(analytics, "churn_probability", 0.0)
    customer_value_source = series_from_frame(analytics, "customer_value", 0.0)
    analytics["Monthly Charges"] = pd.to_numeric(monthly_charge_source, errors="coerce")
    analytics["Usage Score"] = pd.to_numeric(usage_score_source, errors="coerce")
    analytics["Engagement Score"] = pd.to_numeric(engagement_source, errors="coerce").fillna(analytics["Usage Score"])
    analytics["Issue Category"] = issue_source.fillna("General Feedback")
    analytics["Sentiment"] = pd.to_numeric(sentiment_source, errors="coerce").fillna(0.0)
    analytics["Churn Probability"] = (pd.to_numeric(churn_source, errors="coerce").fillna(0.0) * 100).round(2)
    analytics["Risk Level"] = analytics["Churn Probability"].apply(lambda value: risk_level_from_probability(value / 100))
    analytics["Segment"] = analytics.get("customer_segment", analytics.get("Segment", "Growth Opportunity"))
    analytics["Customer Value"] = pd.to_numeric(customer_value_source, errors="coerce").fillna(0.0).round(2)
    analytics["Analytics Source"] = "Model Scored" if artifacts is not None else "Heuristic Estimated"
    offer_series = analytics["personalized_offer"] if "personalized_offer" in analytics.columns else pd.Series([{}] * len(analytics), index=analytics.index)
    strategy_series = analytics["retention_strategy"] if "retention_strategy" in analytics.columns else pd.Series([{}] * len(analytics), index=analytics.index)
    analytics["Offer"] = offer_series.apply(lambda item: item.get("offer_name") if isinstance(item, dict) else "Retention Offer")
    analytics["Action"] = strategy_series.apply(lambda item: item.get("action") if isinstance(item, dict) else "Review customer manually")
    analytics["Strategy"] = strategy_series.apply(lambda item: item.get("strategy") if isinstance(item, dict) else "Review")
    analytics["Priority"] = strategy_series.apply(lambda item: item.get("priority") if isinstance(item, dict) else "Medium")
    analytics["Reason"] = analytics.get("reason", pd.Series(["Model scored this customer for churn risk."] * len(analytics), index=analytics.index))

    sentiment_labels = analytics["Sentiment"].apply(dataset_sentiment_label)
    total = max(len(analytics), 1)
    nlp_data = {
        "issue_summary": [
            {"issue": issue, "count": int(count), "percentage": round((count / total) * 100, 2)}
            for issue, count in analytics["Issue Category"].fillna("General Feedback").astype(str).value_counts().items()
        ],
        "sentiment_summary": [
            {"sentiment": label, "count": int(count), "percentage": round((count / total) * 100, 2)}
            for label, count in sentiment_labels.value_counts().items()
        ],
        "keywords": extract_keywords(series_from_frame(analytics, "Feedback", "").fillna("").astype(str).tolist(), top_n=10),
        "keyword_summary": [],
        "issue_sentiment_summary": [],
        "recent_feedback": [],
        "feedback_count": int(len(analytics)),
        "avg_sentiment_score": round(float(analytics["Sentiment"].mean()), 3) if not analytics.empty else 0.0,
    }

    forecast_data = {
        "next_month_churn_pct": round(float(analytics["Churn Probability"].mean()), 2) if not analytics.empty else 0.0,
        "trend": [],
    }
    return analytics, nlp_data, forecast_data


def build_preview_summary(data: pd.DataFrame) -> dict:
    if data.empty:
        return {"companies": 1, "customers": 0, "high_risk": 0, "avg_churn": 0.0, "actions": 0, "revenue_at_risk": 0.0, "potential_recovery": 0.0}
    revenue_at_risk = round(float(data.loc[data["Churn Probability"] >= 65, "Customer Value"].fillna(0).sum()), 2)
    high_risk = int((data["Churn Probability"] >= 65).sum())
    return {
        "companies": 1,
        "customers": int(len(data)),
        "high_risk": high_risk,
        "avg_churn": round(float(data["Churn Probability"].mean()), 2),
        "actions": high_risk,
        "revenue_at_risk": revenue_at_risk,
        "potential_recovery": round(revenue_at_risk * 0.6, 2),
    }


def build_preview_action_center(data: pd.DataFrame) -> list[dict]:
    if data.empty:
        return []
    focus = data.sort_values(["Churn Probability", "Customer Value"], ascending=[False, False]).head(25)
    rows = []
    for index, row in focus.iterrows():
        probability = float(row.get("Churn Probability", 0)) / 100
        if probability < 0.6:
            continue
        rows.append(
            {
                "customer_record_id": int(index) + 1,
                "customer_id": row.get("Customer ID"),
                "customer_name": row.get("Customer Name"),
                "churn_probability": round(probability, 4),
                "risk_level": row.get("Risk Level", "Medium"),
                "reason": row.get("Reason", "Model scored this customer for churn risk."),
                "recommended_actions": [
                    row.get("Action", "Review customer manually"),
                    f"Offer {row.get('Offer', 'Retention Offer')}",
                    "Follow up in 72 hours",
                ],
                "offer": row.get("Offer", "Retention Offer"),
                "status": "pending",
                "assigned_agent": None,
                "email": row.get("Email"),
                "action_statuses": {"call": "pending", "email": "pending", "offer": "pending", "assign": "pending"},
            }
        )
    return rows


def build_preview_workflow(data: pd.DataFrame) -> dict:
    if data.empty:
        return {"stage_counts": {}, "status_counts": {}, "records": []}
    records = []
    for index, row in data.sort_values("Churn Probability", ascending=False).head(40).iterrows():
        risk = float(row.get("Churn Probability", 0))
        owner = "Retention Team" if risk >= 65 else "Customer Success"
        action_status = "pending" if risk >= 65 else "completed"
        records.append(
            {
                "customer_record_id": int(index) + 1,
                "customer_id": row.get("Customer ID"),
                "stage": "Risk Detection",
                "status": "completed",
                "owner": "ML Engine",
                "deadline": None,
                "outcome": row.get("Risk Level"),
            }
        )
        records.append(
            {
                "customer_record_id": int(index) + 1,
                "customer_id": row.get("Customer ID"),
                "stage": "Action",
                "status": action_status,
                "owner": owner,
                "deadline": None,
                "outcome": row.get("Action"),
            }
        )
    stage_counts = pd.DataFrame(records)["stage"].value_counts().to_dict() if records else {}
    status_counts = pd.DataFrame(records)["status"].value_counts().to_dict() if records else {}
    return {"stage_counts": stage_counts, "status_counts": status_counts, "records": records}


def clear_auth_state() -> None:
    for key in ["authenticated", "auth_token", "user_email", "user_role", "user_name"]:
        st.session_state.pop(key, None)


def restore_auth_session() -> None:
    token = st.session_state.get("auth_token")
    if not token:
        clear_auth_state()
        return
    try:
        payload = decode_token(token)
    except Exception:
        clear_auth_state()
        return
    st.session_state.authenticated = True
    st.session_state.user_email = payload.get("email", "")
    st.session_state.user_role = payload.get("role", "manager")
    st.session_state.user_name = st.session_state.get("user_name") or payload.get("email", "")


def render_login_screen() -> None:
    st.markdown(
        """
        <div class="login-shell">
            <div class="hero">
                <div class="hero-kicker">Secure Access</div>
                <h1 style="margin:0;">Retention Intelligence Platform</h1>
                <p style="margin:0.8rem 0 0 0;">
                    Login is required to access predictions, NLP insights, workflow actions, and customer records.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.15, 0.85])
    with left:
        st.markdown('<div class="login-card">', unsafe_allow_html=True)
        st.subheader("Login")
        with st.form("login_form", clear_on_submit=False):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login", use_container_width=True)
        if submitted:
            user = authenticate_user(email.strip(), password)
            if user is None:
                st.error("Invalid email or password.")
            else:
                st.session_state.authenticated = True
                st.session_state.user_email = user.email
                st.session_state.user_role = user.role
                st.session_state.user_name = user.full_name
                st.session_state.auth_token = create_access_token(user)
                st.rerun()
        st.caption("Use your own account to access the platform.")
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown('<div class="login-card">', unsafe_allow_html=True)
        st.subheader("Create Account")
        with st.form("register_form", clear_on_submit=True):
            full_name = st.text_input("Full Name")
            new_email = st.text_input("Work Email")
            new_password = st.text_input("New Password", type="password")
            role = st.selectbox("Role", ["manager", "analyst", "admin"])
            register_submitted = st.form_submit_button("Create Account", use_container_width=True)
        if register_submitted:
            if len(new_password) < 8:
                st.error("Password must be at least 8 characters.")
            elif not new_email.strip():
                st.error("Email is required.")
            else:
                user = create_user(full_name.strip() or "Platform User", new_email.strip(), new_password, role=role)
                st.success(f"Account ready for {user.email}. You can log in now.")
        if not SEED_DEFAULT_USERS:
            st.caption("Default demo accounts are disabled in the current environment.")
        elif not IS_PRODUCTION:
            st.info("Demo accounts are optional and only created when `DEFAULT_ADMIN_PASSWORD` and `DEFAULT_ANALYST_PASSWORD` environment variables are set.")
        st.markdown("</div>", unsafe_allow_html=True)


restore_auth_session()
if not st.session_state.get("authenticated", False):
    render_login_screen()
    st.stop()

st.session_state.setdefault("prediction_history", {})
st.session_state.setdefault("retention_action_statuses", {})

MODULES = [
    "Executive Dashboard",
    "Dataset Intelligence",
    "Customer Insights",
    "Training Lab",
    "Batch Scoring",
    "Customer Predictor",
    "Action Center",
    "NLP Insights",
    "Workflow Monitor",
    "Retention Command Center",
    "Deployment Center",
]
pending_module = st.session_state.pop("pending_module", None)
if pending_module in MODULES:
    st.session_state["active_module"] = pending_module


with st.sidebar:
    st.title("Retention OS")
    st.caption("Industry-ready churn intelligence")
    st.success(f"Active session: {st.session_state.user_email}")
    st.caption(f"Role: {st.session_state.get('user_role', 'manager')}")
    if st.button("Logout", use_container_width=True):
        clear_auth_state()
        st.rerun()
    page = st.radio(
        "Modules",
        MODULES,
        key="active_module",
    )
    companies = fetch_companies()
    company_map = {"All Companies": None}
    for company in companies:
        company_map[company.name] = company.id
    selected_company_name = st.selectbox("Workspace", list(company_map.keys()), index=1 if len(company_map) > 1 else 0)
    selected_company_id = company_map[selected_company_name]
    current_user_email = st.session_state.get("user_email", "")
    user_uploads = fetch_uploads(selected_company_id, limit=50, user_email=current_user_email)
    dataset_option_map = {}
    for upload in user_uploads:
        created_label = upload.created_at.strftime("%d %b %Y %H:%M") if upload.created_at else "Unknown"
        label = f"{upload.filename} | {upload.dataset_role.title()} | {created_label}"
        dataset_option_map[label] = ("upload", upload.id)
    for demo_label in DEMO_DATASETS:
        dataset_option_map[f"Demo | {demo_label}"] = ("demo", demo_label)
    active_dataset_label = st.selectbox(
        "Analysis Dataset",
        list(dataset_option_map.keys()),
        help="Uploaded datasets for the logged-in user appear here. If no upload is available, choose a demo dataset.",
    )


artifacts = None
selected_dataset_type, selected_dataset_value = dataset_option_map[active_dataset_label]
active_dataset_upload = None
if selected_dataset_type == "upload":
    active_dataset_upload = next((upload for upload in user_uploads if upload.id == selected_dataset_value), None)
    seeded_dataset = load_uploaded_dataset(active_dataset_upload)
    if seeded_dataset.empty:
        seeded_dataset = load_demo_dataset("Native Sample Dataset")
        active_dataset_label = "Demo | Native Sample Dataset"
        selected_dataset_type, selected_dataset_value = "demo", "Native Sample Dataset"
else:
    seeded_dataset = load_demo_dataset(str(selected_dataset_value))

active_dataset_name = (
    active_dataset_upload.filename
    if active_dataset_upload is not None
    else str(selected_dataset_value)
)
active_dataset_role = active_dataset_upload.dataset_role if active_dataset_upload is not None else "demo"
active_dataset_source = str(active_dataset_upload.stored_path) if active_dataset_upload is not None else DEMO_DATASETS[str(selected_dataset_value)].as_posix()
pages_requiring_dataset_analysis = {
    "Executive Dashboard",
    "Dataset Intelligence",
    "Customer Insights",
    "Action Center",
    "NLP Insights",
    "Workflow Monitor",
    "Retention Command Center",
}
artifact_signature = "no-model"
dashboard_sampled = False
dashboard_seed_dataset = seeded_dataset.head(0).copy()
dashboard_data = pd.DataFrame()
seeded_nlp_data = {}
seeded_forecast_data = {}
if page in pages_requiring_dataset_analysis:
    dashboard_seed_dataset, dashboard_sampled = prepare_dashboard_dataset(seeded_dataset)
    dataset_json = dashboard_seed_dataset.to_json(orient="split", date_format="iso")
    dashboard_data, seeded_nlp_data, seeded_forecast_data = cached_dataset_views(dataset_json, artifact_signature)
active_demo_dataset_name = active_dataset_name
preview_summary = build_preview_summary(dashboard_data)
action_center_data = build_preview_action_center(dashboard_data)
nlp_data = seeded_nlp_data
workflow_data = build_preview_workflow(dashboard_data)
churn_forecast_data = seeded_forecast_data

if page == "Executive Dashboard":
    summary = preview_summary
    customers = dashboard_data.to_dict(orient="records")

    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">Industry Version • Retention Intelligence Suite</div>
            <h1 style="margin:0;">AI-Powered Customer Retention System</h1>
            <p style="margin:0.8rem 0 0 0; font-size:1.05rem; max-width: 880px;">
                Real-time churn prediction, explainable AI, customer segmentation, retention offers, revenue-risk intelligence,
                email automation, and deployment-ready operations in one executive command center.
            </p>
            <div class="pill-row">
                <div class="pill">Churn Prediction</div>
                <div class="pill">Why Churn</div>
                <div class="pill">Retention Actions</div>
                <div class="pill">Revenue Risk</div>
                <div class="pill">Executive Analytics</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="version-chip">Updated dashboard design is active</div>', unsafe_allow_html=True)
    st.caption(
        f"All numbers below are calculated from the currently selected dataset `{active_demo_dataset_name}` with {len(seeded_dataset):,} customer rows."
    )
    predicted_count = len(prediction_history_frame(active_demo_dataset_name))
    org_summary_df = pd.DataFrame(
        [
            {"Field": "Workspace / Organization", "Value": selected_company_name},
            {"Field": "Dataset", "Value": active_demo_dataset_name},
            {"Field": "Dataset Type", "Value": active_dataset_role},
            {"Field": "Rows Available", "Value": f"{len(seeded_dataset):,}"},
            {"Field": "Customers Predicted This Session", "Value": predicted_count},
            {"Field": "Current User", "Value": current_user_email},
        ]
    )
    st.subheader("Organization And Dataset Overview")
    st.dataframe(org_summary_df, use_container_width=True, hide_index=True)
    if dashboard_sampled:
        st.info(
            f"Fast dashboard mode is active. Live charts are calculated on a {len(dashboard_seed_dataset):,}-row sample from the full {len(seeded_dataset):,}-row dataset so the page opens faster."
        )

    revenue_risk_value = float(summary.get("revenue_at_risk", 0))

    st.markdown(
        f"""
        <div class="insight-grid">
            <div class="insight-card">
                <div class="insight-label">Customers In Dataset</div>
                <div class="insight-value">{summary["customers"]}</div>
                <div class="insight-note">Total customers in the currently selected dataset</div>
            </div>
            <div class="insight-card">
                <div class="insight-label">Customers Needing Action</div>
                <div class="insight-value">{summary["high_risk"]}</div>
                <div class="insight-note">Customers with churn probability above 65%</div>
            </div>
            <div class="insight-card">
                <div class="insight-label">Average Churn Risk</div>
                <div class="insight-value">{summary["avg_churn"]}%</div>
                <div class="insight-note">Average churn probability across the selected dataset</div>
            </div>
            <div class="insight-card">
                <div class="insight-label">Selected Dataset</div>
                <div class="insight-value" style="font-size:1.05rem;">{active_demo_dataset_name}</div>
                <div class="insight-note">All charts below are based on this dataset only</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1.25, 1])
    with col1:
        st.markdown(
            f"""
            <div class="spotlight">
                <div class="hero-kicker" style="color:#0f172a; opacity:1;">Revenue Risk Spotlight</div>
                <h3 style="margin:0; color:#0f172a;">Revenue at risk in selected dataset</h3>
                <div style="font-size:2.4rem; font-weight:800; color:#0b3b8f; margin-top:0.4rem;">{revenue_risk_value:,.0f}</div>
                <p style="color:#475569; margin:0.55rem 0 0 0;">
                    Sum of customer value for customers whose churn probability is above 65% in the selected dataset.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="spotlight">
                <div class="hero-kicker" style="color:#0f172a; opacity:1;">Operations Pulse</div>
                <h3 style="margin:0; color:#0f172a;">Customers needing immediate action</h3>
                <div style="font-size:2.4rem; font-weight:800; color:#0d9488; margin-top:0.4rem;">{summary["high_risk"]}</div>
                <p style="color:#475569; margin:0.55rem 0 0 0;">
                    This is not historical workflow count. This is the number of high-risk customers in the selected dataset preview.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"""
            <div class="spotlight">
                <div class="hero-kicker" style="color:#0f172a; opacity:1;">Recovery Potential</div>
                <h3 style="margin:0; color:#0f172a;">Estimated recoverable revenue</h3>
                <div style="font-size:2rem; font-weight:800; color:#0d9488; margin-top:0.4rem;">{summary.get("potential_recovery", 0):,.0f}</div>
                <p style="color:#475569; margin:0.55rem 0 0 0;">Estimated as 60% of revenue at risk. This is a business estimate, not an actual booked recovery amount.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="spotlight">
                <div class="hero-kicker" style="color:#0f172a; opacity:1;">Forecast</div>
                <h3 style="margin:0; color:#0f172a;">Projected next-month churn</h3>
                <div style="font-size:2rem; font-weight:800; color:#ea580c; margin-top:0.4rem;">{churn_forecast_data.get("next_month_churn_pct", 0):.1f}%</div>
                <p style="color:#475569; margin:0.55rem 0 0 0;">In preview mode this is based on the selected dataset's average churn risk, so it changes when the dataset changes.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if not dashboard_data.empty:
        data = dashboard_data.copy()
        tabs = st.tabs(["Executive Overview", "Behavior Analytics", "Retention Pipeline"])

        with tabs[0]:
            col1, col2 = st.columns(2)
            with col1:
                segment_chart = px.bar(
                    data.groupby("Segment", as_index=False).size(),
                    x="Segment",
                    y="size",
                    color="Segment",
                    title="Customer Segments",
                )
                st.plotly_chart(segment_chart, use_container_width=True)
            with col2:
                risk_chart = px.pie(
                    data.groupby("Risk Level", as_index=False).size(),
                    names="Risk Level",
                    values="size",
                    title="Risk Distribution",
                    color="Risk Level",
                    color_discrete_map={"Low": "#22c55e", "Medium": "#f59e0b", "High": "#ef4444", "Critical": "#7f1d1d"},
                )
                st.plotly_chart(risk_chart, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                plan_chart = px.bar(
                    data.groupby("Subscription Type", as_index=False)["Churn Probability"].mean(),
                    x="Subscription Type",
                    y="Churn Probability",
                    color="Subscription Type",
                    title="Average Churn Probability by Plan",
                )
                st.plotly_chart(plan_chart, use_container_width=True)
            with col2:
                gender_chart = px.bar(
                    data.groupby("Gender", as_index=False)["Churn Probability"].mean(),
                    x="Gender",
                    y="Churn Probability",
                    color="Gender",
                    title="Average Churn Probability by Gender",
                )
                st.plotly_chart(gender_chart, use_container_width=True)

            if churn_forecast_data.get("trend"):
                forecast_df = pd.DataFrame(churn_forecast_data["trend"])
                forecast_chart = px.line(
                    forecast_df,
                    x="month",
                    y="churn_pct",
                    markers=True,
                    title="Churn Trend Forecast",
                )
                st.plotly_chart(forecast_chart, use_container_width=True)

        with tabs[1]:
            col1, col2 = st.columns(2)
            with col1:
                engagement_chart = px.scatter(
                    data,
                    x="Engagement Score",
                    y="Churn Probability",
                    color="Issue Category",
                    size="Customer Value",
                    hover_data=["Customer ID", "Subscription Type", "Risk Level"],
                    title="Engagement vs Churn Probability",
                )
                st.plotly_chart(engagement_chart, use_container_width=True)
            with col2:
                issue_chart = px.bar(
                    data.groupby("Issue Category", as_index=False)["Churn Probability"].mean().sort_values("Churn Probability", ascending=False),
                    x="Issue Category",
                    y="Churn Probability",
                    color="Issue Category",
                    title="Issue Category Impact on Churn",
                )
                st.plotly_chart(issue_chart, use_container_width=True)

            heatmap_frame = data[["Age", "Tenure", "Monthly Charges", "Usage Score", "Engagement Score", "Churn Probability"]].corr(numeric_only=True).round(2)
            st.subheader("Correlation View")
            st.dataframe(heatmap_frame, use_container_width=True)

        with tabs[2]:
            st.subheader("High Value / High Risk Customers")
            focus = data.sort_values(["Churn Probability", "Customer Value"], ascending=[False, False]).head(20)
            st.dataframe(focus, use_container_width=True, hide_index=True)

            st.subheader("Action Queue")
            pipeline = data[["Customer ID", "Risk Level", "Issue Category", "Offer", "Action", "Churn Probability"]].sort_values(
                ["Churn Probability"], ascending=False
            )
            st.dataframe(pipeline.head(30), use_container_width=True, hide_index=True)
    else:
        st.warning("Dataset is ready, but no scored customer records were found in the database yet.")


elif page == "Dataset Intelligence":
    st.markdown(
        """
        <div class="hero">
            <h1 style="margin:0;">Dataset Intelligence</h1>
            <p style="margin:0.7rem 0 0 0;">
                Review the active churn dataset, required prediction fields, customer sample records, and data quality profile
                before training or scoring.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    recent_uploads = fetch_uploads(selected_company_id, limit=10, user_email=current_user_email)
    dataset_tabs = st.tabs(["Current Dataset", "Schema Mapping", "Risk & Churn", "NLP & Sentiment", "Customer Sample"])
    profile_df = dataset_profile(seeded_dataset)

    with dataset_tabs[0]:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Current Rows", f"{len(seeded_dataset):,}")
        col2.metric("Columns", len(seeded_dataset.columns))
        col3.metric("Churn Rate", f"{safe_dataset_churn_rate(seeded_dataset):.1f}%")
        col4.metric("Missing Cells", int(seeded_dataset.isna().sum().sum()))

        st.caption(f"Active file: {active_dataset_source}")
        st.caption(f"Selected dataset owner: {current_user_email} | Dataset role: {active_dataset_role}")
        st.subheader("Dataset Profile")
        st.dataframe(profile_df, use_container_width=True, hide_index=True)
        if recent_uploads:
            st.subheader("Your Recent Uploads")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "File": upload.filename,
                            "Uploaded By": upload.uploaded_by_email,
                            "Role": upload.dataset_role,
                            "Rows": upload.row_count,
                            "Model Version": upload.model_version,
                            "Created": upload.created_at,
                        }
                        for upload in recent_uploads
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )
            manager_options = {
                f"{upload.filename} | {upload.dataset_role} | {upload.created_at.strftime('%d %b %Y %H:%M') if upload.created_at else 'Unknown'}": upload.id
                for upload in recent_uploads
            }
            selected_manage_label = st.selectbox("Manage Saved Dataset", list(manager_options.keys()), key="manage_saved_dataset")
            if st.button("Delete Selected Dataset", type="secondary"):
                if delete_dataset_upload(manager_options[selected_manage_label], current_user_email):
                    st.success("Selected dataset deleted from your saved upload list.")
                    st.rerun()
                else:
                    st.error("Dataset delete failed. Only your own saved uploads can be deleted.")

    with dataset_tabs[1]:
        if not recent_uploads:
            st.info("Upload a company dataset to see dynamic schema mapping details here.")
        else:
            mapping = active_dataset_upload.schema_mapping if active_dataset_upload is not None else {}
            st.caption(f"Selected upload: {active_dataset_name}")
            st.subheader("Detected Mapping")
            st.dataframe(
                pd.DataFrame(
                    [{"Canonical Field": key, "Source Column": value} for key, value in (mapping.get("mapped_columns") or {}).items()]
                ),
                use_container_width=True,
                hide_index=True,
            )
            st.subheader("Unmapped / Extra Columns")
            st.write(", ".join(mapping.get("unmapped_columns", [])) or "No extra columns detected.")
            st.subheader("Retained Extra Features")
            st.write(", ".join(mapping.get("extra_feature_columns", [])) or "No retained extra features.")

    with dataset_tabs[2]:
        risk_view = dashboard_data.copy()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("High Risk Customers", int((risk_view.get("Churn Probability", pd.Series(dtype=float)) >= 65).sum()) if not risk_view.empty else 0)
        col2.metric("Critical Customers", int((risk_view.get("Churn Probability", pd.Series(dtype=float)) >= 80).sum()) if not risk_view.empty else 0)
        col3.metric("Avg Churn Probability", f"{risk_view.get('Churn Probability', pd.Series(dtype=float)).mean():.1f}%" if not risk_view.empty else "0.0%")
        col4.metric("Revenue At Risk", f"{risk_view.loc[risk_view['Churn Probability'] >= 65, 'Customer Value'].fillna(0).sum():,.0f}" if not risk_view.empty else "0")
        if not risk_view.empty:
            col1, col2 = st.columns(2)
            with col1:
                risk_dist = px.histogram(risk_view, x="Churn Probability", nbins=25, color="Risk Level", title="Churn Probability Distribution")
                st.plotly_chart(risk_dist, use_container_width=True)
            with col2:
                segment_mix = px.bar(risk_view.groupby("Segment", as_index=False).size(), x="Segment", y="size", color="Segment", title="Segment Mix")
                st.plotly_chart(segment_mix, use_container_width=True)
            st.subheader("Customers Most Likely To Churn")
            st.dataframe(
                risk_view[["Customer ID", "Customer Name", "Subscription Type", "Issue Category", "Churn Probability", "Risk Level", "Action"]]
                .sort_values("Churn Probability", ascending=False)
                .head(30),
                use_container_width=True,
                hide_index=True,
            )

    with dataset_tabs[3]:
        sentiment_df = dashboard_data.copy()
        if sentiment_df.empty:
            st.info("Selected dataset is empty.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                sentiment_summary = pd.DataFrame(nlp_data.get("sentiment_summary", []))
                if not sentiment_summary.empty:
                    st.plotly_chart(
                        px.pie(sentiment_summary, names="sentiment", values="count", title="Sentiment Split"),
                        use_container_width=True,
                    )
            with col2:
                issue_summary = pd.DataFrame(nlp_data.get("issue_summary", []))
                if not issue_summary.empty:
                    st.plotly_chart(
                        px.bar(issue_summary, x="issue", y="count", color="issue", title="Issue Categories Detected From Dataset"),
                        use_container_width=True,
                    )
            st.subheader("Top Dataset Keywords")
            st.write(", ".join(nlp_data.get("keywords", [])) or "No keywords detected from current feedback text.")
            sentiment_preview = sentiment_df[["Customer ID", "Customer Name", "Issue Category", "Sentiment", "Reason"]].head(30)
            st.dataframe(sentiment_preview, use_container_width=True, hide_index=True)

    with dataset_tabs[4]:
        show_cols = [
            "Customer_ID",
            "Customer_Name",
            "Email",
            "Gender",
            "Age",
            "Tenure",
            "Subscription_Type",
            "Monthly_Charges",
            "Usage_Score",
            "Last_Interaction_Days",
            "Issue_Category",
            "Sentiment",
            "Churn",
        ]
        available_cols = [column for column in show_cols if column in seeded_dataset.columns]
        st.dataframe(seeded_dataset[available_cols].head(50), use_container_width=True, hide_index=True)

    st.subheader("Required Inputs Reference")
    required_inputs = pd.DataFrame(
            [
                {"Field": "Customer_ID", "Required": "Yes", "Why It Matters": "Unique customer tracking in scoring and action workflows"},
                {"Field": "Customer_Name", "Required": "Recommended", "Why It Matters": "Used by Action Center and personalized emails"},
                {"Field": "Gender", "Required": "Yes", "Why It Matters": "Demographic churn trend analysis"},
                {"Field": "Age", "Required": "Yes", "Why It Matters": "Lifecycle behavior segmentation"},
                {"Field": "Tenure", "Required": "Yes", "Why It Matters": "New vs loyal customer risk separation"},
                {"Field": "Subscription_Type", "Required": "Yes", "Why It Matters": "Plan-wise churn and pricing analysis"},
                {"Field": "Monthly_Charges", "Required": "Yes", "Why It Matters": "Revenue risk and pricing sensitivity"},
                {"Field": "Total_Spend", "Required": "Engineered", "Why It Matters": "Customer lifetime value proxy"},
                {"Field": "Usage_Score", "Required": "Yes", "Why It Matters": "Low engagement is a major churn signal"},
                {"Field": "Last_Interaction_Days", "Required": "Yes", "Why It Matters": "Recency drives retention urgency"},
                {"Field": "Feedback", "Required": "Yes", "Why It Matters": "NLP sentiment and issue classification"},
                {"Field": "Sentiment", "Required": "Derived", "Why It Matters": "Positive/negative intent signal"},
                {"Field": "Issue_Category", "Required": "Derived", "Why It Matters": "Pricing, service, or network root-cause tracking"},
                {"Field": "Churn", "Required": "Training only", "Why It Matters": "Supervised learning target"},
            ]
        )
    st.dataframe(required_inputs, use_container_width=True, hide_index=True)


elif page == "Customer Insights":
    st.markdown(
        """
        <div class="hero">
            <h1 style="margin:0;">Customer Insights</h1>
            <p style="margin:0.7rem 0 0 0;">
                Explore filterable customer risk, churn reason, recommended action, and customer timeline in one BI-style workspace.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    insights = dashboard_data.copy()
    focused_customer_id = st.session_state.get("focused_customer_id")
    history_focus = st.session_state.get("prediction_history", {})
    if focused_customer_id:
        matching_history = [
            item for item in history_focus.values() if str(item.get("customer_id")) == str(focused_customer_id)
        ]
        if matching_history:
            st.success(f"Focused customer loaded from Customer Predictor: {focused_customer_id}")
            st.dataframe(
                pd.DataFrame(matching_history).rename(
                    columns={
                        "customer_id": "Customer ID",
                        "risk_level": "Risk Level",
                        "churn_probability": "Churn Probability",
                        "issue_category": "Issue Category",
                        "offer": "Offer",
                        "action": "Next Action",
                        "predicted_at": "Predicted At",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
    col1, col2, col3 = st.columns(3)
    selected_plan = col1.selectbox("Plan Filter", ["All"] + sorted(insights["Subscription Type"].dropna().astype(str).unique().tolist()))
    selected_risk = col2.selectbox("Risk Filter", ["All", "Low", "Medium", "High", "Critical"])
    max_age = int(insights["Age"].fillna(0).max()) if not insights.empty else 80
    age_range = col3.slider("Age Range", 18, max(18, max_age), (18, max(18, max_age)))
    if selected_plan != "All":
        insights = insights[insights["Subscription Type"].astype(str) == selected_plan]
    if selected_risk != "All":
        insights = insights[insights["Risk Level"] == selected_risk]
    insights = insights[(insights["Age"].fillna(0) >= age_range[0]) & (insights["Age"].fillna(0) <= age_range[1])]

    st.dataframe(insights, use_container_width=True, hide_index=True)

    payload_df = seeded_dataset.copy()
    if not payload_df.empty:
        st.subheader("Adaptive Dataset Charts")
        categorical_candidates = safe_categorical_candidates(payload_df)
        numeric_candidates = [column for column in payload_df.columns if pd.api.types.is_numeric_dtype(payload_df[column])]
        if categorical_candidates and numeric_candidates:
            col1, col2 = st.columns(2)
            selected_category = col1.selectbox("Group Field", categorical_candidates, key="adaptive_category")
            selected_metric = col2.selectbox("Numeric Metric", numeric_candidates, key="adaptive_metric")
            adaptive_chart_df = payload_df.groupby(selected_category, as_index=False)[selected_metric].mean().sort_values(selected_metric, ascending=False)
            st.plotly_chart(
                px.bar(adaptive_chart_df, x=selected_category, y=selected_metric, color=selected_category, title="Adaptive chart from selected dataset"),
                use_container_width=True,
            )

    if not insights.empty:
        st.subheader("Individual Customer Drilldown")
        customer_labels = insights.apply(
            lambda row: f"{row.get('Customer ID', 'Unknown')} | {row.get('Customer Name', 'Unknown')} | {row.get('Risk Level', 'Unknown')}",
            axis=1,
        ).tolist()
        selected_customer_label = st.selectbox("Choose Customer", customer_labels, key="insight_customer_pick")
        selected_customer_id = selected_customer_label.split("|", 1)[0].strip()
        selected_customer_view = insights[insights["Customer ID"].astype(str) == selected_customer_id].head(1)
        if not selected_customer_view.empty:
            st.dataframe(selected_customer_view.T.rename(columns={selected_customer_view.index[0]: "Value"}), use_container_width=True)


elif page == "Training Lab":
    st.markdown(
        """
        <div class="hero">
            <h1 style="margin:0;">Training Lab</h1>
            <p style="margin:0.7rem 0 0 0;">
                Upload a labeled churn dataset to train a stronger churn model with automatic preprocessing,
                leakage-safe feature selection, cross-validation, validation-based threshold tuning, and holdout testing.
                Large datasets now automatically use a faster training profile so results come sooner.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("train_form"):
        company_name = st.text_input("Company Name", value=selected_company_name if selected_company_id else "Enterprise Telecom")
        industry = st.selectbox("Industry", ["Telecom", "Subscription", "SaaS", "Banking", "Insurance", "Retail"])
        training_file = st.file_uploader("Training Dataset", type=["csv", "xlsx", "xls"])
        auto_score_after_training = st.checkbox("After training, also score this same dataset and save it for dashboard analysis", value=True)
        submitted = st.form_submit_button("Train Platform Models")

    if submitted:
        if not training_file:
            st.error("Please upload a labeled churn dataset.")
        else:
            session = get_session()
            try:
                raw_training_df = load_dataset_from_upload(training_file)
                with st.spinner("Training models with adaptive fast mode and selecting the best profile..."):
                    result = train_models_from_dataframe(
                        session,
                        company_name,
                        industry,
                        raw_training_df,
                        source_name=getattr(training_file, "name", "training_dataset.csv"),
                        uploaded_by_email=current_user_email,
                    )
                st.success(f"Best model: {result['best_model']}")
                st.caption(f"Model version: {result.get('model_version', 'n/a')}")
                evaluation_summary = result.get("evaluation_summary", {})
                if evaluation_summary:
                    st.info(
                        "Training profile: "
                        f"{evaluation_summary.get('training_profile', 'balanced')} | "
                        f"Rows used for model selection: {evaluation_summary.get('model_selection_rows', 0):,} | "
                        f"CV folds: {evaluation_summary.get('cv_folds', 0)}"
                    )
                st.success("Training dataset has been saved and will appear in the dataset picker for this logged-in user.")
                metrics_df = pd.DataFrame(result["metrics"]).T.reset_index().rename(columns={"index": "Model"})
                st.dataframe(metrics_df, use_container_width=True, hide_index=True)
                best_summary = result["metrics"].get(result["best_model"], {}).get("test_summary", {})
                if best_summary:
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Holdout F1", best_summary.get("f1_score", 0))
                    col2.metric("ROC-AUC", best_summary.get("roc_auc", 0))
                    col3.metric("Recall", best_summary.get("recall", 0))
                    col4.metric("Decision Threshold", best_summary.get("threshold", 0.55))
                mapping = result.get("schema_mapping", {})
                if mapping:
                    st.subheader("Detected Schema Mapping")
                    st.dataframe(
                        pd.DataFrame(
                            [{"Canonical Field": key, "Source Column": value} for key, value in (mapping.get("mapped_columns") or {}).items()]
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )
                    st.caption(
                        "Retained extra features: "
                        + (", ".join(mapping.get("extra_feature_columns", [])) or "None")
                    )
                st.subheader("Training Preview")
                st.dataframe(pd.DataFrame(result["preview"]), use_container_width=True, hide_index=True)
                if auto_score_after_training:
                    with st.spinner("Scoring the same in-memory dataset for immediate dashboard analysis..."):
                        score_result = score_dataframe_and_store(
                            session,
                            result["company_id"],
                            raw_training_df,
                            source_name=getattr(training_file, "name", "trained_dataset.csv"),
                            uploaded_by_email=current_user_email,
                        )
                    st.success(
                        f"Same dataset scored and saved for dashboard use: {score_result['rows']} customer rows analyzed."
                    )
            except Exception as exc:
                st.error(str(exc))
            finally:
                session.close()


elif page == "Batch Scoring":
    st.markdown(
        """
        <div class="hero">
            <h1 style="margin:0;">Batch Scoring and Segmentation</h1>
            <p style="margin:0.7rem 0 0 0;">
                Upload customer datasets to validate columns, preprocess automatically, predict churn,
                assign KMeans segments, generate SHAP explanations, and store everything in MySQL-ready tables.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not selected_company_id:
        st.warning("Select a company workspace from the sidebar before scoring.")
    else:
        scoring_file = st.file_uploader("Customer Dataset for Scoring", type=["csv", "xlsx", "xls"])
        if st.button("Score Dataset and Store Results", disabled=scoring_file is None):
            session = get_session()
            try:
                with st.spinner("Scoring customers and creating retention actions..."):
                    result = score_dataset_and_store(session, selected_company_id, scoring_file, uploaded_by_email=current_user_email)
                st.success(f"Stored {result['rows']} scored customer records for {result['company']}.")
                st.caption(f"Model version used: {result.get('model_version', 'n/a')}")
                preview = pd.DataFrame(result["results"])
                st.dataframe(preview[["customer_id", "churn_probability", "segment", "customer_value"]], use_container_width=True, hide_index=True)
                st.subheader("Offers and Retention Strategies")
                offers = preview[["customer_id", "retention_strategy", "offer"]].copy()
                st.dataframe(offers, use_container_width=True, hide_index=True)
                mapping = result.get("schema_mapping", {})
                if mapping:
                    st.subheader("Scoring Dataset Mapping")
                    st.dataframe(
                        pd.DataFrame(
                            [{"Canonical Field": key, "Source Column": value} for key, value in (mapping.get("mapped_columns") or {}).items()]
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )
            except Exception as exc:
                st.error(str(exc))
            finally:
                session.close()


elif page == "Customer Predictor":
    st.markdown(
        """
        <div class="hero">
            <h1 style="margin:0;">Real-Time Predictor</h1>
            <p style="margin:0.7rem 0 0 0;">
                Send one customer profile through the trained enterprise pipeline and get churn probability,
                customer segment, SHAP explanation, strategy, and personalized offer.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    artifacts = get_cached_artifacts()
    if artifacts is None:
        st.info("Fast prediction mode is active. Train a dataset in Training Lab to switch to model-scored predictions.")

    with st.container():
        predictor_dataset = seeded_dataset.copy()
        sample_options = demo_customer_options(predictor_dataset)
        st.caption(
            f"Selected dataset for prediction: `{active_dataset_name}` | Source: `{active_dataset_role}` | Available customers: {len(predictor_dataset):,}. "
            "Use the `Analysis Dataset` selector in the sidebar to change the active dataset."
        )
        history_df = prediction_history_frame(active_dataset_name)
        if not history_df.empty:
            st.subheader("Already Predicted Customers")
            history_view = history_df[
                ["Customer ID", "Risk Level", "Churn Probability", "Issue Category", "Offer", "Next Action", "Predicted At"]
            ].sort_values("Predicted At", ascending=False)
            st.dataframe(history_view, use_container_width=True, hide_index=True)
            st.caption("Use this table to avoid predicting the same customer repeatedly during the current session.")
        selected_sample_label = st.selectbox(
            "Select Customer From Chosen Dataset",
            ["Manual Entry"] + sample_options,
            help="Pick any customer from the chosen dataset. The full form below will auto-fill from that exact dataset row.",
        )
        selected_sample_row = None
        if selected_sample_label != "Manual Entry":
            selected_customer_id = selected_sample_label.split("|", 1)[0].strip()
            match = predictor_dataset[predictor_dataset["Customer_ID"].astype(str) == selected_customer_id]
            if not match.empty:
                selected_sample_row = match.iloc[0].to_dict()

        if selected_sample_row:
            existing_key = prediction_storage_key(active_dataset_name, str(selected_sample_row.get("Customer_ID", "")))
            existing_prediction = st.session_state.get("prediction_history", {}).get(existing_key)
            if existing_prediction:
                st.success(
                    f"This customer was already predicted at {existing_prediction['predicted_at']}. "
                    f"Last result: {existing_prediction['risk_level']} risk, "
                    f"{existing_prediction['churn_probability'] * 100:.1f}% churn probability."
                )
            st.subheader("Selected Customer Dataset Row")
            selected_snapshot = pd.DataFrame(
                [{"Field": key, "Value": value} for key, value in selected_sample_row.items()]
            )
            st.dataframe(selected_snapshot, use_container_width=True, hide_index=True)
        else:
            st.info("Manual Entry selected. Choose any customer from the current dataset to auto-load all available values into the prediction form.")

        with st.form("predictor"):
            col1, col2 = st.columns(2)
            customer_id = col1.text_input("Customer ID", str((selected_sample_row or {}).get("Customer_ID", "CUS-1001")))
            email = col2.text_input("Email", str((selected_sample_row or {}).get("Email", "customer@example.com")))
            col1, col2, col3 = st.columns(3)
            age = col1.number_input("Age", min_value=18, max_value=90, value=int((selected_sample_row or {}).get("Age", 34)))
            tenure = col2.number_input("Tenure", min_value=1, max_value=120, value=int((selected_sample_row or {}).get("Tenure", 12)))
            monthly_charges = col3.number_input("Monthly Charges", min_value=100.0, max_value=5000.0, value=float((selected_sample_row or {}).get("Monthly_Charges", 899.0)))
            col1, col2, col3 = st.columns(3)
            total_spend = col1.number_input("Total Spend", min_value=0.0, max_value=500000.0, value=float((selected_sample_row or {}).get("Total_Spend", 15000.0)))
            last_interaction_days = col2.number_input("Last Interaction Days", min_value=0, max_value=365, value=int((selected_sample_row or {}).get("Last_Interaction_Days", 21)))
            usage_score = col3.number_input("Usage Score", min_value=0.0, max_value=100.0, value=float((selected_sample_row or {}).get("Usage_Score", 35.0)))
            col1, col2, col3 = st.columns(3)
            subscription_choices = ["Basic", "Standard", "Premium", "Enterprise"]
            contract_choices = ["Monthly", "Quarterly", "Annual"]
            payment_choices = ["Card", "UPI", "NetBanking", "AutoPay"]
            subscription_default = str((selected_sample_row or {}).get("Subscription_Type", "Basic"))
            contract_default = str((selected_sample_row or {}).get("Contract_Type", "Monthly"))
            payment_default = str((selected_sample_row or {}).get("Payment_Mode", "Card"))
            subscription_type = col1.selectbox("Subscription Type", subscription_choices, index=max(0, subscription_choices.index(subscription_default) if subscription_default in subscription_choices else 0))
            contract_type = col2.selectbox("Contract Type", contract_choices, index=max(0, contract_choices.index(contract_default) if contract_default in contract_choices else 0))
            payment_mode = col3.selectbox("Payment Mode", payment_choices, index=max(0, payment_choices.index(payment_default) if payment_default in payment_choices else 0))
            col1, col2, col3 = st.columns(3)
            gender_choices = ["Male", "Female"]
            gender_default = str((selected_sample_row or {}).get("Gender", "Male"))
            gender = col1.selectbox("Gender", gender_choices, index=max(0, gender_choices.index(gender_default) if gender_default in gender_choices else 0))
            support_tickets = col2.number_input("Support Tickets", min_value=0, max_value=20, value=int((selected_sample_row or {}).get("Support_Tickets", 2)))
            payment_delay_days = col3.number_input("Payment Delay Days", min_value=0, max_value=90, value=int((selected_sample_row or {}).get("Payment_Delay_Days", 4)))
            feedback = st.text_area("Feedback", str((selected_sample_row or {}).get("Feedback", "Service quality has dropped and billing feels expensive.")))
            submitted = st.form_submit_button("Predict and Explain")

        last_prediction_package = st.session_state.get("last_prediction_package")
        if submitted or (last_prediction_package and last_prediction_package.get("dataset") == active_dataset_name):
            if submitted:
                record = {
                    "Customer_ID": customer_id,
                    "Email": email,
                    "Age": age,
                    "Tenure": tenure,
                    "Monthly_Charges": monthly_charges,
                    "Total_Spend": total_spend,
                    "Last_Interaction_Days": last_interaction_days,
                    "Usage_Score": usage_score,
                    "Subscription_Type": subscription_type,
                    "Contract_Type": contract_type,
                    "Payment_Mode": payment_mode,
                    "Gender": gender,
                    "Support_Tickets": support_tickets,
                    "Payment_Delay_Days": payment_delay_days,
                    "Feedback": feedback,
                }
                prepared = prediction_input_frame(record)
                results = predict_records(None, prepared.to_dict(orient="records"))
                result = results[0]
                save_prediction_history(active_dataset_name, selected_sample_label, prepared, result)
                st.session_state["last_prediction_package"] = {
                    "dataset": active_dataset_name,
                    "selected_label": selected_sample_label,
                    "prepared": prepared.to_dict(orient="records"),
                    "result": result,
                }
            else:
                package = st.session_state["last_prediction_package"]
                selected_sample_label = package.get("selected_label", "Manual Entry")
                prepared = pd.DataFrame(package["prepared"])
                result = package["result"]
                customer_id = str(prepared.iloc[0]["Customer_ID"])
            explanation = explain_prediction(prepared, result)
            risk_level = risk_level_from_probability(result["churn_probability"])
            priority_label = result.get("retention_strategy", {}).get("priority", "Medium")
            action_text = result.get("retention_strategy", {}).get("action", "Review customer manually.")
            strategy_name = result.get("retention_strategy", {}).get("strategy", "Review")
            offer_name = result.get("personalized_offer", {}).get("offer_name", "Retention Offer")
            offer_details = result.get("personalized_offer", {}).get("offer_details", "Manual review recommended.")
            next_steps = [
                f"1. {action_text}",
                f"2. Offer this customer: {offer_name}",
                "3. Track response within 72 hours and re-score if behavior changes.",
            ]
            if result["churn_probability"] >= 0.75:
                executive_decision = "Immediate intervention needed. This customer is highly likely to churn unless the retention team acts quickly."
            elif result["churn_probability"] >= 0.55:
                executive_decision = "Proactive retention action is recommended now before this customer moves into the critical churn band."
            else:
                executive_decision = "Customer is currently stable, but guided engagement should continue to keep churn risk low."

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Churn Probability", f"{result['churn_probability'] * 100:.1f}%")
            col2.metric("Prediction", "Churn" if result["churn_prediction"] else "Retain")
            col3.metric("Risk Level", risk_level)
            col4.metric("Customer Value", round(result["customer_value"], 2))
            col1, col2, col3 = st.columns(3)
            col1.metric("Model Score", f"{result['raw_model_probability'] * 100:.1f}%")
            col2.metric("Hybrid Risk Score", f"{result['heuristic_probability'] * 100:.1f}%")
            col3.metric("Sentiment", f"{result['sentiment_label']} ({result['sentiment_score']:+.2f})")
            st.caption(f"Issue category detected from feedback: {prepared.iloc[0]['Issue_Category']}")
            if selected_sample_label != "Manual Entry":
                st.info(
                    f"This prediction is being run on the exact selected dataset row `{selected_sample_label}` from `{active_dataset_name}`."
                )

            st.markdown(
                f"""
                <div class="decision-banner">
                    <div class="decision-title">Main Decision</div>
                    <div class="decision-text">{executive_decision}</div>
                    <p style="margin:0.8rem 0 0 0; opacity:0.92;">
                        Recommended strategy: <strong>{strategy_name}</strong> | Priority: <strong>{priority_label}</strong> | Best offer: <strong>{offer_name}</strong>
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            decision_summary_df = pd.DataFrame(
                [
                    {"Metric": "Customer ID", "Value": prepared.iloc[0]["Customer_ID"]},
                    {"Metric": "Decision", "Value": "Likely to Churn" if result["churn_prediction"] else "Likely to Stay"},
                    {"Metric": "Risk Level", "Value": risk_level},
                    {"Metric": "Churn Probability", "Value": f"{result['churn_probability'] * 100:.1f}%"},
                    {"Metric": "Recommended Strategy", "Value": strategy_name},
                    {"Metric": "Priority", "Value": priority_label},
                    {"Metric": "Best Offer", "Value": offer_name},
                    {"Metric": "Issue Category", "Value": prepared.iloc[0]["Issue_Category"]},
                ]
            )
            st.subheader("Prediction Summary")
            st.dataframe(decision_summary_df, use_container_width=True, hide_index=True)

            profile_col1, profile_col2 = st.columns([1.1, 1])
            with profile_col1:
                st.subheader("Customer Summary")
                customer_summary_df = pd.DataFrame(
                    [
                        {"Field": "Customer ID", "Value": prepared.iloc[0]["Customer_ID"]},
                        {"Field": "Email", "Value": prepared.iloc[0]["Email"]},
                        {"Field": "Age", "Value": prepared.iloc[0]["Age"]},
                        {"Field": "Tenure", "Value": prepared.iloc[0]["Tenure"]},
                        {"Field": "Subscription", "Value": prepared.iloc[0]["Subscription_Type"]},
                        {"Field": "Contract", "Value": prepared.iloc[0]["Contract_Type"]},
                        {"Field": "Payment", "Value": prepared.iloc[0]["Payment_Mode"]},
                        {"Field": "Issue Category", "Value": prepared.iloc[0]["Issue_Category"]},
                    ]
                )
                st.dataframe(customer_summary_df, use_container_width=True, hide_index=True)
            with profile_col2:
                st.subheader("What To Do Now")
                action_plan_df = pd.DataFrame(
                    [
                        {"Step": "1", "Action": action_text, "Owner": "Retention Team", "Timing": "Today"},
                        {"Step": "2", "Action": f"Use offer: {offer_name}", "Owner": "Customer Success", "Timing": "During outreach"},
                        {"Step": "3", "Action": "Track response and re-score if behavior changes.", "Owner": "Retention Analyst", "Timing": "Within 72 hours"},
                    ]
                )
                st.markdown(
                    f"""
                    <div class="action-box">
                        <h4>{strategy_name}</h4>
                        <p><strong>Why:</strong> {explanation}</p>
                        <p><strong>Offer:</strong> {offer_details}</p>
                        <p><strong>Immediate action:</strong> {action_text}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.dataframe(action_plan_df, use_container_width=True, hide_index=True)

            st.subheader("Offer Options For This Customer")
            offer_options_df = build_offer_options(prepared, result)
            st.dataframe(offer_options_df, use_container_width=True, hide_index=True)
            selected_offer = st.selectbox(
                "Select offer to use",
                offer_options_df["Offer"].tolist(),
                index=0,
                key=f"selected_offer_{prepared.iloc[0]['Customer_ID']}",
            )

            st.subheader("Customer Contact And Action Options")
            contact_plan_df = build_contact_plan(prepared, result)
            st.dataframe(contact_plan_df, use_container_width=True, hide_index=True)
            selected_customer_key = str(prepared.iloc[0]["Customer_ID"])
            action_state = st.session_state.get("retention_action_statuses", {}).get(selected_customer_key, {})
            status_df = pd.DataFrame(
                [
                    {"Action": "Email", "Status": action_state.get("email", "pending").title()},
                    {"Action": "Phone Call", "Status": action_state.get("call", "pending").title()},
                    {"Action": "SMS / WhatsApp", "Status": action_state.get("message", "pending").title()},
                    {"Action": "Offer", "Status": action_state.get("offer", "pending").title()},
                    {"Action": "Owner Assignment", "Status": action_state.get("assign", "pending").title()},
                ]
            )
            st.dataframe(status_df, use_container_width=True, hide_index=True)
            contact_col1, contact_col2, contact_col3, contact_col4 = st.columns(4)
            if contact_col1.button("Prepare Email", key=f"email_{selected_customer_key}", use_container_width=True):
                mark_action_status(selected_customer_key, "email", "prepared")
                st.success(f"Email action prepared for {selected_customer_key} with offer: {selected_offer}.")
            if contact_col2.button("Log Phone Call", key=f"call_{selected_customer_key}", use_container_width=True):
                mark_action_status(selected_customer_key, "call")
                st.success(f"Phone call logged for {selected_customer_key}.")
            if contact_col3.button("Prepare Message", key=f"message_{selected_customer_key}", use_container_width=True):
                mark_action_status(selected_customer_key, "message", "prepared")
                st.success(f"SMS / WhatsApp message prepared for {selected_customer_key}.")
            if contact_col4.button("Apply Offer", key=f"offer_{selected_customer_key}", use_container_width=True):
                mark_action_status(selected_customer_key, "offer")
                st.success(f"Selected offer applied in the action plan: {selected_offer}.")

            st.markdown(
                f"""
                <div class="spotlight">
                    <div class="hero-kicker" style="color:#0f172a; opacity:1;">AI Explanation</div>
                    <h3 style="margin:0; color:#0f172a;">Why this customer may churn</h3>
                    <p style="color:#475569; margin:0.65rem 0 0 0;">
                        The explanation below combines model output, customer behavior, feedback issue, payment behavior,
                        and account value. It is designed for retention action, not only prediction.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.subheader("Risk Signal Explanation")
            st.dataframe(prediction_signal_frame(prepared, result), use_container_width=True, hide_index=True)
            st.subheader("Prediction Policy")
            st.dataframe(prediction_policy_frame(prepared, result), use_container_width=True, hide_index=True)

            st.markdown(
                f"""
                <div class="offer">
                    <h4 style="margin-top:0;">{result['personalized_offer']['offer_name']}</h4>
                    <p>{result['personalized_offer']['offer_details']}</p>
                    <strong>Strategy:</strong> {result['retention_strategy']['strategy']}<br>
                    <strong>Action:</strong> {result['retention_strategy']['action']}
                </div>
                """,
                unsafe_allow_html=True,
            )

            recommendation_df = pd.DataFrame(result.get("recommendation_plan", []))
            reason_df = pd.DataFrame(result["shap_explanation"]).rename(
                columns={"feature": "Driver", "contribution": "Impact", "direction": "Effect"}
            )
            if not reason_df.empty:
                st.subheader("Why This Customer Is At Risk")
                st.dataframe(reason_df, use_container_width=True, hide_index=True)
            if not recommendation_df.empty:
                st.subheader("Retention Plan Table")
                st.dataframe(recommendation_df, use_container_width=True, hide_index=True)

            jump_col1, jump_col2 = st.columns(2)
            with jump_col1:
                if st.button("Open Action Center For This Customer", use_container_width=True):
                    st.session_state["focused_customer_id"] = str(prepared.iloc[0]["Customer_ID"])
                    st.session_state["pending_module"] = "Action Center"
                    st.rerun()
            with jump_col2:
                if st.button("Open Customer Insights", use_container_width=True):
                    st.session_state["focused_customer_id"] = str(prepared.iloc[0]["Customer_ID"])
                    st.session_state["pending_module"] = "Customer Insights"
                    st.rerun()

            chart_col1, chart_col2 = st.columns(2)
            with chart_col1:
                driver_df = prediction_driver_frame(prepared, result)
                st.subheader("Risk Driver Chart")
                driver_chart = px.bar(
                    driver_df,
                    x="Impact",
                    y="Driver",
                    color="Direction",
                    orientation="h",
                    title="Why the customer may churn",
                )
                st.plotly_chart(driver_chart, use_container_width=True)
            with chart_col2:
                health_df = customer_health_frame(prepared, result)
                st.subheader("Customer Health Snapshot")
                health_chart = px.line_polar(
                    health_df,
                    r="Value",
                    theta="Metric",
                    line_close=True,
                    range_r=[0, 100],
                    title="Customer stability vs churn risk",
                )
                st.plotly_chart(health_chart, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Detailed Risk Factors")
                st.dataframe(pd.DataFrame(result["shap_explanation"]), use_container_width=True, hide_index=True)
            with col2:
                st.subheader("Full Customer Input Table")
                st.dataframe(prepared.T.rename(columns={0: "Value"}), use_container_width=True)

            st.subheader("What-If Analysis")
            scenario_col1, scenario_col2 = st.columns(2)
            price_cut = scenario_col1.slider("If price is reduced by %", min_value=0, max_value=30, value=10)
            improve_support = scenario_col2.checkbox("If support experience improves", value=True)
            scenario_result = simulate_business_impact(prepared.iloc[0].to_dict(), price_cut, improve_support)
            before_df = pd.DataFrame(
                [
                    {"Scenario": "Before", "Churn Probability": scenario_result["before"]["churn_probability"] * 100},
                    {"Scenario": "After", "Churn Probability": scenario_result["after"]["churn_probability"] * 100},
                ]
            )
            scenario_chart = px.bar(before_df, x="Scenario", y="Churn Probability", color="Scenario", title="Before vs After Simulation")
            st.plotly_chart(scenario_chart, use_container_width=True)
            st.caption(f"Probability change: {scenario_result['change_in_probability']:.2f} percentage points")

            st.subheader("AI Assistant")
            st.write(result.get("assistant_default_answer", "Ask anything about this customer profile."))
            assistant_question = st.text_input(
                "Ask about this customer",
                value="What should we do to stop this customer from churning?",
                key=f"assistant_question_{customer_id}",
            )
            if assistant_question:
                assistant_answer = predictor_assistant_response(assistant_question, prepared.iloc[0], result)
                st.markdown(
                    f"""
                    <div class="soft">
                        <strong>Question:</strong> {assistant_question}<br><br>
                        <strong>AI Assistant:</strong> {assistant_answer}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


elif page == "Action Center":
    st.markdown(
        """
        <div class="hero">
            <h1 style="margin:0;">Action Center</h1>
            <p style="margin:0.7rem 0 0 0;">
                Execute retention actions directly for high-risk customers and track pending vs completed interventions.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    action_df = pd.DataFrame(action_center_data)
    history_for_actions = list(st.session_state.get("prediction_history", {}).values())
    if history_for_actions:
        history_action_df = pd.DataFrame(
            [
                {
                    "customer_id": item["customer_id"],
                    "customer_name": item.get("customer_name") or item["customer_id"],
                    "churn_probability": item["churn_probability"],
                    "risk_level": item["risk_level"],
                    "reason": f"{item['issue_category']} detected. Recommended strategy: {item['strategy']}.",
                    "recommended_actions": [item["action"]],
                    "offer": item["offer"],
                    "status": "pending",
                    "assigned_agent": "Retention Team",
                    "action_statuses": st.session_state.get("retention_action_statuses", {}).get(item["customer_id"], {}),
                }
                for item in history_for_actions
            ]
        )
        action_df = pd.concat([history_action_df, action_df], ignore_index=True)
        action_df = action_df.drop_duplicates(subset=["customer_id"], keep="first")
    if action_df.empty:
        st.info("No high-risk customers are currently available for action.")
    else:
        st.caption(f"Preview mode is active. Action recommendations below are generated from the selected dataset `{active_demo_dataset_name}`.")
        focused_customer_id = st.session_state.get("focused_customer_id")
        if focused_customer_id and focused_customer_id in action_df["customer_id"].astype(str).tolist():
            st.success(f"Focused customer loaded from Customer Predictor: {focused_customer_id}")
        display_df = action_df.copy()
        display_df["action_statuses"] = display_df.apply(
            lambda row: {
                **(row.get("action_statuses") or {}),
                **st.session_state.get("retention_action_statuses", {}).get(str(row["customer_id"]), {}),
            },
            axis=1,
        )
        display_df["Churn Probability"] = (display_df["churn_probability"] * 100).round(2).astype(str) + "%"
        display_df["Call Status"] = display_df["action_statuses"].apply(lambda item: "Completed" if item.get("call") == "completed" else "Pending")
        display_df["Email Status"] = display_df["action_statuses"].apply(
            lambda item: "Completed" if item.get("email") == "completed" else "Prepared" if item.get("email") == "prepared" else "Pending"
        )
        display_df["Offer Status"] = display_df["action_statuses"].apply(lambda item: item.get("offer", "pending").title())
        display_df["Assign Status"] = display_df["action_statuses"].apply(lambda item: item.get("assign", "pending").title())
        st.dataframe(
            display_df[
                [
                    "customer_id",
                    "customer_name",
                    "Churn Probability",
                    "risk_level",
                    "reason",
                    "Call Status",
                    "Email Status",
                    "Offer Status",
                    "Assign Status",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )
        customer_options = action_df["customer_id"].astype(str).tolist()
        selected_index = customer_options.index(str(focused_customer_id)) if focused_customer_id in customer_options else 0
        selected_customer = st.selectbox("Select customer for action", customer_options, index=selected_index)
        selected_row = action_df[action_df["customer_id"].astype(str) == selected_customer].iloc[0]
        selected_statuses = {
            **(selected_row.get("action_statuses") or {}),
            **st.session_state.get("retention_action_statuses", {}).get(str(selected_customer), {}),
        }
        st.markdown(
            f"""
            <div class="spotlight">
                <div class="hero-kicker" style="color:#0f172a; opacity:1;">Retention Context</div>
                <h3 style="margin:0; color:#0f172a;">{selected_row['customer_name']} ({selected_row['customer_id']}) | {selected_row['risk_level']} Risk</h3>
                <p style="color:#475569; margin:0.65rem 0 0 0;">{selected_row['reason']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(
            f"Current statuses: Call={selected_statuses.get('call', 'pending')} | "
            f"Email={selected_statuses.get('email', 'pending')} | "
            f"Message={selected_statuses.get('message', 'pending')} | "
            f"Offer={selected_statuses.get('offer', 'pending')} | "
            f"Assign={selected_statuses.get('assign', 'pending')}"
        )
        assigned_agent = st.text_input("Assign support agent", value=selected_row.get("assigned_agent") or "agent@retentionos.local")
        offer_choices = [
            str(selected_row.get("offer", "Retention Offer")),
            "Price Protection Plan",
            "Usage Booster Pack",
            "Priority Retention Save",
            "Loyalty Reward",
        ]
        action_offer = st.selectbox("Offer to send", list(dict.fromkeys(offer_choices)), key=f"action_offer_{selected_customer}")
        action_col1, action_col2, action_col3, action_col4, action_col5 = st.columns(5)
        if action_col1.button("Call Customer", key=f"action_call_{selected_customer}", use_container_width=True):
            mark_action_status(str(selected_customer), "call")
            st.success(f"Phone call logged for {selected_row['customer_name']}.")
        if action_col2.button("Prepare Email", key=f"action_email_{selected_customer}", use_container_width=True):
            mark_action_status(str(selected_customer), "email", "prepared")
            st.success(f"Email prepared for {selected_row['customer_name']} with offer: {action_offer}.")
        if action_col3.button("Prepare Message", key=f"action_message_{selected_customer}", use_container_width=True):
            mark_action_status(str(selected_customer), "message", "prepared")
            st.success(f"SMS / WhatsApp message prepared for {selected_row['customer_name']}.")
        if action_col4.button("Apply Offer", key=f"action_offer_btn_{selected_customer}", use_container_width=True):
            mark_action_status(str(selected_customer), "offer")
            st.success(f"Offer applied in the action plan: {action_offer}.")
        if action_col5.button("Assign Agent", key=f"action_assign_{selected_customer}", use_container_width=True):
            mark_action_status(str(selected_customer), "assign")
            st.success(f"Assigned owner: {assigned_agent}.")


elif page == "NLP Insights":
    st.markdown(
        """
        <div class="hero">
            <h1 style="margin:0;">NLP Insights</h1>
            <p style="margin:0.7rem 0 0 0;">
                Track issue themes, customer sentiment, extracted keywords, and complaint mix from customer feedback.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    issue_df = pd.DataFrame(nlp_data.get("issue_summary", []))
    sentiment_df = pd.DataFrame(nlp_data.get("sentiment_summary", []))
    keyword_df = pd.DataFrame(nlp_data.get("keyword_summary", []))
    issue_sentiment_df = pd.DataFrame(nlp_data.get("issue_sentiment_summary", []))
    recent_feedback_df = pd.DataFrame(nlp_data.get("recent_feedback", []))

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Feedback Records", int(nlp_data.get("feedback_count", 0)))
    metric_col2.metric("Average Sentiment", f"{float(nlp_data.get('avg_sentiment_score', 0)):+.3f}")
    metric_col3.metric("Tracked Keywords", len(nlp_data.get("keywords", [])))

    if issue_df.empty and sentiment_df.empty and recent_feedback_df.empty:
        st.info("No customer feedback has been captured yet. Score or store customer records to generate NLP insights.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            if not issue_df.empty:
                issue_chart = px.bar(
                    issue_df,
                    x="issue",
                    y="percentage",
                    color="issue",
                    text="count",
                    title="Issue Distribution",
                )
                issue_chart.update_layout(xaxis_title="Issue", yaxis_title="Share of Feedback (%)", showlegend=False)
                st.plotly_chart(issue_chart, use_container_width=True)
        with col2:
            if not sentiment_df.empty:
                sentiment_chart = px.pie(
                    sentiment_df,
                    names="sentiment",
                    values="percentage",
                    title="Sentiment Mix",
                    color="sentiment",
                    color_discrete_map={"Positive": "#0d9488", "Neutral": "#f59e0b", "Negative": "#dc2626"},
                )
                st.plotly_chart(sentiment_chart, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            if not keyword_df.empty:
                keyword_chart = px.bar(
                    keyword_df.sort_values("count", ascending=True),
                    x="count",
                    y="keyword",
                    orientation="h",
                    color="count",
                    title="Top Keywords",
                )
                keyword_chart.update_layout(xaxis_title="Mentions", yaxis_title="Keyword", coloraxis_showscale=False)
                st.plotly_chart(keyword_chart, use_container_width=True)
            else:
                st.subheader("Top Keywords")
                st.write(", ".join(nlp_data.get("keywords", [])) or "No keywords available yet.")
        with col2:
            if not issue_sentiment_df.empty:
                heatmap_chart = px.density_heatmap(
                    issue_sentiment_df,
                    x="issue",
                    y="sentiment",
                    z="feedback_count",
                    histfunc="sum",
                    text_auto=True,
                    color_continuous_scale="Teal",
                    title="Issue vs Sentiment",
                )
                heatmap_chart.update_layout(xaxis_title="Issue Category", yaxis_title="Sentiment")
                st.plotly_chart(heatmap_chart, use_container_width=True)

        if not recent_feedback_df.empty:
            st.subheader("Recent Feedback Signals")
            st.dataframe(recent_feedback_df, use_container_width=True, hide_index=True)


elif page == "Workflow Monitor":
    st.markdown(
        """
        <div class="hero">
            <h1 style="margin:0;">Workflow Monitor</h1>
            <p style="margin:0.7rem 0 0 0;">
                Follow the operational pipeline from risk detection to action, follow-up, and outcome.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns(2)
    with col1:
        stage_df = pd.DataFrame(
            [{"Stage": key, "Count": value} for key, value in workflow_data.get("stage_counts", {}).items()]
        )
        if not stage_df.empty:
            st.plotly_chart(px.bar(stage_df, x="Stage", y="Count", color="Stage", title="Workflow Stage Volume"), use_container_width=True)
    with col2:
        status_df = pd.DataFrame(
            [{"Status": key, "Count": value} for key, value in workflow_data.get("status_counts", {}).items()]
        )
        if not status_df.empty:
            st.plotly_chart(px.pie(status_df, names="Status", values="Count", title="Workflow Status Mix"), use_container_width=True)
    st.dataframe(pd.DataFrame(workflow_data.get("records", [])), use_container_width=True, hide_index=True)


elif page == "Retention Command Center":
    st.markdown(
        """
        <div class="hero">
            <h1 style="margin:0;">Retention Command Center</h1>
            <p style="margin:0.7rem 0 0 0;">
                Review customer records, retention actions, and email automation status from one operations view.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(f"This command center is showing preview data from `{active_demo_dataset_name}`.")
    if st.button("Prepare Retention Email Preview"):
        st.success("Preview email workflow is ready. To persist and send actual emails, first run Batch Scoring for the selected company.")

    if not dashboard_data.empty:
        st.subheader("Customer Intelligence")
        st.dataframe(
            dashboard_data[
                [
                    "Customer ID",
                    "Customer Name",
                    "Email",
                    "Risk Level",
                    "Churn Probability",
                    "Issue Category",
                    "Offer",
                    "Action",
                ]
            ].sort_values("Churn Probability", ascending=False).head(40),
            use_container_width=True,
            hide_index=True,
        )
    if action_center_data:
        st.subheader("Retention Actions")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Action Type": "Preview Action",
                        "Title": item["recommended_actions"][0],
                        "Priority": "High" if item["churn_probability"] >= 0.8 else "Medium",
                        "Status": item["status"],
                        "Assigned Agent": item.get("assigned_agent") or "Retention Team",
                        "Offer": item.get("offer"),
                        "Email Status": item["action_statuses"].get("email", "pending"),
                    }
                    for item in action_center_data[:25]
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )


elif page == "Deployment Center":
    st.markdown(
        """
        <div class="hero">
            <h1 style="margin:0;">Deployment Center</h1>
            <p style="margin:0.7rem 0 0 0;">
                The project now includes FastAPI integration, MySQL-ready storage, Docker support, and cloud-friendly configuration.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    deployment_points = [
        "FastAPI backend available in api.py for training, prediction, segmentation, recommendations, and email actions.",
        "Streamlit dashboard available in app.py for business users.",
        f"SQLAlchemy database layer supports MySQL and {'SQLite fallback' if USE_SQLITE_FALLBACK else 'strict primary database mode'}.",
        "SHAP explanations and KMeans segmentation are wired into the ML flow.",
        "Retention actions and email status are persisted in the database.",
        "Action Center, NLP Insights, workflow monitoring, timeline, and what-if simulation are enabled in the UI.",
        f"Environment mode: {('Production' if IS_PRODUCTION else 'Non-production')} | Demo mode: {'Enabled' if ENABLE_DEMO_MODE else 'Disabled'}.",
        "JWT secret, database URL, and any optional demo accounts should be configured through environment variables before deployment.",
    ]
    for point in deployment_points:
        st.markdown(f"- {point}")

    st.code(
        """
docker compose up --build

# API
http://localhost:8000/docs

# Dashboard
http://localhost:8501
        """.strip(),
        language="bash",
    )
