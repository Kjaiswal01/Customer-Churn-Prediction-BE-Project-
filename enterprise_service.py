from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta
import json
import logging
import smtplib
import threading
from email.mime.text import MIMEText
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sqlalchemy.orm import Session

from data_pipeline import classify_issue, ensure_dataset_exists, simple_sentiment_score
from dynamic_dataset import adapt_company_dataset
from enterprise_config import (
    ARTIFACT_PATH,
    DATA_DIR,
    MODEL_VERSION_DIR,
    SMTP_FROM_EMAIL,
    SMTP_HOST,
    SMTP_PASSWORD,
    SMTP_PORT,
    SMTP_USE_TLS,
    SMTP_USERNAME,
    UPLOAD_DIR,
)
from enterprise_database import (
    ActionLog,
    AuditLog,
    Company,
    CustomerFeedback,
    CustomerRecord,
    DatasetUpload,
    PredictionLog,
    RetentionAction,
    SessionLocal,
    TimelineEvent,
    WorkflowRecord,
)
from enterprise_ml import (
    TrainingArtifacts,
    evaluate_models,
    load_artifacts,
    load_artifacts_from_path,
    load_dataset_from_upload,
    persist_artifacts,
    score_customers,
    validate_dataset,
)


MODEL_SIGNAL_COLUMNS = [
    "Tenure",
    "Monthly_Charges",
    "Total_Spend",
    "Usage_Score",
    "Last_Interaction_Days",
    "Support_Tickets",
    "Payment_Delay_Days",
    "Feedback",
    "Contract_Type",
    "Subscription_Type",
]

MIN_SIGNAL_COLUMNS_FOR_SCORING = 5
MIN_SIGNAL_COLUMNS_FOR_TRAINING = 6
_BOOTSTRAP_LOCK = threading.Lock()


def ensure_company(session: Session, company_name: str, industry: str = "Telecom") -> Company:
    company = session.query(Company).filter(Company.name == company_name).one_or_none()
    if company:
        return company
    company = Company(name=company_name, industry=industry)
    session.add(company)
    session.commit()
    session.refresh(company)
    return company


def save_upload_locally(uploaded_file) -> Path:
    file_name = getattr(uploaded_file, "name", None) or getattr(uploaded_file, "filename", "dataset.csv")
    destination = UPLOAD_DIR / f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{file_name}"
    if hasattr(uploaded_file, "getbuffer"):
        content = uploaded_file.getbuffer()
    else:
        uploaded_file.file.seek(0)
        content = uploaded_file.file.read()
    destination.write_bytes(content)
    return destination


def save_dataframe_snapshot(df: pd.DataFrame, source_name: str) -> Path:
    safe_name = Path(source_name or "dataset.csv").name
    stem = Path(safe_name).stem or "dataset"
    destination = UPLOAD_DIR / f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{stem}.csv"
    df.to_csv(destination, index=False)
    return destination


def load_dataset_from_path(file_path: Path) -> pd.DataFrame:
    if file_path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(file_path)
    return pd.read_csv(file_path)


def enrich_customer_dataset(df: pd.DataFrame) -> pd.DataFrame:
    enriched, _ = prepare_company_dataset(df)
    enriched["Email"] = enriched["Email"].fillna(
        enriched["Customer_ID"].astype(str).str.lower() + "@demo-retention.local"
    )
    return enriched


def sentiment_label(score: float) -> str:
    if score <= -0.15:
        return "Negative"
    if score >= 0.15:
        return "Positive"
    return "Neutral"


def extract_keywords(texts: list[str], top_n: int = 8) -> list[str]:
    clean_texts = [text.strip() for text in texts if str(text).strip()]
    if not clean_texts:
        return []
    vectorizer = TfidfVectorizer(stop_words="english", max_features=40, ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(clean_texts)
    scores = np.asarray(matrix.mean(axis=0)).ravel()
    features = np.asarray(vectorizer.get_feature_names_out())
    top_indices = scores.argsort()[::-1][:top_n]
    return [str(features[index]) for index in top_indices]


def build_reason_text(row: pd.Series, explanation: list[dict] | None = None) -> str:
    reasons = []
    monthly = float(row.get("Monthly_Charges", row.get("monthly_charges", 0)) or 0)
    usage = float(row.get("Usage_Score", row.get("usage_score", 50)) or 50)
    tenure = float(row.get("Tenure", row.get("tenure", 0)) or 0)
    last_interaction = float(row.get("Last_Interaction_Days", row.get("last_interaction_days", 0)) or 0)
    sentiment = float(row.get("Sentiment", row.get("sentiment", 0)) or 0)
    issue = str(row.get("Issue_Category", row.get("issue_category", "")) or "")

    if monthly >= 1000:
        reasons.append("premium pricing pressure")
    if usage <= 35:
        reasons.append("low engagement")
    if tenure <= 6:
        reasons.append("new-customer onboarding risk")
    if last_interaction >= 30:
        reasons.append("reduced recent interaction")
    if sentiment <= -0.1:
        reasons.append("negative feedback sentiment")
    if issue and issue != "General Feedback":
        reasons.append(issue.lower())

    if explanation:
        top_drivers = [item.get("feature", "") for item in explanation[:3] if item.get("feature")]
        if top_drivers:
            reasons.append("model drivers: " + ", ".join(top_drivers))

    if not reasons:
        reasons.append("moderate commercial and behavioral churn signals")
    return "Customer is likely to churn due to " + ", ".join(reasons) + "."


def retention_success_probability(churn_probability: float, customer_value: float) -> float:
    success = 0.78 - (churn_probability * 0.42) + (0.06 if customer_value >= 12000 else 0)
    return round(float(np.clip(success, 0.15, 0.92)), 4)


def structured_recommendations(row: pd.Series, result: dict) -> list[dict]:
    probability = float(result["churn_probability"])
    monthly = float(row.get("Monthly_Charges", row.get("monthly_charges", 0)) or 0)
    usage = float(row.get("Usage_Score", row.get("usage_score", 50)) or 50)
    sentiment = float(row.get("Sentiment", row.get("sentiment", 0)) or 0)
    last_interaction = float(row.get("Last_Interaction_Days", row.get("last_interaction_days", 0)) or 0)
    issue = str(row.get("Issue_Category", row.get("issue_category", "General Feedback")) or "")
    plan: list[dict] = []

    if probability >= 0.75:
        plan.append({"title": "Immediate save call", "detail": "Call the customer within 24 hours with a retention specialist.", "priority": "Critical"})
    if issue == "Pricing Issue" or monthly >= 1000:
        plan.append({"title": "Pricing intervention", "detail": "Offer a temporary bill relief plan or bundle upgrade with price protection.", "priority": "High"})
    if issue == "Service Issue" or sentiment < -0.15:
        plan.append({"title": "Service recovery", "detail": "Assign priority support and commit to a clear resolution timeline.", "priority": "High"})
    if issue == "Network Issue":
        plan.append({"title": "Technical fix", "detail": "Schedule network diagnostics and share an issue-resolution ETA.", "priority": "High"})
    if usage < 40:
        plan.append({"title": "Usage coaching", "detail": "Provide onboarding guidance, feature education, and a product usage session.", "priority": "Medium"})
    if last_interaction >= 30:
        plan.append({"title": "Re-engagement follow-up", "detail": "Send a proactive check-in and personalize communication around customer goals.", "priority": "Medium"})
    if not plan:
        plan.append({"title": "Loyalty nurture", "detail": "Maintain regular engagement and reward continued usage with loyalty benefits.", "priority": "Low"})
    return plan[:5]


def predictor_assistant_response(question: str, row: pd.Series, result: dict) -> str:
    q = str(question or "").strip().lower()
    probability_pct = round(float(result["churn_probability"]) * 100, 1)
    issue = str(result.get("issue_category", "General Feedback"))
    strategy = result["retention_strategy"]["strategy"]
    offer = result["personalized_offer"]["offer_name"]
    recommendations = structured_recommendations(row, result)
    top_actions = "; ".join(f"{item['title']}: {item['detail']}" for item in recommendations[:3])

    if any(term in q for term in ["why", "kyu", "reason"]):
        return f"The customer's churn risk is {probability_pct}%. Main reasons include {issue.lower()}, weak recent behavior signals, and the model drivers shown in the chart. The best immediate strategy is {strategy}."
    if any(term in q for term in ["offer", "discount", "price"]):
        return f"The best offer for this customer is `{offer}` because the current churn risk is {probability_pct}% and the profile suggests targeted retention is better than a generic discount."
    if any(term in q for term in ["save", "retain", "rok", "stop"]):
        return f"Top actions to retain this customer: {top_actions}"
    if any(term in q for term in ["sentiment", "feedback", "nlp"]):
        return f"Sentiment analysis shows `{result.get('sentiment_label', 'Neutral')}` tone with issue category `{issue}`. This means outreach should directly address that concern instead of sending a generic campaign."
    return (
        f"Current churn probability is {probability_pct}%. Recommended strategy is {strategy} and best offer is {offer}. "
        f"Top next steps: {top_actions}"
    )


def build_workflow_payload(row: pd.Series, strategy: dict, offer: dict, explanation: list[dict]) -> dict:
    churn_probability = float(row["churn_probability"])
    customer_value = float(row["customer_value"])
    return {
        "priority": strategy.get("priority", "Medium"),
        "reason": build_reason_text(row, explanation),
        "retention_success_probability": retention_success_probability(churn_probability, customer_value),
        "recommended_actions": [
            strategy.get("action"),
            f"Offer: {offer.get('offer_name')}",
            "Follow up in 72 hours",
        ],
        "issue_category": row.get("Issue_Category", "General Feedback"),
        "sentiment_label": sentiment_label(float(row.get("Sentiment", 0) or 0)),
    }


def log_audit_event(
    session: Session,
    event_type: str,
    entity_type: str,
    entity_id: str,
    company_id: int | None = None,
    user_email: str | None = None,
    details: dict | None = None,
) -> None:
    session.add(
        AuditLog(
            company_id=company_id,
            user_email=user_email,
            event_type=event_type,
            entity_type=entity_type,
            entity_id=entity_id,
            details_json=details or {},
        )
    )


def row_from_customer_record(record: CustomerRecord) -> pd.Series:
    payload = dict(record.raw_payload or {})
    payload["churn_probability"] = float(record.churn_probability or payload.get("churn_probability", 0))
    payload["customer_value"] = float(record.customer_value or payload.get("customer_value", 0))
    payload["customer_segment"] = record.churn_segment or payload.get("customer_segment", "Growth Opportunity")
    return pd.Series(payload)


def prepare_company_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    adapted = adapt_company_dataset(df)
    schema_mapping = dict(adapted.schema_mapping)
    schema_mapping["adapted_row_count"] = int(len(adapted.dataframe))
    mapped_columns = schema_mapping.get("mapped_columns", {})
    signal_columns_present = [column for column in MODEL_SIGNAL_COLUMNS if column in mapped_columns]
    schema_mapping["signal_columns_present"] = signal_columns_present
    schema_mapping["signal_column_count"] = len(signal_columns_present)
    return adapted.dataframe, schema_mapping


def infer_missing_business_signals(df: pd.DataFrame, schema_mapping: dict) -> tuple[pd.DataFrame, dict]:
    enriched = df.copy()
    mapped_columns = dict(schema_mapping.get("mapped_columns") or {})
    inferred_columns: dict[str, str] = {}
    normalized_lookup = {str(column).strip().lower().replace(" ", "_"): column for column in enriched.columns}

    def mark_inferred(column_name: str, source: str) -> None:
        if column_name not in mapped_columns:
            mapped_columns[column_name] = source
        inferred_columns[column_name] = source

    def find_source(*aliases: str):
        for alias in aliases:
            if alias in normalized_lookup:
                return normalized_lookup[alias]
        return None

    internet_service_col = find_source("internetservice", "internet_service")
    contract_col = find_source("contract", "contract_type")
    payment_method_col = find_source("paymentmethod", "payment_method", "payment_mode")
    tech_support_col = find_source("techsupport", "tech_support")
    phone_service_col = find_source("phoneservice", "phone_service")
    online_security_col = find_source("onlinesecurity", "online_security")
    online_backup_col = find_source("onlinebackup", "online_backup")
    device_protection_col = find_source("deviceprotection", "device_protection")
    streaming_tv_col = find_source("streamingtv", "streaming_tv")
    streaming_movies_col = find_source("streamingmovies", "streaming_movies")
    paperless_billing_col = find_source("paperlessbilling", "paperless_billing")
    partner_col = find_source("partner")
    dependents_col = find_source("dependents")

    if "Subscription_Type" not in mapped_columns:
        if internet_service_col is not None:
            internet_service = enriched[internet_service_col].fillna("Unknown").astype(str).str.strip()
            contract = (
                enriched[contract_col].fillna("Unknown").astype(str).str.strip()
                if contract_col is not None else pd.Series(["Unknown"] * len(enriched), index=enriched.index)
            )
            enriched["Subscription_Type"] = np.where(
                contract.str.contains("two year|one year|annual", case=False, regex=True),
                "Premium",
                np.where(internet_service.str.contains("fiber", case=False, regex=True), "Premium", np.where(internet_service.str.contains("dsl", case=False, regex=True), "Standard", "Basic")),
            )
            mark_inferred("Subscription_Type", "derived:InternetService+Contract")

    if "Usage_Score" not in mapped_columns:
        service_signal_columns = [
            phone_service_col,
            online_security_col,
            online_backup_col,
            device_protection_col,
            tech_support_col,
            streaming_tv_col,
            streaming_movies_col,
        ]
        service_signal_columns = [column for column in service_signal_columns if column is not None]
        if service_signal_columns:
            active_service_count = pd.Series([0] * len(enriched), index=enriched.index, dtype=float)
            for column in service_signal_columns:
                values = enriched[column].fillna("No").astype(str).str.lower()
                active_service_count += values.isin({"yes", "dsl", "fiber optic", "fiber", "month-to-month", "one year", "two year"}).astype(float)
            tenure_signal = pd.to_numeric(enriched.get("Tenure"), errors="coerce").fillna(12).clip(0, 72) / 72
            enriched["Usage_Score"] = np.clip(20 + (active_service_count * 10) + (tenure_signal * 18), 5, 95)
            mark_inferred("Usage_Score", "derived:service_bundle")

    if "Monthly_Charges" not in mapped_columns:
        total_spend = pd.to_numeric(enriched.get("Total_Spend"), errors="coerce")
        tenure = pd.to_numeric(enriched.get("Tenure"), errors="coerce").replace(0, np.nan)
        estimated_monthly = (total_spend / tenure).replace([np.inf, -np.inf], np.nan)
        if estimated_monthly.notna().any():
            enriched["Monthly_Charges"] = estimated_monthly
            mark_inferred("Monthly_Charges", "derived:Total_Spend/Tenure")

    if "Total_Spend" not in mapped_columns:
        monthly = pd.to_numeric(enriched.get("Monthly_Charges"), errors="coerce")
        tenure = pd.to_numeric(enriched.get("Tenure"), errors="coerce")
        estimated_total = monthly * tenure
        if estimated_total.notna().any():
            enriched["Total_Spend"] = estimated_total
            mark_inferred("Total_Spend", "derived:Monthly_Charges*Tenure")

    if "Last_Interaction_Days" not in mapped_columns:
        usage = pd.to_numeric(enriched.get("Usage_Score"), errors="coerce")
        if usage.notna().any():
            inferred_last_interaction = np.clip(45 - (usage.fillna(50) * 0.35), 0, 180)
            enriched["Last_Interaction_Days"] = inferred_last_interaction
            mark_inferred("Last_Interaction_Days", "derived:Usage_Score")

    if "Usage_Score" not in mapped_columns:
        last_interaction = pd.to_numeric(enriched.get("Last_Interaction_Days"), errors="coerce")
        if last_interaction.notna().any():
            inferred_usage = np.clip(100 - (last_interaction.fillna(15) * 1.6), 0, 100)
            enriched["Usage_Score"] = inferred_usage
            mark_inferred("Usage_Score", "derived:Last_Interaction_Days")

    if "Support_Tickets" not in mapped_columns and "Issue_Category" in enriched.columns:
        inferred_tickets = enriched["Issue_Category"].fillna("").astype(str).str.strip().ne("").astype(int)
        if inferred_tickets.any():
            enriched["Support_Tickets"] = inferred_tickets
            mark_inferred("Support_Tickets", "derived:Issue_Category")

    if "Support_Tickets" not in mapped_columns and tech_support_col is not None:
        tech_support = enriched[tech_support_col].fillna("No").astype(str).str.lower()
        online_security = (
            enriched[online_security_col].fillna("No").astype(str).str.lower()
            if online_security_col is not None else pd.Series(["no"] * len(enriched), index=enriched.index)
        )
        streaming_pressure = pd.Series([0] * len(enriched), index=enriched.index, dtype=float)
        for column in [streaming_tv_col, streaming_movies_col]:
            if column is not None:
                streaming_pressure += enriched[column].fillna("No").astype(str).str.lower().eq("yes").astype(float)
        enriched["Support_Tickets"] = (
            1
            + tech_support.eq("no").astype(int)
            + online_security.eq("no").astype(int)
            + (streaming_pressure >= 2).astype(int)
        ).clip(0, 6)
        mark_inferred("Support_Tickets", "derived:TechSupport+OnlineSecurity")

    if "Payment_Delay_Days" not in mapped_columns:
        payment_mode = enriched.get("Payment_Mode")
        if payment_mode is not None:
            inferred_delay = np.where(
                payment_mode.fillna("").astype(str).str.lower().str.contains("autopay|auto pay|upi"),
                0,
                4,
            )
            enriched["Payment_Delay_Days"] = inferred_delay
            mark_inferred("Payment_Delay_Days", "derived:Payment_Mode")

    if "Payment_Mode" not in mapped_columns and payment_method_col is not None:
        enriched["Payment_Mode"] = enriched[payment_method_col]
        mark_inferred("Payment_Mode", f"source:{payment_method_col}")

    if "Contract_Type" not in mapped_columns and contract_col is not None:
        enriched["Contract_Type"] = enriched[contract_col]
        mark_inferred("Contract_Type", f"source:{contract_col}")

    if "Feedback" not in mapped_columns:
        contract_series = (
            enriched["Contract_Type"].fillna("Unknown").astype(str)
            if "Contract_Type" in enriched.columns else pd.Series(["Unknown"] * len(enriched), index=enriched.index)
        )
        payment_series = (
            enriched["Payment_Mode"].fillna("Unknown").astype(str)
            if "Payment_Mode" in enriched.columns else pd.Series(["Unknown"] * len(enriched), index=enriched.index)
        )
        internet_series = (
            enriched[internet_service_col].fillna("Unknown").astype(str)
            if internet_service_col is not None else pd.Series(["Unknown"] * len(enriched), index=enriched.index)
        )
        tech_series = (
            enriched[tech_support_col].fillna("No").astype(str)
            if tech_support_col is not None else pd.Series(["No"] * len(enriched), index=enriched.index)
        )
        enriched["Feedback"] = np.where(
            tech_series.str.lower().eq("no"),
            "Customer may need stronger support and onboarding help.",
            np.where(
                payment_series.str.lower().str.contains("electronic check|mailed check", regex=True),
                "Customer billing and payment experience may need attention.",
                np.where(
                    internet_series.str.lower().str.contains("fiber", regex=True),
                    "Customer expects strong network and premium service quality.",
                    "Customer shows standard service usage and moderate engagement.",
                ),
            ),
        )
        enriched["Feedback"] = np.where(
            contract_series.str.lower().str.contains("month", regex=True),
            enriched["Feedback"].astype(str) + " Short-term contract suggests higher churn sensitivity.",
            enriched["Feedback"],
        )
        mark_inferred("Feedback", "derived:contract+payment+service")

    if "Issue_Category" not in mapped_columns:
        payment_series = (
            enriched["Payment_Mode"].fillna("Unknown").astype(str)
            if "Payment_Mode" in enriched.columns else pd.Series(["Unknown"] * len(enriched), index=enriched.index)
        )
        tech_series = (
            enriched[tech_support_col].fillna("No").astype(str)
            if tech_support_col is not None else pd.Series(["No"] * len(enriched), index=enriched.index)
        )
        internet_series = (
            enriched[internet_service_col].fillna("Unknown").astype(str)
            if internet_service_col is not None else pd.Series(["Unknown"] * len(enriched), index=enriched.index)
        )
        enriched["Issue_Category"] = np.where(
            payment_series.str.lower().str.contains("check", regex=True),
            "Pricing Issue",
            np.where(
                tech_series.str.lower().eq("no"),
                "Service Issue",
                np.where(internet_series.str.lower().str.contains("fiber|dsl", regex=True), "Network Issue", "General Feedback"),
            ),
        )
        mark_inferred("Issue_Category", "derived:payment+support+internet")

    if "Last_Interaction_Days" not in mapped_columns:
        usage = pd.to_numeric(enriched.get("Usage_Score"), errors="coerce")
        tenure = pd.to_numeric(enriched.get("Tenure"), errors="coerce").fillna(12)
        if usage.notna().any():
            contract_factor = (
                enriched["Contract_Type"].fillna("").astype(str).str.lower().str.contains("month").astype(int) * 8
                if "Contract_Type" in enriched.columns else 0
            )
            enriched["Last_Interaction_Days"] = np.clip(55 - (usage.fillna(50) * 0.4) - (tenure.clip(0, 60) * 0.18) + contract_factor, 0, 180)
            mark_inferred("Last_Interaction_Days", "derived:Usage_Score+Tenure")

    if "Age" not in mapped_columns:
        senior = (
            pd.to_numeric(enriched[find_source("seniorcitizen")] if find_source("seniorcitizen") else 0, errors="coerce").fillna(0)
            if find_source("seniorcitizen") is not None else pd.Series([0] * len(enriched), index=enriched.index)
        )
        partner = (
            enriched[partner_col].fillna("No").astype(str).str.lower().eq("yes").astype(int)
            if partner_col is not None else pd.Series([0] * len(enriched), index=enriched.index)
        )
        dependents = (
            enriched[dependents_col].fillna("No").astype(str).str.lower().eq("yes").astype(int)
            if dependents_col is not None else pd.Series([0] * len(enriched), index=enriched.index)
        )
        enriched["Age"] = np.clip(32 + (senior * 24) + (partner * 4) + (dependents * 3), 18, 78)
        mark_inferred("Age", "derived:SeniorCitizen+Partner+Dependents")

    enriched, refreshed_mapping = prepare_company_dataset(enriched)
    refreshed_mapping["mapped_columns"] = mapped_columns
    refreshed_mapping["inferred_columns"] = inferred_columns
    refreshed_mapping["signal_columns_present"] = [column for column in MODEL_SIGNAL_COLUMNS if column in mapped_columns]
    refreshed_mapping["signal_column_count"] = len(refreshed_mapping["signal_columns_present"])
    refreshed_mapping["usable_feature_count"] = int(
        len(
            [
                column for column in refreshed_mapping.get("adapted_columns", [])
                if column not in {"Customer_ID", "Customer_Name", "Email", "Churn"}
            ]
        )
    )
    return enriched, refreshed_mapping


def validate_dataset_readiness(schema_mapping: dict, require_target: bool) -> None:
    mapped_columns = schema_mapping.get("mapped_columns", {})
    signal_columns_present = schema_mapping.get("signal_columns_present") or [
        column for column in MODEL_SIGNAL_COLUMNS if column in mapped_columns
    ]
    minimum_required = MIN_SIGNAL_COLUMNS_FOR_TRAINING if require_target else MIN_SIGNAL_COLUMNS_FOR_SCORING
    usable_feature_count = int(schema_mapping.get("usable_feature_count", 0))
    if len(signal_columns_present) < minimum_required and usable_feature_count < 8:
        raise ValueError(
            "Dataset does not have enough predictive business signals. "
            f"Found {len(signal_columns_present)} useful mapped fields and {usable_feature_count} usable features, but need at least {minimum_required} signals or 8 usable features. "
            "Please include columns such as tenure, monthly charges, total spend, usage score, last interaction days, "
            "support tickets, payment delay days, subscription type, contract type, or feedback."
        )
    if require_target and "Churn" not in mapped_columns:
        raise ValueError(
            "Training dataset must include a churn target column such as Churn, target, label, attrition, or exit_flag."
        )


def latest_training_upload(session: Session, company_id: int) -> DatasetUpload | None:
    return (
        session.query(DatasetUpload)
        .filter(DatasetUpload.company_id == company_id, DatasetUpload.dataset_role == "training")
        .order_by(DatasetUpload.created_at.desc())
        .first()
    )


def resolve_artifacts_for_company(
    session: Session | None,
    company_id: int | None = None,
    allow_global_fallback: bool = True,
) -> TrainingArtifacts | None:
    if company_id and session is not None:
        training_upload = latest_training_upload(session, company_id)
        if training_upload:
            schema_mapping = training_upload.schema_mapping or {}
            artifact_path = schema_mapping.get("artifact_path")
            artifacts = load_artifacts_from_path(artifact_path) if artifact_path else None
            if artifacts is not None:
                return artifacts
        if not allow_global_fallback:
            return None

    if allow_global_fallback:
        artifacts = load_artifacts()
        if artifacts is not None:
            return artifacts

        return ensure_artifacts_available()
    return None


def ensure_artifacts_available(session: Session | None = None) -> TrainingArtifacts | None:
    artifacts = load_artifacts()
    if artifacts is not None:
        return artifacts

    owns_session = session is None
    session = session or SessionLocal()
    try:
        base_df = ensure_dataset_exists()
        train_models_from_dataframe(session, "Demo Telecom", "Telecom", base_df, "customer_churn.csv")
        artifacts = load_artifacts()
        return artifacts
    except Exception as exc:
        logger.warning("Unable to ensure artifacts automatically: %s", exc)
        return None
    finally:
        if owns_session:
            session.close()


def build_model_version(company_name: str) -> str:
    safe_name = "".join(character.lower() if character.isalnum() else "-" for character in company_name).strip("-")
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    return f"{safe_name or 'company'}-{timestamp}-{uuid4().hex[:6]}"


def persist_versioned_artifacts(artifacts: TrainingArtifacts) -> str:
    version_path = MODEL_VERSION_DIR / f"{artifacts.model_version}.joblib"
    persist_artifacts(artifacts)
    if ARTIFACT_PATH != version_path:
        version_path.write_bytes(ARTIFACT_PATH.read_bytes())
    return str(version_path)


def detect_customer_name(row: pd.Series) -> str:
    for candidate in ("Customer_Name", "Name", "Full_Name", "customer_name", "name"):
        value = row.get(candidate)
        if pd.notna(value) and str(value).strip():
            return str(value).strip()
    customer_id = row.get("Customer_ID") or row.get("customer_id") or "Customer"
    return str(customer_id)


def risk_level_from_probability(probability: float) -> str:
    if probability >= 0.85:
        return "High"
    if probability >= 0.6:
        return "Medium"
    return "Low"


def build_personalized_email(customer_name: str, offer_name: str, offer_details: str, action_text: str) -> tuple[str, str]:
    subject = f"Personalized retention offer for {customer_name}"
    body = (
        f"Hi {customer_name},\n\n"
        "We noticed a few signals that suggest your experience can be improved, and we want to help right away.\n\n"
        f"Recommended support action: {action_text}\n"
        f"Retention offer: {offer_name}\n"
        f"Offer details: {offer_details}\n\n"
        "Reply to this email and our retention team will prioritize your case.\n\n"
        "Best regards,\nRetention Team"
    )
    return subject, body


def latest_action_statuses(session: Session, customer_record_ids: list[int]) -> dict[int, dict[str, str]]:
    if not customer_record_ids:
        return {}
    logs = (
        session.query(ActionLog)
        .filter(ActionLog.customer_record_id.in_(customer_record_ids))
        .order_by(ActionLog.created_at.desc())
        .all()
    )
    statuses: dict[int, dict[str, str]] = {}
    for log in logs:
        customer_status = statuses.setdefault(log.customer_record_id, {})
        action_key = (log.action_type or log.action_name or "action").lower()
        if action_key not in customer_status:
            customer_status[action_key] = str(log.status)
    return statuses


def purge_company_scoring_data(session: Session, company_id: int) -> None:
    customer_ids = [
        row[0]
        for row in session.query(CustomerRecord.id).filter(CustomerRecord.company_id == company_id).all()
    ]
    if customer_ids:
        session.query(ActionLog).filter(ActionLog.company_id == company_id).delete(synchronize_session=False)
        session.query(CustomerFeedback).filter(CustomerFeedback.company_id == company_id).delete(synchronize_session=False)
        session.query(TimelineEvent).filter(TimelineEvent.company_id == company_id).delete(synchronize_session=False)
        session.query(WorkflowRecord).filter(WorkflowRecord.company_id == company_id).delete(synchronize_session=False)
        session.query(RetentionAction).filter(RetentionAction.company_id == company_id).delete(synchronize_session=False)
        session.query(PredictionLog).filter(PredictionLog.company_id == company_id).delete(synchronize_session=False)
        session.query(CustomerRecord).filter(CustomerRecord.company_id == company_id).delete(synchronize_session=False)
    session.query(DatasetUpload).filter(
        DatasetUpload.company_id == company_id,
        DatasetUpload.dataset_role == "scoring",
    ).delete(synchronize_session=False)
    session.commit()


def train_models_from_upload(session: Session, company_name: str, industry: str, uploaded_file, uploaded_by_email: str | None = None) -> dict:
    company = ensure_company(session, company_name, industry)
    stored_path = save_upload_locally(uploaded_file)
    df, schema_mapping = prepare_company_dataset(load_dataset_from_upload(uploaded_file))
    df, schema_mapping = infer_missing_business_signals(df, schema_mapping)
    validate_dataset_readiness(schema_mapping, require_target=True)
    validation = validate_dataset(df, require_target=True)
    model_version = build_model_version(company.name)
    artifacts, preview = evaluate_models(df, validation["target_column"])
    artifacts.model_version = model_version
    artifacts.schema_profile = schema_mapping
    versioned_path = persist_versioned_artifacts(artifacts)

    upload = DatasetUpload(
        company_id=company.id,
        filename=getattr(uploaded_file, "name", None) or getattr(uploaded_file, "filename", "dataset.csv"),
        uploaded_by_email=uploaded_by_email,
        stored_path=str(stored_path),
        row_count=len(df),
        dataset_role="training",
        model_version=model_version,
        schema_mapping={**schema_mapping, **validation, "artifact_path": versioned_path},
    )
    session.add(upload)
    session.commit()
    session.refresh(upload)

    return {
        "company_id": company.id,
        "upload_id": upload.id,
        "model_version": model_version,
        "schema_mapping": upload.schema_mapping,
        "metrics": artifacts.metrics,
        "best_model": artifacts.best_model_name,
        "training_columns": artifacts.training_columns,
        "dropped_columns": artifacts.dropped_columns,
        "evaluation_summary": artifacts.evaluation_summary,
        "preview": preview.head(20).to_dict(orient="records"),
    }


def train_models_from_dataframe(session: Session, company_name: str, industry: str, df: pd.DataFrame, source_name: str, uploaded_by_email: str | None = None) -> dict:
    company = ensure_company(session, company_name, industry)
    enriched, schema_mapping = prepare_company_dataset(df)
    enriched, schema_mapping = infer_missing_business_signals(enriched, schema_mapping)
    validate_dataset_readiness(schema_mapping, require_target=True)
    validation = validate_dataset(enriched, require_target=True)
    model_version = build_model_version(company.name)
    artifacts, preview = evaluate_models(enriched, validation["target_column"])
    artifacts.model_version = model_version
    artifacts.schema_profile = schema_mapping
    versioned_path = persist_versioned_artifacts(artifacts)
    stored_path = save_dataframe_snapshot(df, source_name)

    upload = DatasetUpload(
        company_id=company.id,
        filename=source_name,
        uploaded_by_email=uploaded_by_email,
        stored_path=str(stored_path),
        row_count=len(enriched),
        dataset_role="training",
        model_version=model_version,
        schema_mapping={**schema_mapping, **validation, "artifact_path": versioned_path},
    )
    session.add(upload)
    session.commit()
    session.refresh(upload)

    return {
        "company_id": company.id,
        "upload_id": upload.id,
        "model_version": model_version,
        "schema_mapping": upload.schema_mapping,
        "metrics": artifacts.metrics,
        "best_model": artifacts.best_model_name,
        "training_columns": artifacts.training_columns,
        "dropped_columns": artifacts.dropped_columns,
        "evaluation_summary": artifacts.evaluation_summary,
        "preview": preview.head(20).to_dict(orient="records"),
    }


def create_supporting_records(
    session: Session,
    company_id: int,
    record: CustomerRecord,
    row: pd.Series,
    strategy: dict,
    offer: dict,
    explanation: list[dict],
) -> None:
    reason = build_reason_text(row, explanation)
    metadata = build_workflow_payload(row, strategy, offer, explanation)
    feedback_text = str(row.get("Feedback", row.get("feedback", "")) or "")
    if feedback_text:
        session.add(
            CustomerFeedback(
                company_id=company_id,
                customer_record_id=record.id,
                feedback_text=feedback_text,
                sentiment_label=sentiment_label(float(row.get("Sentiment", 0) or 0)),
                sentiment_score=float(row.get("Sentiment", 0) or 0),
                issue_category=str(row.get("Issue_Category", "General Feedback")),
                keywords_json=extract_keywords([feedback_text], top_n=5),
            )
        )

    event_time = datetime.utcnow()
    timeline_events = [
        TimelineEvent(
            company_id=company_id,
            customer_record_id=record.id,
            event_type="prediction",
            title="Churn risk scored",
            description=reason,
            event_time=event_time,
            metadata_json={"probability": float(row["churn_probability"]), "segment": str(row["customer_segment"])},
        ),
        TimelineEvent(
            company_id=company_id,
            customer_record_id=record.id,
            event_type="complaint",
            title=f"{row.get('Issue_Category', 'General Feedback')} detected",
            description=feedback_text[:300] if feedback_text else "No feedback text supplied.",
            event_time=event_time - timedelta(days=2),
            metadata_json={"sentiment": float(row.get("Sentiment", 0) or 0)},
        ),
        TimelineEvent(
            company_id=company_id,
            customer_record_id=record.id,
            event_type="plan_change",
            title="Plan profile recorded",
            description=f"Current plan: {row.get('Subscription_Type', 'Unknown')} | Contract: {row.get('Contract_Type', 'Unknown')}",
            event_time=event_time - timedelta(days=7),
            metadata_json={"payment_mode": row.get("Payment_Mode", "Unknown")},
        ),
    ]
    for event in timeline_events:
        session.add(event)

    session.add(
        WorkflowRecord(
            company_id=company_id,
            customer_record_id=record.id,
            stage="Risk Detection",
            status="completed",
            owner="ML Engine",
            deadline=event_time,
            outcome="High Risk" if float(row["churn_probability"]) >= 0.7 else "Monitor",
            metadata_json=metadata,
        )
    )
    session.add(
        WorkflowRecord(
            company_id=company_id,
            customer_record_id=record.id,
            stage="Action",
            status="pending",
            owner="Retention Team",
            deadline=event_time + timedelta(days=3),
            metadata_json=metadata,
        )
    )
    log_audit_event(
        session,
        event_type="customer_scored",
        entity_type="customer_record",
        entity_id=str(record.id),
        company_id=company_id,
        details={"customer_id": record.external_customer_id, "probability": float(row["churn_probability"])},
    )


def ensure_company_operational_data(session: Session, company_id: int) -> dict:
    customer_rows = session.query(CustomerRecord).filter(CustomerRecord.company_id == company_id).all()
    if not customer_rows:
        return {"feedback_created": 0, "workflow_created": 0, "timeline_created": 0}

    existing_feedback = {
        customer_id
        for (customer_id,) in session.query(CustomerFeedback.customer_record_id)
        .filter(CustomerFeedback.company_id == company_id)
        .all()
    }
    existing_timeline = {
        customer_id
        for (customer_id,) in session.query(TimelineEvent.customer_record_id)
        .filter(TimelineEvent.company_id == company_id)
        .all()
    }
    existing_workflow = {
        customer_id
        for (customer_id,) in session.query(WorkflowRecord.customer_record_id)
        .filter(WorkflowRecord.company_id == company_id)
        .all()
    }
    existing_actions = {
        customer_id
        for (customer_id,) in session.query(RetentionAction.customer_record_id)
        .filter(RetentionAction.company_id == company_id)
        .all()
    }

    feedback_created = 0
    workflow_created = 0
    timeline_created = 0

    for record in customer_rows:
        row = row_from_customer_record(record)
        explanation = record.explanation if isinstance(record.explanation, list) else (record.explanation or {}).get("drivers", [])
        stored_strategy = row.get("retention_strategy")
        if isinstance(stored_strategy, dict):
            strategy = stored_strategy
        else:
            probability = float(row.get("churn_probability", 0))
            strategy = {
                "strategy": "VIP Rescue" if probability >= 0.8 else "Save Campaign" if probability >= 0.65 else "Engagement Boost",
                "action": record.recommended_action or "Trigger retention follow-up.",
                "priority": "Critical" if probability >= 0.8 else "High" if probability >= 0.65 else "Medium",
            }
        offer = generate_offer_payload(record)

        if record.id not in existing_actions:
            session.add(
                RetentionAction(
                    company_id=company_id,
                    customer_record_id=record.id,
                    action_type="Email Outreach" if record.email else "Manual Outreach",
                    title=strategy["strategy"],
                    description=strategy["action"],
                    offer_name=offer["offer_name"],
                    priority=strategy["priority"],
                    status="pending",
                    due_at=datetime.utcnow() + timedelta(days=3),
                    metadata_json={"offer": offer},
                )
            )

        if record.id not in existing_feedback:
            feedback_text = str(row.get("Feedback", "") or "").strip()
            if feedback_text:
                session.add(
                    CustomerFeedback(
                        company_id=company_id,
                        customer_record_id=record.id,
                        feedback_text=feedback_text,
                        sentiment_label=sentiment_label(float(row.get("Sentiment", 0) or 0)),
                        sentiment_score=float(row.get("Sentiment", 0) or 0),
                        issue_category=str(row.get("Issue_Category", "General Feedback")),
                        keywords_json=extract_keywords([feedback_text], top_n=5),
                    )
                )
                feedback_created += 1

        if record.id not in existing_timeline:
            reason = build_reason_text(row, explanation)
            session.add(
                TimelineEvent(
                    company_id=company_id,
                    customer_record_id=record.id,
                    event_type="prediction",
                    title="Churn risk scored",
                    description=reason,
                    metadata_json={"probability": float(row.get("churn_probability", 0)), "segment": str(row.get("customer_segment", ""))},
                )
            )
            timeline_created += 1

        if record.id not in existing_workflow:
            session.add(
                WorkflowRecord(
                    company_id=company_id,
                    customer_record_id=record.id,
                    stage="Risk Detection",
                    status="completed",
                    owner="ML Engine",
                    deadline=datetime.utcnow(),
                    outcome="High Risk" if float(row.get("churn_probability", 0)) >= 0.7 else "Monitor",
                    metadata_json={"reason": build_reason_text(row, explanation)},
                )
            )
            session.add(
                WorkflowRecord(
                    company_id=company_id,
                    customer_record_id=record.id,
                    stage="Action",
                    status="pending",
                    owner="Retention Team",
                    deadline=datetime.utcnow() + timedelta(days=3),
                    metadata_json={"offer": offer, "priority": strategy["priority"]},
                )
            )
            workflow_created += 2

    session.commit()
    return {
        "feedback_created": feedback_created,
        "workflow_created": workflow_created,
        "timeline_created": timeline_created,
    }


def score_dataset_and_store(session: Session, company_id: int, uploaded_file, uploaded_by_email: str | None = None) -> dict:
    artifacts = resolve_artifacts_for_company(session, company_id, allow_global_fallback=True)
    if artifacts is None:
        raise ValueError("No trained churn model is available yet. Train a labeled dataset once, then batch scoring will use that model automatically.")

    company = session.query(Company).filter(Company.id == company_id).one()
    purge_company_scoring_data(session, company.id)
    stored_path = save_upload_locally(uploaded_file)
    df, schema_mapping = prepare_company_dataset(load_dataset_from_upload(uploaded_file))
    df, schema_mapping = infer_missing_business_signals(df, schema_mapping)
    validate_dataset_readiness(schema_mapping, require_target=False)
    validation = validate_dataset(df, require_target=False)
    scored = score_customers(df, artifacts)

    upload = DatasetUpload(
        company_id=company.id,
        filename=getattr(uploaded_file, "name", None) or getattr(uploaded_file, "filename", "dataset.csv"),
        uploaded_by_email=uploaded_by_email,
        stored_path=str(stored_path),
        row_count=len(scored),
        dataset_role="scoring",
        model_version=getattr(artifacts, "model_version", None),
        schema_mapping={**schema_mapping, **validation, "model_version": getattr(artifacts, "model_version", None)},
    )
    session.add(upload)
    session.commit()
    session.refresh(upload)

    results = []
    id_column = validation["id_column"]
    email_column = validation["email_column"]

    for _, row in scored.iterrows():
        strategy = row["retention_strategy"]
        offer = row["personalized_offer"]
        explanation = row["shap_explanation"]
        reason = build_reason_text(row, explanation)
        success_probability = retention_success_probability(float(row["churn_probability"]), float(row["customer_value"]))
        external_customer_id = str(row[id_column]) if id_column and id_column in row else f"CUST-{_ + 1:05d}"
        email = str(row[email_column]) if email_column and email_column in row and pd.notna(row[email_column]) else None
        customer_name = detect_customer_name(row)

        record = CustomerRecord(
            company_id=company.id,
            external_customer_id=external_customer_id,
            customer_name=customer_name,
            email=email,
            raw_payload=json.loads(pd.Series(row).to_json(date_format="iso")),
            churn_probability=float(row["churn_probability"]),
            churn_prediction=bool(row["churn_prediction"]),
            churn_segment=str(row["customer_segment"]),
            churn_cluster=str(row["customer_segment"]),
            customer_value=float(row["customer_value"]),
            recommended_offer=offer["offer_name"],
            recommended_action=strategy["action"],
            explanation={"reason": reason, "drivers": explanation, "success_probability": success_probability},
        )
        session.add(record)
        session.flush()

        prediction_log = PredictionLog(
            company_id=company.id,
            customer_record_id=record.id,
            input_payload=record.raw_payload,
            prediction_payload={
                "churn_probability": float(row["churn_probability"]),
                "customer_segment": str(row["customer_segment"]),
                "retention_strategy": strategy,
                "personalized_offer": offer,
                "explanation": explanation,
            },
        )
        action = RetentionAction(
            company_id=company.id,
            customer_record_id=record.id,
            action_type="Email Outreach" if email else "Manual Outreach",
            title=strategy["strategy"],
            description=strategy["action"],
            offer_name=offer["offer_name"],
            priority=strategy["priority"],
            status="pending",
            due_at=datetime.utcnow() + timedelta(days=3),
            metadata_json={"offer": offer, "explanation": explanation, "reason": reason, "success_probability": success_probability},
        )
        session.add(prediction_log)
        session.add(action)
        create_supporting_records(session, company.id, record, row, strategy, offer, explanation)

        results.append(
            {
                "customer_id": external_customer_id,
                "customer_name": customer_name,
                "email": email,
                "churn_probability": round(float(row["churn_probability"]), 4),
                "segment": str(row["customer_segment"]),
                "customer_value": float(row["customer_value"]),
                "retention_strategy": strategy,
                "offer": offer,
                "explanation": explanation,
                "reason": reason,
                "retention_success_probability": success_probability,
            }
        )

    session.commit()
    return {
        "company": company.name,
        "rows": len(results),
        "model_version": getattr(artifacts, "model_version", None),
        "schema_mapping": upload.schema_mapping,
        "results": results,
    }


def score_dataframe_and_store(session: Session, company_id: int, df: pd.DataFrame, source_name: str = "seed_dataset.csv", uploaded_by_email: str | None = None) -> dict:
    artifacts = resolve_artifacts_for_company(session, company_id, allow_global_fallback=True)
    if artifacts is None:
        raise ValueError("No trained churn model is available yet. Train a labeled dataset once, then batch scoring will use that model automatically.")

    company = session.query(Company).filter(Company.id == company_id).one()
    purge_company_scoring_data(session, company.id)
    enriched, schema_mapping = prepare_company_dataset(df)
    enriched, schema_mapping = infer_missing_business_signals(enriched, schema_mapping)
    validate_dataset_readiness(schema_mapping, require_target=False)
    validation = validate_dataset(enriched, require_target=False)
    scored = score_customers(enriched, artifacts)
    stored_path = save_dataframe_snapshot(df, source_name)

    upload = DatasetUpload(
        company_id=company.id,
        filename=source_name,
        uploaded_by_email=uploaded_by_email,
        stored_path=str(stored_path),
        row_count=len(scored),
        dataset_role="scoring",
        model_version=getattr(artifacts, "model_version", None),
        schema_mapping={**schema_mapping, **validation, "model_version": getattr(artifacts, "model_version", None)},
    )
    session.add(upload)
    session.commit()
    session.refresh(upload)

    results = []
    id_column = validation["id_column"]
    email_column = validation["email_column"]

    for index, row in scored.iterrows():
        strategy = row["retention_strategy"]
        offer = row["personalized_offer"]
        explanation = row["shap_explanation"]
        reason = build_reason_text(row, explanation)
        success_probability = retention_success_probability(float(row["churn_probability"]), float(row["customer_value"]))
        external_customer_id = str(row[id_column]) if id_column and id_column in row else f"CUST-{index + 1:05d}"
        email = str(row[email_column]) if email_column and email_column in row and pd.notna(row[email_column]) else None
        customer_name = detect_customer_name(row)

        existing = session.query(CustomerRecord).filter(
            CustomerRecord.company_id == company.id,
            CustomerRecord.external_customer_id == external_customer_id,
        ).one_or_none()
        if existing:
            continue

        record = CustomerRecord(
            company_id=company.id,
            external_customer_id=external_customer_id,
            customer_name=customer_name,
            email=email,
            raw_payload=json.loads(pd.Series(row).to_json(date_format="iso")),
            churn_probability=float(row["churn_probability"]),
            churn_prediction=bool(row["churn_prediction"]),
            churn_segment=str(row["customer_segment"]),
            churn_cluster=str(row["customer_segment"]),
            customer_value=float(row["customer_value"]),
            recommended_offer=offer["offer_name"],
            recommended_action=strategy["action"],
            explanation={"reason": reason, "drivers": explanation, "success_probability": success_probability},
        )
        session.add(record)
        session.flush()

        session.add(
            PredictionLog(
                company_id=company.id,
                customer_record_id=record.id,
                input_payload=record.raw_payload,
                prediction_payload={
                    "churn_probability": float(row["churn_probability"]),
                    "customer_segment": str(row["customer_segment"]),
                    "retention_strategy": strategy,
                    "personalized_offer": offer,
                    "explanation": explanation,
                },
            )
        )
        session.add(
            RetentionAction(
                company_id=company.id,
                customer_record_id=record.id,
                action_type="Email Outreach" if email else "Manual Outreach",
                title=strategy["strategy"],
                description=strategy["action"],
                offer_name=offer["offer_name"],
                priority=strategy["priority"],
                status="pending",
                due_at=datetime.utcnow() + timedelta(days=3),
                metadata_json={"offer": offer, "explanation": explanation, "reason": reason, "success_probability": success_probability},
            )
        )
        create_supporting_records(session, company.id, record, row, strategy, offer, explanation)

        results.append(
            {
                "customer_id": external_customer_id,
                "customer_name": customer_name,
                "email": email,
                "churn_probability": round(float(row["churn_probability"]), 4),
                "segment": str(row["customer_segment"]),
                "customer_value": float(row["customer_value"]),
                "retention_strategy": strategy,
                "offer": offer,
                "explanation": explanation,
                "reason": reason,
                "retention_success_probability": success_probability,
            }
        )

    session.commit()
    return {
        "company": company.name,
        "rows": len(results),
        "model_version": getattr(artifacts, "model_version", None),
        "schema_mapping": upload.schema_mapping,
        "results": results,
    }


def predict_records(session: Session, records: list[dict]) -> list[dict]:
    artifacts = ensure_artifacts_available()
    if artifacts is None:
        raise ValueError("No trained model found. Train the platform first.")
    df = pd.DataFrame(records)
    scored = score_customers(df, artifacts)
    output = []
    for _, row in scored.iterrows():
        reason = build_reason_text(row, row["shap_explanation"])
        success_probability = retention_success_probability(float(row["churn_probability"]), float(row["customer_value"]))
        temp_result = {
            "churn_probability": round(float(row["churn_probability"]), 4),
            "retention_strategy": row["retention_strategy"],
            "personalized_offer": row["personalized_offer"],
            "issue_category": str(row.get("Issue_Category", "General Feedback")),
            "sentiment_label": sentiment_label(float(row.get("Sentiment", 0) or 0)),
        }
        recommendation_plan = structured_recommendations(row, temp_result)
        output.append(
            {
                "churn_probability": temp_result["churn_probability"],
                "churn_prediction": bool(row["churn_prediction"]),
                "customer_segment": str(row["customer_segment"]),
                "customer_value": float(row["customer_value"]),
                "retention_strategy": row["retention_strategy"],
                "personalized_offer": row["personalized_offer"],
                "shap_explanation": row["shap_explanation"],
                "reason": reason,
                "issue_category": temp_result["issue_category"],
                "sentiment_label": temp_result["sentiment_label"],
                "sentiment_score": round(float(row.get("Sentiment", 0) or 0), 3),
                "raw_model_probability": round(float(row.get("raw_model_probability", row["churn_probability"])), 4),
                "heuristic_probability": round(float(row.get("heuristic_probability", row["churn_probability"])), 4),
                "recommendation_plan": recommendation_plan,
                "retention_success_probability": success_probability,
                "assistant_default_answer": predictor_assistant_response("what should we do", row, temp_result),
            }
        )
    return output


def revenue_impact_metrics(rows: list[CustomerRecord]) -> dict:
    total_clv = sum(float(row.customer_value or 0) for row in rows)
    revenue_at_risk = sum(float(row.customer_value or 0) for row in rows if float(row.churn_probability or 0) >= 0.7)
    potential_recovery = revenue_at_risk * 0.6
    avg_clv = total_clv / len(rows) if rows else 0
    return {
        "total_revenue_at_risk": round(revenue_at_risk, 2),
        "potential_revenue_recovery": round(potential_recovery, 2),
        "average_clv": round(avg_clv, 2),
    }


def high_risk_customers(session: Session, company_id: int | None = None, threshold: float | None = None) -> list[dict]:
    artifacts = load_artifacts()
    effective_threshold = threshold or max(float(getattr(artifacts, "decision_threshold", 0.55)), 0.6)
    query = session.query(CustomerRecord).filter(CustomerRecord.churn_probability.is_not(None))
    if company_id:
        query = query.filter(CustomerRecord.company_id == company_id)
    rows = query.order_by(CustomerRecord.churn_probability.desc()).all()
    actions_query = session.query(RetentionAction)
    if company_id:
        actions_query = actions_query.filter(RetentionAction.company_id == company_id)
    actions = {action.customer_record_id: action for action in actions_query.order_by(RetentionAction.created_at.desc()).all()}
    status_map = latest_action_statuses(session, [row.id for row in rows])
    results = []
    for row in rows:
        probability = float(row.churn_probability or 0)
        if probability < effective_threshold:
            continue
        action = actions.get(row.id)
        payload = row.raw_payload or {}
        explanation = row.explanation or {}
        reason = explanation.get("reason") if isinstance(explanation, dict) else str(explanation)
        statuses = status_map.get(row.id, {})
        results.append(
            {
                "customer_record_id": row.id,
                "customer_id": row.external_customer_id,
                "customer_name": row.customer_name or str(payload.get("Customer_Name") or payload.get("name") or row.external_customer_id),
                "churn_probability": round(probability, 4),
                "risk_level": risk_level_from_probability(probability),
                "reason": reason,
                "recommended_actions": [
                    row.recommended_action,
                    f"Offer {row.recommended_offer}",
                    "Follow up after outreach",
                ],
                "offer": row.recommended_offer,
                "status": action.status if action else "pending",
                "assigned_agent": action.assigned_agent if action else None,
                "email": row.email,
                "action_statuses": {
                    "call": statuses.get("call", "pending"),
                    "email": statuses.get("email", action.email_status.lower() if action and action.email_status else "pending"),
                    "offer": statuses.get("offer", "pending"),
                    "assign": statuses.get("assign", "pending"),
                },
            }
        )
    return results


def simulate_business_impact(record: dict, reduce_price_pct: float = 0.0, improve_support: bool = False) -> dict:
    baseline = predict_records(None, [record])[0]
    scenario = dict(record)
    scenario["Monthly_Charges"] = round(float(scenario.get("Monthly_Charges", 0)) * (1 - reduce_price_pct / 100), 2)
    scenario["Total_Spend"] = round(float(scenario.get("Monthly_Charges", 0)) * float(scenario.get("Tenure", 1)), 2)
    if improve_support:
        scenario["Support_Tickets"] = max(0, int(scenario.get("Support_Tickets", 1)) - 1)
        scenario["Last_Interaction_Days"] = max(0, int(scenario.get("Last_Interaction_Days", 10)) - 10)
        current_feedback = str(scenario.get("Feedback", "") or "")
        scenario["Feedback"] = current_feedback + " Support follow-up improved recently."
        scenario["Sentiment"] = min(1.0, float(scenario.get("Sentiment", simple_sentiment_score(current_feedback))) + 0.25)
    scenario_result = predict_records(None, [scenario])[0]
    return {
        "before": baseline,
        "after": scenario_result,
        "change_in_probability": round((scenario_result["churn_probability"] - baseline["churn_probability"]) * 100, 2),
        "scenario": {"reduce_price_pct": reduce_price_pct, "improve_support": improve_support},
    }


def customer_timeline(session: Session, customer_record_id: int) -> list[dict]:
    events = (
        session.query(TimelineEvent)
        .filter(TimelineEvent.customer_record_id == customer_record_id)
        .order_by(TimelineEvent.event_time.desc())
        .all()
    )
    return [
        {
            "event_type": event.event_type,
            "title": event.title,
            "description": event.description,
            "event_time": event.event_time.isoformat(),
            "metadata": event.metadata_json or {},
        }
        for event in events
    ]


def workflow_overview(session: Session, company_id: int | None = None) -> dict:
    query = session.query(WorkflowRecord)
    if company_id:
        query = query.filter(WorkflowRecord.company_id == company_id)
    rows = query.all()
    stage_counts = Counter(row.stage for row in rows)
    status_counts = Counter(row.status for row in rows)
    return {
        "stage_counts": dict(stage_counts),
        "status_counts": dict(status_counts),
        "records": [
            {
                "customer_record_id": row.customer_record_id,
                "stage": row.stage,
                "status": row.status,
                "owner": row.owner,
                "deadline": row.deadline.isoformat() if row.deadline else None,
                "outcome": row.outcome,
            }
            for row in sorted(rows, key=lambda item: item.created_at, reverse=True)
        ],
    }


def nlp_issue_insights(session: Session, company_id: int | None = None) -> dict:
    query = session.query(CustomerFeedback)
    if company_id:
        query = query.filter(CustomerFeedback.company_id == company_id)
    rows = query.all()
    texts = [row.feedback_text for row in rows if row.feedback_text]
    keyword_counter = Counter()
    for row in rows:
        for keyword in row.keywords_json or []:
            keyword_counter[str(keyword)] += 1
    total = len(rows) or 1
    issue_counts = Counter(row.issue_category or "General Feedback" for row in rows)
    sentiment_counts = Counter(row.sentiment_label or "Neutral" for row in rows)
    avg_sentiment = round(
        float(np.mean([float(row.sentiment_score or 0) for row in rows])) if rows else 0.0,
        3,
    )
    issue_sentiment_rows = []
    for row in rows:
        issue_sentiment_rows.append(
            {
                "issue": row.issue_category or "General Feedback",
                "sentiment": row.sentiment_label or "Neutral",
                "score": float(row.sentiment_score or 0),
            }
        )
    issue_sentiment_summary = []
    if issue_sentiment_rows:
        issue_sentiment_df = pd.DataFrame(issue_sentiment_rows)
        grouped = (
            issue_sentiment_df.groupby(["issue", "sentiment"], as_index=False)
            .agg(feedback_count=("score", "size"), avg_sentiment=("score", "mean"))
            .sort_values(["issue", "feedback_count"], ascending=[True, False])
        )
        grouped["avg_sentiment"] = grouped["avg_sentiment"].round(3)
        issue_sentiment_summary = grouped.to_dict(orient="records")

    recent_feedback = [
        {
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "issue_category": row.issue_category or "General Feedback",
            "sentiment_label": row.sentiment_label or "Neutral",
            "sentiment_score": round(float(row.sentiment_score or 0), 3),
            "feedback_text": row.feedback_text,
            "keywords": ", ".join(row.keywords_json or []),
        }
        for row in sorted(rows, key=lambda item: item.created_at or datetime.min, reverse=True)[:12]
    ]
    return {
        "issue_summary": [
            {
                "issue": issue,
                "count": count,
                "percentage": round((count / total) * 100, 2),
            }
            for issue, count in issue_counts.most_common()
        ],
        "sentiment_summary": [
            {
                "sentiment": label,
                "count": count,
                "percentage": round((count / total) * 100, 2),
            }
            for label, count in sentiment_counts.most_common()
        ],
        "keywords": [keyword for keyword, _ in keyword_counter.most_common(10)] or extract_keywords(texts, top_n=10),
        "keyword_summary": [
            {"keyword": keyword, "count": count}
            for keyword, count in keyword_counter.most_common(12)
        ],
        "issue_sentiment_summary": issue_sentiment_summary,
        "recent_feedback": recent_feedback,
        "feedback_count": len(rows),
        "avg_sentiment_score": avg_sentiment,
    }


def forecast_churn(session: Session, company_id: int | None = None) -> dict:
    query = session.query(PredictionLog)
    if company_id:
        query = query.filter(PredictionLog.company_id == company_id)
    rows = query.order_by(PredictionLog.created_at.asc()).all()
    if not rows:
        return {"next_month_churn_pct": 0.0, "trend": []}
    trend_df = pd.DataFrame(
        [
            {
                "month": log.created_at.strftime("%Y-%m"),
                "churn_probability": float((log.prediction_payload or {}).get("churn_probability", 0)),
            }
            for log in rows
        ]
    )
    monthly = trend_df.groupby("month", as_index=False)["churn_probability"].mean()
    monthly["churn_pct"] = (monthly["churn_probability"] * 100).round(2)
    if len(monthly) >= 2:
        monthly["index"] = np.arange(len(monthly))
        slope, intercept = np.polyfit(monthly["index"], monthly["churn_pct"], 1)
        next_value = intercept + slope * len(monthly)
    else:
        next_value = float(monthly["churn_pct"].iloc[-1])
    return {
        "next_month_churn_pct": round(float(np.clip(next_value, 0, 100)), 2),
        "trend": monthly[["month", "churn_pct"]].to_dict(orient="records"),
    }


def ensure_retention_action(session: Session, record: CustomerRecord, action_name: str, assigned_agent: str | None = None) -> RetentionAction:
    action = (
        session.query(RetentionAction)
        .filter(RetentionAction.customer_record_id == record.id)
        .order_by(RetentionAction.created_at.desc())
        .one_or_none()
    )
    if action is None:
        action = RetentionAction(
            company_id=record.company_id,
            customer_record_id=record.id,
            action_type=action_name,
            title=action_name,
            description=record.recommended_action or action_name,
            offer_name=record.recommended_offer,
            priority="High",
            status="pending",
            due_at=datetime.utcnow() + timedelta(days=2),
            assigned_agent=assigned_agent,
            metadata_json={},
        )
        session.add(action)
        session.flush()
    if assigned_agent:
        action.assigned_agent = assigned_agent
    return action


def record_action_event(
    session: Session,
    record: CustomerRecord,
    action: RetentionAction,
    action_name: str,
    action_type: str,
    status: str,
    notes: str,
    actor_email: str | None = None,
    metadata: dict | None = None,
) -> dict:
    log = ActionLog(
        company_id=record.company_id,
        customer_record_id=record.id,
        customer_external_id=record.external_customer_id,
        retention_action_id=action.id,
        action_type=action_type,
        action_name=action_name,
        actor_email=actor_email,
        status=status,
        notes=notes,
        metadata_json=metadata or {},
    )
    session.add(log)
    session.add(
        TimelineEvent(
            company_id=record.company_id,
            customer_record_id=record.id,
            event_type="action",
            title=action_name,
            description=notes,
            metadata_json={"status": status, **(metadata or {})},
        )
    )
    session.add(
        WorkflowRecord(
            company_id=record.company_id,
            customer_record_id=record.id,
            stage="Follow-up" if status == "completed" else "Action",
            status=status,
            owner=(action.assigned_agent or actor_email or "Retention Team"),
            deadline=datetime.utcnow() + timedelta(days=2),
            outcome=action_name,
            metadata_json={"action_id": action.id, **(metadata or {})},
        )
    )
    log_audit_event(
        session,
        event_type="action_executed",
        entity_type="retention_action",
        entity_id=str(action.id),
        company_id=record.company_id,
        user_email=actor_email,
        details={"action_name": action_name, "action_type": action_type, "customer_record_id": record.id},
    )
    session.commit()
    return {
        "status": status,
        "message": notes,
        "action_id": action.id,
        "customer_id": record.external_customer_id,
        "action_type": action_type,
    }


def generate_offer_payload(record: CustomerRecord) -> dict:
    probability = float(record.churn_probability or 0)
    customer_value = float(record.customer_value or 0)
    payload = record.raw_payload or {}
    tenure = float(payload.get("Tenure", payload.get("tenure", 0)) or 0)

    if customer_value >= 12000 or probability >= 0.85:
        return {
            "offer_name": "Premium Rescue Bundle",
            "offer_details": "Dedicated manager, loyalty upgrade, and 20% credit for 3 months.",
        }
    if tenure >= 24:
        return {
            "offer_name": "Loyalty Plus Discount",
            "offer_details": "12% renewal discount and bonus loyalty points.",
        }
    return {
        "offer_name": "Retention Discount",
        "offer_details": "8% discount for 2 months and onboarding support refresh.",
    }


def execute_call_action(session: Session, customer_record_id: int, actor_email: str | None = None) -> dict:
    record = session.query(CustomerRecord).filter(CustomerRecord.id == customer_record_id).one()
    action = ensure_retention_action(session, record, "Call Customer")
    action.action_type = "Call"
    action.status = "completed"
    notes = "Customer call simulated successfully and logged."
    return record_action_event(
        session,
        record,
        action,
        action_name="Call Customer",
        action_type="call",
        status="completed",
        notes=notes,
        actor_email=actor_email,
    )


def execute_offer_action(session: Session, customer_record_id: int, actor_email: str | None = None) -> dict:
    record = session.query(CustomerRecord).filter(CustomerRecord.id == customer_record_id).one()
    action = ensure_retention_action(session, record, "Send Offer")
    offer = generate_offer_payload(record)
    action.action_type = "Offer"
    action.offer_name = offer["offer_name"]
    action.description = offer["offer_details"]
    action.status = "completed"
    action.metadata_json = {**(action.metadata_json or {}), "offer": offer}
    return record_action_event(
        session,
        record,
        action,
        action_name="Send Offer",
        action_type="offer",
        status="completed",
        notes=f"Offer generated: {offer['offer_name']}.",
        actor_email=actor_email,
        metadata={"offer": offer},
    )


def execute_assign_action(
    session: Session,
    customer_record_id: int,
    assigned_agent: str,
    actor_email: str | None = None,
) -> dict:
    record = session.query(CustomerRecord).filter(CustomerRecord.id == customer_record_id).one()
    action = ensure_retention_action(session, record, "Assign Agent", assigned_agent=assigned_agent)
    action.action_type = "Assignment"
    action.assigned_agent = assigned_agent
    action.status = "pending"
    return record_action_event(
        session,
        record,
        action,
        action_name="Assign Agent",
        action_type="assign",
        status="pending",
        notes=f"Customer assigned to {assigned_agent}.",
        actor_email=actor_email,
        metadata={"assigned_agent": assigned_agent},
    )


def execute_email_action(session: Session, customer_record_id: int, actor_email: str | None = None) -> dict:
    record = session.query(CustomerRecord).filter(CustomerRecord.id == customer_record_id).one()
    action = ensure_retention_action(session, record, "Send Email")
    offer = generate_offer_payload(record)
    customer_name = record.customer_name or record.external_customer_id
    if not record.email:
        action.email_status = "Skipped - No Email"
        action.status = "pending"
        return record_action_event(
            session,
            record,
            action,
            action_name="Send Email",
            action_type="email",
            status="pending",
            notes="Email skipped because the customer has no email address.",
            actor_email=actor_email,
        )
    subject, body = build_personalized_email(customer_name, offer["offer_name"], offer["offer_details"], record.recommended_action or "Priority follow-up")
    delivery_status = send_retention_email(record.email, subject, body)
    action.action_type = "Email"
    action.offer_name = offer["offer_name"]
    action.email_status = delivery_status
    action.description = body
    action.status = "completed" if delivery_status == "Sent" else "pending"
    return record_action_event(
        session,
        record,
        action,
        action_name="Send Email",
        action_type="email",
        status=action.status,
        notes="Email Sent" if delivery_status == "Sent" else delivery_status,
        actor_email=actor_email,
        metadata={"email_status": delivery_status, "offer": offer},
    )


def log_customer_action(
    session: Session,
    customer_record_id: int,
    action_name: str,
    actor_email: str | None = None,
    assigned_agent: str | None = None,
) -> dict:
    if action_name == "Call Customer":
        return execute_call_action(session, customer_record_id, actor_email)
    if action_name == "Send Email":
        return execute_email_action(session, customer_record_id, actor_email)
    if action_name == "Send Offer":
        return execute_offer_action(session, customer_record_id, actor_email)
    if action_name == "Assign Agent":
        return execute_assign_action(
            session,
            customer_record_id,
            assigned_agent or "agent@retentionos.local",
            actor_email,
        )
    raise ValueError(f"Unsupported action: {action_name}")


def send_retention_email(to_email: str, subject: str, body: str) -> str:
    if not SMTP_HOST or not SMTP_USERNAME or not SMTP_PASSWORD:
        return "Email skipped: SMTP configuration is not set."

    message = MIMEText(body)
    message["Subject"] = subject
    message["From"] = SMTP_FROM_EMAIL
    message["To"] = to_email

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        if SMTP_USE_TLS:
            server.starttls()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        server.send_message(message)
    return "Sent"


def send_retention_emails_for_company(session: Session, company_id: int) -> dict:
    customers = session.query(CustomerRecord).filter(CustomerRecord.company_id == company_id).all()
    sent = 0
    skipped = 0
    for customer in customers:
        if not customer.email:
            skipped += 1
            continue
        result = execute_email_action(session, customer.id, "system@retentionos.local")
        sent += 1 if result["status"] == "completed" else 0
        skipped += 0 if result["status"] == "completed" else 1
    return {"sent": sent, "skipped": skipped}


def dashboard_stats(session: Session, company_id: int | None = None) -> dict:
    query = session.query(CustomerRecord)
    if company_id:
        query = query.filter(CustomerRecord.company_id == company_id)
    rows = query.all()
    total_customers = len(rows)
    high_risk_count = sum(1 for row in rows if (row.churn_probability or 0) > 0.7)
    churn_rate = round((sum(1 for row in rows if row.churn_prediction) / total_customers) * 100, 2) if total_customers else 0
    revenue_stats = revenue_impact_metrics(rows)
    return {
        "total_customers": total_customers,
        "churn_rate": churn_rate,
        "high_risk_count": high_risk_count,
        "revenue_risk": revenue_stats["total_revenue_at_risk"],
        "potential_recovery": revenue_stats["potential_revenue_recovery"],
        "average_clv": revenue_stats["average_clv"],
    }


def churn_by_dimension(session: Session, dimension: str, company_id: int | None = None) -> list[dict]:
    rows = session.query(CustomerRecord).filter(CustomerRecord.churn_probability.is_not(None))
    if company_id:
        rows = rows.filter(CustomerRecord.company_id == company_id)
    records = rows.all()
    grouped: dict[str, list[float]] = {}
    for row in records:
        raw_payload = row.raw_payload or {}
        value = raw_payload.get(dimension) or raw_payload.get(dimension.lower()) or "Unknown"
        grouped.setdefault(str(value), []).append(float(row.churn_probability or 0))
    return [
        {"label": label, "churn_rate": round((sum(values) / len(values)) * 100, 2), "count": len(values)}
        for label, values in grouped.items()
    ]


def bootstrap_demo_environment(session: Session) -> None:
    with _BOOTSTRAP_LOCK:
        demo_company = ensure_company(session, "Demo Telecom", "Telecom")
        existing_records = session.query(CustomerRecord).filter(CustomerRecord.company_id == demo_company.id).count()
        if existing_records > 0:
            ensure_company_operational_data(session, demo_company.id)
            return

        enriched = ensure_dataset_exists()

        if ensure_artifacts_available(session) is None:
            train_models_from_dataframe(session, demo_company.name, demo_company.industry, enriched, "customer_churn.csv")

        score_dataframe_and_store(session, demo_company.id, enriched, "customer_churn.csv")
        ensure_company_operational_data(session, demo_company.id)
logger = logging.getLogger(__name__)
