from __future__ import annotations

from datetime import datetime, timedelta
import json
from pathlib import Path
import sqlite3
from typing import Any

import joblib
import pandas as pd
from textblob import TextBlob

try:
    import shap  # type: ignore
except Exception:  # pragma: no cover - optional dependency fallback
    shap = None

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
INSTANCE_DIR = BASE_DIR / "instance"
UPLOADS_DIR = DATA_DIR / "uploads"
DB_PATH = INSTANCE_DIR / "churn_retention.db"

INSTANCE_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = MODELS_DIR / "churn_model_tuned.pkl"
if not MODEL_FILE.exists():
    MODEL_FILE = MODELS_DIR / "churn_model.pkl"

SCALER_FILE = MODELS_DIR / "scaler.pkl"
FEATURE_LIST_FILE = MODELS_DIR / "feature_list.pkl"
LABEL_ENCODERS_FILE = MODELS_DIR / "label_encoders.pkl"
DATA_FILE = DATA_DIR / "customer_churn.csv"

model = None
scaler = None
feature_cols: list[str] = []
label_encoders: dict[str, Any] = {}
seed_dataset = pd.DataFrame()
model_load_error = None
shap_explainer = None

CANONICAL_COLUMNS = {
    "Customer_ID": ["customer_id", "cust_id", "subscriber_id", "account_id", "user_id"],
    "Gender": ["gender", "sex"],
    "Age": ["age", "customer_age"],
    "Tenure": ["tenure", "tenure_months", "months_active", "customer_tenure"],
    "Subscription_Type": ["subscription_type", "plan_type", "package", "segment_plan", "contract_type"],
    "Monthly_Charges": ["monthly_charges", "monthly_charge", "arpu", "monthly_spend", "bill_amount"],
    "Total_Spend": ["total_spend", "total_revenue", "lifetime_value", "revenue", "customer_value"],
    "Last_Interaction_Days": [
        "last_interaction_days",
        "days_since_last_interaction",
        "last_contact_days",
        "days_since_last_contact",
    ],
    "Feedback": ["feedback", "customer_feedback", "review_text", "complaint_text", "remarks", "comments"],
    "Churn": ["churn", "churn_status", "status", "is_churned"],
}


def get_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def utcnow_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")


def init_db() -> None:
    with get_connection() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS companies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                industry TEXT DEFAULT 'Telecom',
                contact_email TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS dataset_uploads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                company_id INTEGER NOT NULL,
                filename TEXT NOT NULL,
                stored_path TEXT NOT NULL,
                row_count INTEGER DEFAULT 0,
                mapping_json TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(company_id) REFERENCES companies(id)
            );

            CREATE TABLE IF NOT EXISTS customers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                company_id INTEGER NOT NULL,
                upload_id INTEGER,
                external_customer_id TEXT NOT NULL,
                gender TEXT,
                age INTEGER,
                tenure INTEGER,
                subscription_type TEXT,
                monthly_charges REAL,
                total_spend REAL,
                last_interaction_days INTEGER,
                feedback TEXT,
                actual_churn TEXT,
                sentiment REAL,
                churn_probability REAL,
                churn_prediction INTEGER,
                risk_score REAL,
                risk_category TEXT,
                health_score REAL,
                health_category TEXT,
                clv REAL,
                churn_reasons TEXT,
                next_best_action TEXT,
                offer_title TEXT,
                offer_details TEXT,
                last_analysis_at TEXT NOT NULL,
                UNIQUE(company_id, external_customer_id),
                FOREIGN KEY(company_id) REFERENCES companies(id),
                FOREIGN KEY(upload_id) REFERENCES dataset_uploads(id)
            );

            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                company_id INTEGER,
                customer_id INTEGER,
                source TEXT NOT NULL,
                churn_probability REAL,
                risk_score REAL,
                health_score REAL,
                recommendation_json TEXT,
                reasons_json TEXT,
                payload_json TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(company_id) REFERENCES companies(id),
                FOREIGN KEY(customer_id) REFERENCES customers(id)
            );

            CREATE TABLE IF NOT EXISTS retention_actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                company_id INTEGER NOT NULL,
                customer_id INTEGER NOT NULL,
                priority TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                action_type TEXT NOT NULL,
                offer_title TEXT,
                offer_details TEXT,
                status TEXT NOT NULL DEFAULT 'Open',
                due_date TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(company_id) REFERENCES companies(id),
                FOREIGN KEY(customer_id) REFERENCES customers(id)
            );
            """
        )


def load_artifacts() -> None:
    global model, scaler, feature_cols, label_encoders, seed_dataset, model_load_error
    try:
        model = joblib.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE) if SCALER_FILE.exists() else None
        feature_cols = joblib.load(FEATURE_LIST_FILE) if FEATURE_LIST_FILE.exists() else []
        label_encoders = joblib.load(LABEL_ENCODERS_FILE) if LABEL_ENCODERS_FILE.exists() else {}
        seed_dataset = pd.read_csv(DATA_FILE) if DATA_FILE.exists() else pd.DataFrame()
        model_load_error = None
    except Exception as exc:
        model = None
        scaler = None
        feature_cols = []
        label_encoders = {}
        seed_dataset = pd.read_csv(DATA_FILE) if DATA_FILE.exists() else pd.DataFrame()
        model_load_error = str(exc)


def normalize_churn(value: Any) -> int:
    return 1 if str(value).strip().lower() in {"yes", "1", "y", "true", "t", "churned"} else 0


def get_sentiment(text: Any) -> float:
    try:
        return round(TextBlob(str(text)).sentiment.polarity, 3)
    except Exception:
        return 0.0


def to_number(value: Any, default: float = 0) -> float:
    try:
        if pd.isna(value):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def to_int(value: Any, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return int(default)
        return int(float(value))
    except Exception:
        return int(default)


def encode_input(value: Any, column_name: str) -> Any:
    encoder = label_encoders.get(column_name)
    if encoder is None:
        return value
    try:
        return encoder.transform([value])[0]
    except Exception:
        return 0


def calculate_risk_score(churn_probability: float, last_interaction_days: int, tenure: int, monthly_charges: float, sentiment: float) -> float:
    base_risk = churn_probability * 100
    interaction_risk = min(last_interaction_days / 30 * 18, 18)
    tenure_risk = max(0, (12 - tenure) / 12 * 16)
    price_risk = min(monthly_charges / 1200 * 10, 10)
    sentiment_risk = max(0, abs(min(sentiment, 0)) * 20)
    return round(min(100, base_risk + interaction_risk + tenure_risk + price_risk + sentiment_risk), 2)


def get_risk_category(risk_score: float) -> str:
    if risk_score < 30:
        return "Low"
    if risk_score < 60:
        return "Medium"
    if risk_score < 80:
        return "High"
    return "Critical"


def calculate_health_score(churn_probability: float, tenure: int, last_interaction_days: int, sentiment: float) -> float:
    health = 100
    health -= churn_probability * 55
    health += min(tenure * 1.5, 18)
    health -= min(last_interaction_days, 45) * 0.75
    health += sentiment * 15
    return round(max(0, min(100, health)), 2)


def get_health_category(health_score: float) -> str:
    if health_score >= 75:
        return "Excellent"
    if health_score >= 55:
        return "Good"
    if health_score >= 35:
        return "Watch"
    return "Poor"


def estimate_clv(monthly_charges: float, tenure: int, churn_probability: float) -> float:
    expected_months = max(1, tenure + (1 - churn_probability) * 14)
    return round(monthly_charges * expected_months, 2)


def build_reasons(payload: dict[str, Any], churn_probability: float, sentiment: float) -> list[dict[str, str]]:
    reasons: list[dict[str, str]] = []
    tenure = to_int(payload.get("tenure"), 0)
    monthly_charges = to_number(payload.get("monthly_charges"), 0)
    last_interaction_days = to_int(payload.get("last_interaction_days"), 0)
    subscription = str(payload.get("subscription", "Standard"))

    if churn_probability >= 0.75:
        reasons.append({"factor": "Model risk", "detail": "The overall churn probability is already very high."})
    if last_interaction_days >= 21:
        reasons.append({"factor": "Low engagement", "detail": f"Customer has been inactive for {last_interaction_days} days."})
    if sentiment <= -0.2:
        reasons.append({"factor": "Negative sentiment", "detail": "Feedback indicates dissatisfaction or unresolved issues."})
    if tenure <= 6:
        reasons.append({"factor": "Early life-cycle", "detail": "Newer customers are at higher risk before habit formation."})
    if monthly_charges >= 800:
        reasons.append({"factor": "Price sensitivity", "detail": "High monthly cost may be creating value-for-money pressure."})
    if subscription.lower() == "basic":
        reasons.append({"factor": "Plan fit", "detail": "Basic-plan customers may feel under-served or easier to switch."})
    if not reasons:
        reasons.append({"factor": "Stable profile", "detail": "No major churn driver stands out from current signals."})
    return reasons[:5]


def build_offer_and_action(payload: dict[str, Any], risk_category: str, reasons: list[dict[str, str]]) -> dict[str, Any]:
    reason_names = {reason["factor"] for reason in reasons}
    subscription = str(payload.get("subscription", "Standard"))
    monthly_charges = to_number(payload.get("monthly_charges"), 0)

    if "Negative sentiment" in reason_names:
        return {
            "priority": "Critical" if risk_category in {"Critical", "High"} else "High",
            "action_type": "Service Recovery",
            "title": "Launch service recovery intervention",
            "description": "Open a rapid support recovery flow with a senior retention specialist.",
            "offer_title": "Apology credit + priority support",
            "offer_details": "Provide a one-time service credit, fast complaint resolution, and dedicated follow-up within 24 hours.",
            "next_best_action": "Call the customer, resolve the issue, and confirm recovery with a follow-up message.",
        }
    if "Low engagement" in reason_names:
        return {
            "priority": "High",
            "action_type": "Re-engagement",
            "title": "Run re-engagement journey",
            "description": "Customer is drifting away due to low activity.",
            "offer_title": "Usage booster pack",
            "offer_details": "Offer bonus data, free trial features, or a limited-period usage incentive to bring the customer back.",
            "next_best_action": "Send a reactivation campaign and schedule a check-in after 3 days.",
        }
    if "Price sensitivity" in reason_names:
        discount = "15%" if monthly_charges >= 1000 else "10%"
        return {
            "priority": "High",
            "action_type": "Retention Offer",
            "title": "Offer a value-protection plan",
            "description": "Customer may leave because current pricing feels expensive.",
            "offer_title": f"{discount} loyalty discount",
            "offer_details": f"Provide a {discount} retention discount or a bundled plan with extra benefits at the same price.",
            "next_best_action": "Present a price-match or bundle upgrade before the renewal cycle.",
        }
    if "Early life-cycle" in reason_names:
        return {
            "priority": "Medium",
            "action_type": "Onboarding",
            "title": "Strengthen onboarding experience",
            "description": "New customers need clearer value realization and engagement support.",
            "offer_title": "White-glove onboarding",
            "offer_details": "Give a guided setup session, feature education, and a success check-in within the first week.",
            "next_best_action": "Assign an onboarding specialist and send a personalized welcome plan.",
        }
    if subscription.lower() == "basic":
        return {
            "priority": "Medium",
            "action_type": "Upsell",
            "title": "Promote upgrade with incentives",
            "description": "A better-fitting plan can reduce churn and improve satisfaction.",
            "offer_title": "Free upgrade trial",
            "offer_details": "Offer a 30-day Standard or Premium trial with extra support and feature access.",
            "next_best_action": "Recommend a better plan based on usage and invite the customer into a limited trial.",
        }
    return {
        "priority": "Low",
        "action_type": "Nurture",
        "title": "Continue proactive relationship management",
        "description": "The customer looks relatively stable but still benefits from engagement.",
        "offer_title": "Loyalty points booster",
        "offer_details": "Provide referral credits, loyalty rewards, or exclusive access benefits.",
        "next_best_action": "Keep the customer warm with periodic success messaging and reward nudges.",
    }


def build_recommendations(offer_plan: dict[str, Any], reasons: list[dict[str, str]], churn_probability: float) -> list[dict[str, str]]:
    recommendations = [
        {
            "priority": offer_plan["priority"],
            "icon": "🎯",
            "title": offer_plan["title"],
            "description": offer_plan["description"],
            "action": offer_plan["next_best_action"],
        }
    ]
    for reason in reasons[:3]:
        recommendations.append(
            {
                "priority": "Medium",
                "icon": "🔎",
                "title": reason["factor"],
                "description": reason["detail"],
                "action": "Use this signal in the customer conversation and retention workflow.",
            }
        )
    if churn_probability >= 0.7:
        recommendations.append(
            {
                "priority": "Critical",
                "icon": "📞",
                "title": "Escalate to retention desk",
                "description": "This customer should be handled quickly due to high churn risk.",
                "action": "Open a task for manual outreach within the next 24 hours.",
            }
        )
    return recommendations


FEATURE_LABELS = {
    "Gender": "Gender",
    "Age": "Age",
    "Tenure": "Tenure",
    "Subscription_Type": "Subscription Type",
    "Monthly_Charges": "Monthly Charges",
    "Total_Spend": "Total Spend",
    "Last_Interaction_Days": "Last Interaction Gap",
    "Sentiment": "Customer Sentiment",
}


def humanize_feature_value(feature_name: str, payload: dict[str, Any], prepared_features: dict[str, Any]) -> str:
    if feature_name == "Age":
        return f"Age is {to_int(payload.get('age'), 35)}"
    if feature_name == "Tenure":
        return f"Tenure is {to_int(payload.get('tenure'), 12)} months"
    if feature_name == "Monthly_Charges":
        return f"Monthly charges are {to_number(payload.get('monthly_charges'), 499):.0f}"
    if feature_name == "Total_Spend":
        return f"Total spend is {to_number(payload.get('total_spend'), prepared_features.get('Total_Spend', 0)):.0f}"
    if feature_name == "Last_Interaction_Days":
        return f"Last interaction gap is {to_int(payload.get('last_interaction_days'), 7)} days"
    if feature_name == "Sentiment":
        return f"Feedback sentiment score is {prepared_features.get('Sentiment', 0):.2f}"
    if feature_name == "Subscription_Type":
        return f"Subscription type is {normalize_subscription(payload.get('subscription', 'Standard'))}"
    if feature_name == "Gender":
        return f"Customer gender is {str(payload.get('gender', 'Male')).title()}"
    return f"{FEATURE_LABELS.get(feature_name, feature_name)} is influencing the result"


def feature_reason_text(feature_name: str, payload: dict[str, Any], prepared_features: dict[str, Any], contribution: float) -> str:
    direction = "increases churn risk" if contribution >= 0 else "reduces churn risk"
    base_text = humanize_feature_value(feature_name, payload, prepared_features)
    return f"{base_text} and {direction}."


def extract_shap_contributions(shap_values: Any, feature_names: list[str]) -> list[float]:
    if isinstance(shap_values, list):
        if len(shap_values) > 1:
            return list(shap_values[1][0])
        return list(shap_values[0][0])

    if hasattr(shap_values, "values"):
        values = shap_values.values
        if getattr(values, "ndim", 0) == 3:
            return list(values[0, :, -1])
        if getattr(values, "ndim", 0) == 2:
            return list(values[0])

    if hasattr(shap_values, "ndim"):
        if shap_values.ndim == 3:
            return list(shap_values[0, :, -1])
        if shap_values.ndim == 2:
            return list(shap_values[0])

    return [0.0 for _ in feature_names]


def build_xai_explanation(
    payload: dict[str, Any],
    prepared_features: dict[str, Any],
    model_input_df: pd.DataFrame,
    model_input: Any,
) -> dict[str, Any]:
    feature_names = list(model_input_df.columns)
    explanation_method = "feature_importance"
    contribution_map: dict[str, float] = {}

    global shap_explainer
    if shap is not None:
        try:
            if shap_explainer is None:
                shap_explainer = shap.TreeExplainer(model)
            shap_values = shap_explainer.shap_values(model_input)
            contributions = extract_shap_contributions(shap_values, feature_names)
            contribution_map = {feature: float(value) for feature, value in zip(feature_names, contributions)}
            explanation_method = "shap"
        except Exception:
            contribution_map = {}

    if not contribution_map:
        importances = getattr(model, "feature_importances_", None)
        if importances is None:
            importances = [0.0 for _ in feature_names]
        contribution_map = {feature: float(value) for feature, value in zip(feature_names, importances)}
        explanation_method = "feature_importance"

    ranked = sorted(contribution_map.items(), key=lambda item: abs(item[1]), reverse=True)[:5]
    top_features = []
    for feature_name, contribution in ranked:
        top_features.append(
            {
                "feature": feature_name,
                "label": FEATURE_LABELS.get(feature_name, feature_name),
                "contribution": round(float(contribution), 4),
                "direction": "Increase Risk" if contribution >= 0 else "Reduce Risk",
                "detail": feature_reason_text(feature_name, payload, prepared_features, contribution),
            }
        )

    return {
        "method": explanation_method,
        "top_features": top_features,
    }


def normalize_subscription(value: Any) -> str:
    text = str(value or "Standard").strip()
    return text.title() if text else "Standard"


def prepare_model_input(payload: dict[str, Any]) -> dict[str, Any]:
    age = to_int(payload.get("age"), 35)
    tenure = to_int(payload.get("tenure"), 12)
    monthly_charges = to_number(payload.get("monthly_charges"), 499)
    last_interaction_days = to_int(payload.get("last_interaction_days"), 7)
    feedback = payload.get("feedback", "")
    subscription = normalize_subscription(payload.get("subscription", "Standard"))
    gender = str(payload.get("gender", "Male")).title()
    total_spend = to_number(payload.get("total_spend"), monthly_charges * max(tenure, 1))
    return {
        "Gender": encode_input(gender, "Gender"),
        "Age": age,
        "Tenure": tenure,
        "Subscription_Type": encode_input(subscription, "Subscription_Type"),
        "Monthly_Charges": monthly_charges,
        "Total_Spend": total_spend,
        "Last_Interaction_Days": last_interaction_days,
        "Sentiment": get_sentiment(feedback),
    }


def run_prediction(payload: dict[str, Any]) -> dict[str, Any]:
    if model is None:
        raise RuntimeError(model_load_error or "Model files are not available.")
    features = prepare_model_input(payload)
    model_input_df = pd.DataFrame([features])
    if feature_cols:
        for column in feature_cols:
            if column not in model_input_df.columns:
                model_input_df[column] = 0
        model_input_df = model_input_df[feature_cols]
    model_input = scaler.transform(model_input_df) if scaler is not None else model_input_df
    prediction = int(model.predict(model_input)[0])
    churn_probability = float(model.predict_proba(model_input)[0][1])
    sentiment = features["Sentiment"]
    tenure = to_int(payload.get("tenure"), 12)
    monthly_charges = to_number(payload.get("monthly_charges"), 499)
    last_interaction_days = to_int(payload.get("last_interaction_days"), 7)
    risk_score = calculate_risk_score(churn_probability, last_interaction_days, tenure, monthly_charges, sentiment)
    health_score = calculate_health_score(churn_probability, tenure, last_interaction_days, sentiment)
    risk_category = get_risk_category(risk_score)
    xai_explanation = build_xai_explanation(payload, features, model_input_df, model_input)
    reasons = build_reasons(payload, churn_probability, sentiment)
    for item in xai_explanation["top_features"][:3]:
        reasons.append(
            {
                "factor": item["label"],
                "detail": item["detail"],
            }
        )
    deduped_reasons = []
    seen = set()
    for reason in reasons:
        key = (reason["factor"], reason["detail"])
        if key not in seen:
            deduped_reasons.append(reason)
            seen.add(key)
    reasons = deduped_reasons[:6]
    offer_plan = build_offer_and_action(payload, risk_category, reasons)
    recommendations = build_recommendations(offer_plan, reasons, churn_probability)
    return {
        "churn": bool(prediction),
        "churn_probability": round(churn_probability, 4),
        "stay_probability": round(1 - churn_probability, 4),
        "risk_score": risk_score,
        "risk_category": risk_category,
        "health_score": health_score,
        "health_category": get_health_category(health_score),
        "clv": estimate_clv(monthly_charges, tenure, churn_probability),
        "sentiment": sentiment,
        "reasons": reasons,
        "xai_explanation": xai_explanation,
        "offer_plan": offer_plan,
        "recommendations": recommendations,
    }


def get_companies() -> list[sqlite3.Row]:
    with get_connection() as connection:
        return connection.execute("SELECT * FROM companies ORDER BY name").fetchall()


def get_or_create_company(name: str, industry: str = "Telecom") -> sqlite3.Row:
    normalized_name = name.strip()
    if not normalized_name:
        raise ValueError("Company name is required.")
    with get_connection() as connection:
        existing = connection.execute("SELECT * FROM companies WHERE lower(name) = lower(?)", (normalized_name,)).fetchone()
        if existing:
            return existing
        connection.execute(
            "INSERT INTO companies (name, industry, created_at) VALUES (?, ?, ?)",
            (normalized_name, industry or "Telecom", utcnow_iso()),
        )
        connection.commit()
        return connection.execute("SELECT * FROM companies WHERE lower(name) = lower(?)", (normalized_name,)).fetchone()


def get_company(company_id: int | None) -> sqlite3.Row | None:
    if company_id is None:
        return None
    with get_connection() as connection:
        return connection.execute("SELECT * FROM companies WHERE id = ?", (company_id,)).fetchone()


def record_dataset_upload(company_id: int, original_filename: str, stored_path: str, row_count: int, mapping: dict[str, str]) -> int:
    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO dataset_uploads (company_id, filename, stored_path, row_count, mapping_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (company_id, original_filename, stored_path, row_count, json.dumps(mapping), utcnow_iso()),
        )
        connection.commit()
        return int(cursor.lastrowid)


def store_prediction_history(company_id: int | None, customer_id: int | None, source: str, prediction: dict[str, Any], payload: dict[str, Any]) -> None:
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO prediction_history (
                company_id, customer_id, source, churn_probability, risk_score, health_score,
                recommendation_json, reasons_json, payload_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                company_id,
                customer_id,
                source,
                prediction["churn_probability"],
                prediction["risk_score"],
                prediction["health_score"],
                json.dumps(prediction["recommendations"]),
                json.dumps(prediction["reasons"]),
                json.dumps(payload),
                utcnow_iso(),
            ),
        )
        connection.commit()


def upsert_customer(company_id: int, upload_id: int | None, payload: dict[str, Any], prediction: dict[str, Any]) -> int:
    external_customer_id = str(payload.get("customer_id") or f"CUST-{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}")
    reasons_json = json.dumps(prediction["reasons"])
    offer_plan = prediction["offer_plan"]
    with get_connection() as connection:
        existing = connection.execute(
            "SELECT id FROM customers WHERE company_id = ? AND external_customer_id = ?",
            (company_id, external_customer_id),
        ).fetchone()
        base_params = (
            str(payload.get("gender", "Male")).title(),
            to_int(payload.get("age"), 35),
            to_int(payload.get("tenure"), 12),
            normalize_subscription(payload.get("subscription", "Standard")),
            to_number(payload.get("monthly_charges"), 499),
            to_number(payload.get("total_spend"), to_number(payload.get("monthly_charges"), 499) * max(to_int(payload.get("tenure"), 12), 1)),
            to_int(payload.get("last_interaction_days"), 7),
            str(payload.get("feedback", "")),
            str(payload.get("actual_churn", "")),
            prediction["sentiment"],
            prediction["churn_probability"],
            int(prediction["churn"]),
            prediction["risk_score"],
            prediction["risk_category"],
            prediction["health_score"],
            prediction["health_category"],
            prediction["clv"],
            reasons_json,
            offer_plan["next_best_action"],
            offer_plan["offer_title"],
            offer_plan["offer_details"],
            utcnow_iso(),
        )
        update_params = (upload_id,) + base_params
        if existing:
            customer_id = int(existing["id"])
            connection.execute(
                """
                UPDATE customers
                SET upload_id = ?, gender = ?, age = ?, tenure = ?, subscription_type = ?,
                    monthly_charges = ?, total_spend = ?, last_interaction_days = ?, feedback = ?, actual_churn = ?,
                    sentiment = ?, churn_probability = ?, churn_prediction = ?, risk_score = ?, risk_category = ?,
                    health_score = ?, health_category = ?, clv = ?, churn_reasons = ?, next_best_action = ?,
                    offer_title = ?, offer_details = ?, last_analysis_at = ?
                WHERE id = ?
                """,
                update_params + (customer_id,),
            )
        else:
            cursor = connection.execute(
                """
                INSERT INTO customers (
                    company_id, upload_id, external_customer_id, gender, age, tenure, subscription_type,
                    monthly_charges, total_spend, last_interaction_days, feedback, actual_churn, sentiment,
                    churn_probability, churn_prediction, risk_score, risk_category, health_score, health_category,
                    clv, churn_reasons, next_best_action, offer_title, offer_details, last_analysis_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (company_id, upload_id, external_customer_id) + base_params,
            )
            customer_id = int(cursor.lastrowid)
        connection.commit()
    return customer_id


def replace_retention_actions(company_id: int, customer_id: int, prediction: dict[str, Any]) -> None:
    offer_plan = prediction["offer_plan"]
    due_date = (datetime.utcnow() + timedelta(days=1 if prediction["risk_category"] in {"Critical", "High"} else 3)).date().isoformat()
    with get_connection() as connection:
        connection.execute("DELETE FROM retention_actions WHERE company_id = ? AND customer_id = ?", (company_id, customer_id))
        connection.execute(
            """
            INSERT INTO retention_actions (
                company_id, customer_id, priority, title, description, action_type,
                offer_title, offer_details, due_date, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                company_id,
                customer_id,
                offer_plan["priority"],
                offer_plan["title"],
                offer_plan["description"],
                offer_plan["action_type"],
                offer_plan["offer_title"],
                offer_plan["offer_details"],
                due_date,
                utcnow_iso(),
            ),
        )
        connection.commit()


def row_to_payload(row: pd.Series) -> dict[str, Any]:
    return {
        "customer_id": str(row.get("Customer_ID", "")),
        "gender": row.get("Gender", "Male"),
        "age": row.get("Age", 35),
        "tenure": row.get("Tenure", 12),
        "subscription": row.get("Subscription_Type", "Standard"),
        "monthly_charges": row.get("Monthly_Charges", 499),
        "total_spend": row.get("Total_Spend", to_number(row.get("Monthly_Charges"), 499) * max(to_int(row.get("Tenure"), 12), 1)),
        "last_interaction_days": row.get("Last_Interaction_Days", 7),
        "feedback": row.get("Feedback", ""),
        "actual_churn": row.get("Churn", ""),
    }


def canonicalize_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    normalized_map = {column: str(column).strip().lower().replace(" ", "_") for column in df.columns}
    rename_map: dict[str, str] = {}
    for canonical, aliases in CANONICAL_COLUMNS.items():
        alias_set = {alias.lower() for alias in aliases}
        for original, normalized in normalized_map.items():
            if normalized == canonical.lower() or normalized in alias_set:
                rename_map[original] = canonical
                break
    transformed = df.rename(columns=rename_map).copy()
    if "Customer_ID" not in transformed.columns:
        transformed["Customer_ID"] = [f"CUST-{index + 1:05d}" for index in range(len(transformed))]
    if "Feedback" not in transformed.columns:
        transformed["Feedback"] = ""
    if "Gender" not in transformed.columns:
        transformed["Gender"] = "Male"
    if "Subscription_Type" not in transformed.columns:
        transformed["Subscription_Type"] = "Standard"
    if "Last_Interaction_Days" not in transformed.columns:
        transformed["Last_Interaction_Days"] = 7
    if "Total_Spend" not in transformed.columns and {"Monthly_Charges", "Tenure"}.issubset(set(transformed.columns)):
        transformed["Total_Spend"] = transformed["Monthly_Charges"].fillna(0) * transformed["Tenure"].fillna(0)
    required = ["Age", "Tenure", "Monthly_Charges"]
    missing = [column for column in required if column not in transformed.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")
    return transformed, {value: key for key, value in rename_map.items()}


def save_uploaded_file(uploaded_file) -> Path:
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    safe_name = Path(uploaded_file.name).name.replace(" ", "_")
    stored_path = UPLOADS_DIR / f"{timestamp}_{safe_name}"
    stored_path.write_bytes(uploaded_file.getbuffer())
    return stored_path


def process_uploaded_dataset(company_id: int, upload_id: int, df: pd.DataFrame) -> dict[str, Any]:
    transformed, mapping = canonicalize_columns(df)
    rows = []
    risk_counts = {"Low": 0, "Medium": 0, "High": 0, "Critical": 0}
    for _, row in transformed.iterrows():
        payload = row_to_payload(row)
        prediction = run_prediction(payload)
        customer_id = upsert_customer(company_id, upload_id, payload, prediction)
        replace_retention_actions(company_id, customer_id, prediction)
        store_prediction_history(company_id, customer_id, "dataset_upload", prediction, payload)
        risk_counts[prediction["risk_category"]] = risk_counts.get(prediction["risk_category"], 0) + 1
        rows.append(
            {
                "Customer_ID": payload["customer_id"],
                "Risk_Category": prediction["risk_category"],
                "Churn_Probability": round(prediction["churn_probability"] * 100, 2),
                "Risk_Score": prediction["risk_score"],
                "Health_Score": prediction["health_score"],
                "Tenure": payload["tenure"],
                "Subscription_Type": payload["subscription"],
                "Top_Factor": prediction["xai_explanation"]["top_features"][0]["label"] if prediction["xai_explanation"]["top_features"] else "N/A",
                "Offer_Title": prediction["offer_plan"]["offer_title"],
                "Next_Best_Action": prediction["offer_plan"]["next_best_action"],
            }
        )
    return {"rows": rows, "mapping": mapping, "risk_distribution": risk_counts, "row_count": len(rows)}


def process_uploaded_file(company_name: str, industry: str, uploaded_file) -> dict[str, Any]:
    company = get_or_create_company(company_name, industry)
    stored_path = save_uploaded_file(uploaded_file)
    if stored_path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(stored_path)
    else:
        df = pd.read_csv(stored_path)
    transformed, mapping = canonicalize_columns(df)
    upload_id = record_dataset_upload(company["id"], uploaded_file.name, str(stored_path), len(transformed), mapping)
    results = process_uploaded_dataset(company["id"], upload_id, transformed)
    return {"company": dict(company), **results}


def get_platform_summary(company_id: int | None = None) -> dict[str, Any]:
    where_clause = ""
    params: tuple[Any, ...] = ()
    if company_id:
        where_clause = "WHERE company_id = ?"
        params = (company_id,)
    with get_connection() as connection:
        companies_count = connection.execute("SELECT COUNT(*) AS count FROM companies").fetchone()["count"]
        customers_count = connection.execute(f"SELECT COUNT(*) AS count FROM customers {where_clause}", params).fetchone()["count"]
        if company_id:
            high_risk_count = connection.execute(
                "SELECT COUNT(*) AS count FROM customers WHERE company_id = ? AND risk_category IN ('High', 'Critical')",
                (company_id,),
            ).fetchone()["count"]
        else:
            high_risk_count = connection.execute(
                "SELECT COUNT(*) AS count FROM customers WHERE risk_category IN ('High', 'Critical')"
            ).fetchone()["count"]
        avg_churn = connection.execute(f"SELECT AVG(churn_probability) AS avg_value FROM customers {where_clause}", params).fetchone()["avg_value"]
        open_actions = connection.execute(
            "SELECT COUNT(*) AS count FROM retention_actions WHERE status = 'Open'" + (" AND company_id = ?" if company_id else ""),
            params,
        ).fetchone()["count"]
    return {
        "companies_count": int(companies_count or 0),
        "customers_count": int(customers_count or 0),
        "high_risk_count": int(high_risk_count or 0),
        "avg_churn_probability": round(float(avg_churn or 0) * 100, 2),
        "open_actions": int(open_actions or 0),
    }


def get_recent_uploads(limit: int = 6) -> list[sqlite3.Row]:
    with get_connection() as connection:
        return connection.execute(
            """
            SELECT dataset_uploads.*, companies.name AS company_name
            FROM dataset_uploads
            JOIN companies ON companies.id = dataset_uploads.company_id
            ORDER BY dataset_uploads.created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()


def get_top_customers(company_id: int | None = None, limit: int = 12) -> list[sqlite3.Row]:
    query = """
        SELECT customers.*, companies.name AS company_name
        FROM customers
        JOIN companies ON companies.id = customers.company_id
    """
    params: list[Any] = []
    if company_id:
        query += " WHERE customers.company_id = ?"
        params.append(company_id)
    query += " ORDER BY customers.churn_probability DESC, customers.risk_score DESC LIMIT ?"
    params.append(limit)
    with get_connection() as connection:
        return connection.execute(query, tuple(params)).fetchall()


def get_retention_actions(company_id: int | None = None, limit: int = 20) -> list[sqlite3.Row]:
    query = """
        SELECT retention_actions.*, customers.external_customer_id, companies.name AS company_name
        FROM retention_actions
        JOIN customers ON customers.id = retention_actions.customer_id
        JOIN companies ON companies.id = retention_actions.company_id
    """
    params: list[Any] = []
    if company_id:
        query += " WHERE retention_actions.company_id = ?"
        params.append(company_id)
    query += " ORDER BY CASE retention_actions.priority WHEN 'Critical' THEN 1 WHEN 'High' THEN 2 WHEN 'Medium' THEN 3 ELSE 4 END, retention_actions.created_at DESC LIMIT ?"
    params.append(limit)
    with get_connection() as connection:
        return connection.execute(query, tuple(params)).fetchall()


def get_company_overview(company_id: int | None) -> dict[str, Any]:
    if company_id is None:
        return {"company": None}
    with get_connection() as connection:
        company = connection.execute("SELECT * FROM companies WHERE id = ?", (company_id,)).fetchone()
        if company is None:
            return {"company": None}
        customer_count = connection.execute("SELECT COUNT(*) AS count FROM customers WHERE company_id = ?", (company_id,)).fetchone()["count"]
        avg_churn = connection.execute("SELECT AVG(churn_probability) AS avg_value FROM customers WHERE company_id = ?", (company_id,)).fetchone()["avg_value"]
        avg_health = connection.execute("SELECT AVG(health_score) AS avg_value FROM customers WHERE company_id = ?", (company_id,)).fetchone()["avg_value"]
        high_risk = connection.execute(
            "SELECT COUNT(*) AS count FROM customers WHERE company_id = ? AND risk_category IN ('High', 'Critical')",
            (company_id,),
        ).fetchone()["count"]
        open_actions = connection.execute(
            "SELECT COUNT(*) AS count FROM retention_actions WHERE company_id = ? AND status = 'Open'",
            (company_id,),
        ).fetchone()["count"]
    return {
        "company": dict(company),
        "customer_count": int(customer_count or 0),
        "avg_churn_probability": round(float(avg_churn or 0) * 100, 2),
        "avg_health_score": round(float(avg_health or 0), 2),
        "high_risk_count": int(high_risk or 0),
        "open_actions": int(open_actions or 0),
    }


def grouped_metric(company_id: int | None, column: str, label_key: str) -> list[dict[str, Any]]:
    with get_connection() as connection:
        query = f"SELECT {column} AS label, AVG(churn_probability) AS avg_probability FROM customers"
        params: list[Any] = []
        if company_id:
            query += " WHERE company_id = ?"
            params.append(company_id)
        query += f" GROUP BY {column} ORDER BY avg_probability DESC"
        rows = connection.execute(query, tuple(params)).fetchall()
    return [{label_key: row["label"] or "Unknown", "rate": round(float(row["avg_probability"] or 0) * 100, 2)} for row in rows]


def risk_distribution(company_id: int | None) -> list[dict[str, Any]]:
    with get_connection() as connection:
        query = "SELECT risk_category, COUNT(*) AS count FROM customers"
        params: list[Any] = []
        if company_id:
            query += " WHERE company_id = ?"
            params.append(company_id)
        query += " GROUP BY risk_category"
        rows = connection.execute(query, tuple(params)).fetchall()
    return [{"category": row["risk_category"] or "Unknown", "count": row["count"]} for row in rows]


def run_manual_prediction(payload: dict[str, Any], company_id: int | None = None) -> dict[str, Any]:
    prediction = run_prediction(payload)
    customer_id = None
    if company_id:
        customer_id = upsert_customer(company_id, None, payload, prediction)
        replace_retention_actions(company_id, customer_id, prediction)
    store_prediction_history(company_id, customer_id, "manual_predict", prediction, payload)
    return prediction


def seed_platform_company() -> None:
    if get_companies() or seed_dataset.empty or model is None:
        return
    company = get_or_create_company("Demo Telecom", "Telecom")
    upload_id = record_dataset_upload(company["id"], "seed_dataset.csv", str(DATA_FILE), min(len(seed_dataset), 50), {"source": "seed"})
    for _, row in seed_dataset.head(30).iterrows():
        payload = row_to_payload(row)
        prediction = run_prediction(payload)
        customer_id = upsert_customer(company["id"], upload_id, payload, prediction)
        replace_retention_actions(company["id"], customer_id, prediction)
        store_prediction_history(company["id"], customer_id, "seed", prediction, payload)


def initialize_platform() -> None:
    init_db()
    load_artifacts()
    if model is not None:
        seed_platform_company()
