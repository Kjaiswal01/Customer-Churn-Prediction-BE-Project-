from __future__ import annotations

import numpy as np
import pandas as pd
try:
    from textblob import TextBlob
except Exception:  # pragma: no cover
    TextBlob = None

from enterprise_config import DATA_DIR


RAW_DATASET_PATH = DATA_DIR / "customer_churn_raw.csv"
CLEAN_DATASET_PATH = DATA_DIR / "customer_churn.csv"

POSITIVE_WORDS = {
    "great",
    "good",
    "helpful",
    "quick",
    "excellent",
    "satisfied",
    "happy",
    "smooth",
    "reliable",
    "love",
    "friendly",
    "responsive",
    "stable",
    "fast",
    "resolved",
    "affordable",
    "worth",
}
NEGATIVE_WORDS = {
    "poor",
    "bad",
    "delay",
    "expensive",
    "issue",
    "issues",
    "drop",
    "dropped",
    "slow",
    "unhappy",
    "frustrated",
    "disconnect",
    "outage",
    "billing",
    "network",
    "support",
    "confusing",
    "poorly",
    "unresolved",
    "angry",
    "cancel",
    "switch",
    "worst",
    "terrible",
    "unreliable",
    "overpriced",
    "hidden",
    "unresponsive",
    "dissatisfied",
    "broken",
}
INTENSIFIERS = {"very": 1.2, "too": 1.2, "extremely": 1.4, "really": 1.15}
NEGATIONS = {"not", "never", "no", "hardly", "barely", "isnt", "wasnt", "dont", "didnt", "cant"}

PRICING_PATTERNS = ("price", "pricing", "bill", "billing", "expensive", "cost", "charge", "charges")
SERVICE_PATTERNS = ("support", "service", "response", "agent", "help", "onboarding", "resolution")
NETWORK_PATTERNS = ("network", "signal", "speed", "outage", "disconnect", "connectivity", "latency")


def simple_sentiment_score(text: object) -> float:
    normalized_text = str(text or "").strip()
    tokens = (
        normalized_text.lower()
        .replace(",", " ")
        .replace(".", " ")
        .replace("!", " ")
        .replace("?", " ")
        .replace("'", "")
        .split()
    )
    if not tokens:
        return 0.0

    weighted_hits = 0.0
    for index, token in enumerate(tokens):
        window = tokens[max(0, index - 2):index]
        modifier = max((INTENSIFIERS[item] for item in window if item in INTENSIFIERS), default=1.0)
        is_negated = any(item in NEGATIONS for item in window)
        if token in POSITIVE_WORDS:
            weighted_hits += (-1.0 if is_negated else 1.0) * modifier
        elif token in NEGATIVE_WORDS:
            weighted_hits += (1.0 if is_negated else -1.0) * modifier

    phrase_penalties = 0.0
    lower_text = " ".join(tokens)
    if "too expensive" in lower_text or "very expensive" in lower_text:
        phrase_penalties -= 1.4
    if "poor support" in lower_text or "slow support" in lower_text:
        phrase_penalties -= 1.2
    if "network issue" in lower_text or "network issues" in lower_text:
        phrase_penalties -= 1.1
    if "not happy" in lower_text or "not satisfied" in lower_text:
        phrase_penalties -= 1.25
    if "very happy" in lower_text or "really good" in lower_text:
        phrase_penalties += 1.1

    lexicon_score = (weighted_hits + phrase_penalties) / max(len(tokens), 1)
    intensity = max((INTENSIFIERS[token] for token in tokens if token in INTENSIFIERS), default=1.0)
    if TextBlob is not None:
        try:
            blob_score = float(TextBlob(normalized_text).sentiment.polarity)
        except Exception:
            blob_score = 0.0
    else:
        blob_score = 0.0
    score = (lexicon_score * 0.62 * intensity) + (blob_score * 0.38)
    return float(np.clip(round(score, 3), -1, 1))


def classify_issue(text: object) -> str:
    normalized = str(text or "").lower()
    has_positive = any(word in normalized for word in POSITIVE_WORDS)
    has_negative = any(word in normalized for word in NEGATIVE_WORDS)
    if has_positive and not has_negative:
        return "General Feedback"
    if any(pattern in normalized for pattern in PRICING_PATTERNS):
        return "Pricing Issue"
    if any(pattern in normalized for pattern in NETWORK_PATTERNS):
        return "Network Issue"
    if any(pattern in normalized for pattern in SERVICE_PATTERNS):
        return "Service Issue"
    return "General Feedback"


def normalize_gender(value: object) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"m", "male", "man"}:
        return "Male"
    if normalized in {"f", "female", "woman"}:
        return "Female"
    return "Female"


def normalize_plan(value: object) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"basic", "starter", "entry"}:
        return "Basic"
    if normalized in {"standard", "pro", "plus"}:
        return "Standard"
    if normalized in {"premium", "gold", "advanced"}:
        return "Premium"
    if normalized in {"enterprise", "platinum"}:
        return "Enterprise"
    return "Standard"


def normalize_target(value: object) -> str:
    return "Yes" if str(value or "").strip().lower() in {"1", "yes", "true", "y", "churned"} else "No"


def create_raw_dataset(row_count: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    plans = np.array(["Basic", "Standard", "Premium", "Enterprise"])
    plan_weights = np.array([0.36, 0.31, 0.22, 0.11])
    plan_prices = {"Basic": 349, "Standard": 599, "Premium": 899, "Enterprise": 1299}
    contract_types = np.array(["Monthly", "Quarterly", "Annual"])
    payment_modes = np.array(["Card", "UPI", "NetBanking", "AutoPay"])
    positive_feedback = [
        "Great support and reliable network experience",
        "Happy with the service quality and quick response",
        "Smooth onboarding and very helpful account manager",
        "Satisfied with pricing and product value",
        "Excellent issue resolution and friendly support team",
    ]
    pricing_feedback = [
        "Billing feels expensive for the value received",
        "Monthly charges are too high and pricing is confusing",
        "Price increase without enough added features",
        "The plan cost feels high compared with competitors",
    ]
    service_feedback = [
        "Support response is slow and issue resolution takes too long",
        "Customer service quality dropped after onboarding",
        "Agents are not proactive and service follow-up is poor",
        "Frustrated with repeated support delays",
    ]
    network_feedback = [
        "Frequent network outages and unstable connectivity",
        "Internet speed dropped and network quality is poor",
        "Signal issues keep affecting daily usage",
        "Too many disconnects and latency problems",
    ]

    rows: list[dict[str, object]] = []
    for index in range(1, row_count + 1):
        plan = str(rng.choice(plans, p=plan_weights))
        age = int(rng.integers(21, 71))
        tenure = int(rng.integers(1, 73))
        usage_score = float(np.clip(rng.normal(62, 18), 5, 100))
        support_tickets = int(np.clip(rng.poisson(1.8), 0, 8))
        payment_delay_days = int(np.clip(rng.normal(4, 6), 0, 45))
        last_interaction_days = int(np.clip(rng.normal(18, 14), 0, 120))
        contract_type = str(rng.choice(contract_types, p=[0.48, 0.23, 0.29]))
        payment_mode = str(rng.choice(payment_modes, p=[0.28, 0.26, 0.16, 0.30]))
        monthly_charges = float(
            np.clip(plan_prices[plan] + rng.normal(0, 120) + (support_tickets * 12), 199, 2499)
        )

        complaint_roll = rng.random()
        if complaint_roll < 0.24:
            feedback = str(rng.choice(pricing_feedback))
            issue_category = "Pricing Issue"
        elif complaint_roll < 0.49:
            feedback = str(rng.choice(service_feedback))
            issue_category = "Service Issue"
        elif complaint_roll < 0.66:
            feedback = str(rng.choice(network_feedback))
            issue_category = "Network Issue"
        else:
            feedback = str(rng.choice(positive_feedback))
            issue_category = "General Feedback"

        sentiment = simple_sentiment_score(feedback)
        engagement_score = float(
            np.clip((usage_score * 0.55) + (100 - last_interaction_days) * 0.25 + tenure * 0.35, 1, 100)
        )
        total_spend = round(monthly_charges * tenure, 2)
        risk_indicator = round(
            0.36 * monthly_charges
            + 0.24 * last_interaction_days
            + 18 * support_tickets
            + 0.55 * payment_delay_days
            - 0.5 * usage_score
            - 0.18 * tenure,
            2,
        )

        churn_score = (
            -1.25
            + (monthly_charges / 1200)
            + (last_interaction_days / 50)
            + (support_tickets * 0.26)
            + (payment_delay_days / 20)
            - (tenure / 45)
            - (usage_score / 120)
            - (engagement_score / 140)
            - (0.9 * sentiment)
        )
        if issue_category == "Pricing Issue":
            churn_score += 0.35
        if issue_category == "Network Issue":
            churn_score += 0.22
        if contract_type == "Annual":
            churn_score -= 0.28
        if payment_mode == "AutoPay":
            churn_score -= 0.12

        churn_probability = 1 / (1 + np.exp(-churn_score))
        churn_probability = float(np.clip(churn_probability + rng.normal(0, 0.06), 0.01, 0.99))
        churn = "Yes" if rng.random() < churn_probability else "No"

        rows.append(
            {
                "Customer_ID": f"CUST-{index:05d}",
                "Email": f"customer{index:05d}@example.com",
                "Gender": str(rng.choice(["Male", "Female"], p=[0.53, 0.47])),
                "Age": age,
                "Tenure": tenure,
                "Subscription_Type": plan,
                "Contract_Type": contract_type,
                "Payment_Mode": payment_mode,
                "Monthly_Charges": round(monthly_charges, 2),
                "Total_Spend": total_spend,
                "Usage_Score": round(usage_score, 2),
                "Support_Tickets": support_tickets,
                "Payment_Delay_Days": payment_delay_days,
                "Last_Interaction_Days": last_interaction_days,
                "Engagement_Score": round(engagement_score, 2),
                "Risk_Indicator": risk_indicator,
                "Feedback": feedback,
                "Sentiment": sentiment,
                "Issue_Category": issue_category,
                "Churn": churn,
            }
        )

    raw_df = pd.DataFrame(rows)
    row_numbers = pd.Series(np.arange(len(raw_df)))
    raw_df.loc[row_numbers % 13 == 0, "Gender"] = raw_df.loc[row_numbers % 13 == 0, "Gender"].str.lower()
    raw_df.loc[row_numbers % 17 == 0, "Subscription_Type"] = raw_df.loc[row_numbers % 17 == 0, "Subscription_Type"].replace(
        {"Basic": "starter", "Standard": "pro", "Premium": "gold", "Enterprise": "platinum"}
    )
    raw_df.loc[row_numbers % 19 == 0, "Feedback"] = None
    raw_df.loc[row_numbers % 23 == 0, "Monthly_Charges"] = np.nan
    raw_df.loc[row_numbers % 29 == 0, "Usage_Score"] = np.nan
    raw_df.loc[row_numbers % 31 == 0, "Payment_Delay_Days"] = np.nan
    raw_df.loc[row_numbers % 37 == 0, "Age"] = np.nan
    raw_df.loc[row_numbers % 41 == 0, "Total_Spend"] = np.nan
    return raw_df


def clean_customer_dataset(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [column.strip() for column in cleaned.columns]

    if "Customer_ID" not in cleaned.columns:
        cleaned["Customer_ID"] = [f"CUST-{index + 1:05d}" for index in range(len(cleaned))]
    if "Email" not in cleaned.columns:
        cleaned["Email"] = cleaned["Customer_ID"].astype(str).str.lower() + "@example.com"

    cleaned["Gender"] = cleaned.get("Gender", "Female").apply(normalize_gender)
    cleaned["Subscription_Type"] = cleaned.get("Subscription_Type", "Standard").apply(normalize_plan)
    cleaned["Feedback"] = cleaned.get("Feedback", "").fillna("Customer reported average experience with no detailed feedback.")

    numeric_defaults = {
        "Age": 38,
        "Tenure": 12,
        "Monthly_Charges": 649,
        "Usage_Score": 58,
        "Support_Tickets": 1,
        "Payment_Delay_Days": 3,
        "Last_Interaction_Days": 14,
    }
    for column, default in numeric_defaults.items():
        cleaned[column] = pd.to_numeric(cleaned.get(column, default), errors="coerce").fillna(default)

    cleaned["Age"] = cleaned["Age"].clip(18, 80).round().astype(int)
    cleaned["Tenure"] = cleaned["Tenure"].clip(1, 120).round().astype(int)
    cleaned["Monthly_Charges"] = cleaned["Monthly_Charges"].clip(199, 2500).round(2)
    cleaned["Usage_Score"] = cleaned["Usage_Score"].clip(0, 100).round(2)
    cleaned["Support_Tickets"] = cleaned["Support_Tickets"].clip(0, 12).round().astype(int)
    cleaned["Payment_Delay_Days"] = cleaned["Payment_Delay_Days"].clip(0, 60).round().astype(int)
    cleaned["Last_Interaction_Days"] = cleaned["Last_Interaction_Days"].clip(0, 180).round().astype(int)

    cleaned["Contract_Type"] = cleaned.get("Contract_Type", "Monthly").fillna("Monthly").astype(str).str.title()
    cleaned["Payment_Mode"] = cleaned.get("Payment_Mode", "Card").fillna("Card").astype(str)
    cleaned["Total_Spend"] = pd.to_numeric(cleaned.get("Total_Spend"), errors="coerce")
    cleaned["Total_Spend"] = cleaned["Total_Spend"].fillna(cleaned["Monthly_Charges"] * cleaned["Tenure"]).round(2)

    sentiment_series = pd.to_numeric(cleaned.get("Sentiment"), errors="coerce")
    derived_sentiment = cleaned["Feedback"].apply(simple_sentiment_score).astype(float)
    cleaned["Sentiment"] = sentiment_series.fillna(derived_sentiment).astype(float).clip(-1, 1).round(3)

    cleaned["Issue_Category"] = cleaned.get("Issue_Category")
    cleaned["Issue_Category"] = cleaned["Issue_Category"].fillna(cleaned["Feedback"].apply(classify_issue))

    cleaned["Engagement_Score"] = (
        cleaned["Usage_Score"] * 0.55
        + (100 - cleaned["Last_Interaction_Days"].clip(0, 100)) * 0.25
        + cleaned["Tenure"].clip(0, 60) * 0.35
    ).clip(1, 100).round(2)
    cleaned["Risk_Indicator"] = (
        cleaned["Monthly_Charges"] * 0.34
        + cleaned["Last_Interaction_Days"] * 2.1
        + cleaned["Support_Tickets"] * 16
        + cleaned["Payment_Delay_Days"] * 1.8
        - cleaned["Usage_Score"] * 0.45
        - cleaned["Tenure"] * 0.25
    ).round(2)

    if "Churn" in cleaned.columns:
        cleaned["Churn"] = cleaned["Churn"].apply(normalize_target)

    column_order = [
        "Customer_ID",
        "Email",
        "Gender",
        "Age",
        "Tenure",
        "Subscription_Type",
        "Contract_Type",
        "Payment_Mode",
        "Monthly_Charges",
        "Total_Spend",
        "Usage_Score",
        "Support_Tickets",
        "Payment_Delay_Days",
        "Last_Interaction_Days",
        "Engagement_Score",
        "Risk_Indicator",
        "Feedback",
        "Sentiment",
        "Issue_Category",
    ]
    if "Churn" in cleaned.columns:
        column_order.append("Churn")
    return cleaned[column_order].copy()


def ensure_dataset_exists(min_rows: int = 2500) -> pd.DataFrame:
    if CLEAN_DATASET_PATH.exists():
        existing = pd.read_csv(CLEAN_DATASET_PATH)
        if len(existing) >= min_rows:
            return existing

    if RAW_DATASET_PATH.exists():
        raw_df = pd.read_csv(RAW_DATASET_PATH)
    else:
        raw_df = create_raw_dataset(row_count=max(min_rows, 5000))
        raw_df.to_csv(RAW_DATASET_PATH, index=False)

    cleaned = clean_customer_dataset(raw_df)
    cleaned.to_csv(CLEAN_DATASET_PATH, index=False)
    return cleaned
