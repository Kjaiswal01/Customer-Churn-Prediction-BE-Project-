from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from data_pipeline import clean_customer_dataset


CANONICAL_ALIASES: dict[str, tuple[str, ...]] = {
    "Customer_ID": ("customer_id", "customerid", "subscriber_id", "account_id", "user_id", "client_id", "cust_id"),
    "Customer_Name": ("customer_name", "name", "full_name", "customer", "client_name", "subscriber_name"),
    "Email": ("email", "email_id", "mail", "customer_email", "emailaddress"),
    "Gender": ("gender", "sex"),
    "Age": ("age", "customer_age"),
    "Tenure": ("tenure", "tenure_months", "months_with_company", "customer_tenure", "subscription_tenure"),
    "Subscription_Type": ("subscription_type", "plan_type", "plan", "plan_name", "package", "product_plan", "subscription"),
    "Contract_Type": ("contract_type", "contract", "billing_cycle", "term_type", "contract_length"),
    "Payment_Mode": ("payment_mode", "payment_method", "paymenttype", "billing_method", "paymentmethod"),
    "Monthly_Charges": ("monthly_charges", "monthlycharges", "monthlycharge", "monthly_fee", "mrr", "bill_amount", "monthly_amount"),
    "Total_Spend": ("total_spend", "totalspend", "total_charges", "totalcharges", "lifetime_value", "customer_value_total", "revenue_total"),
    "Usage_Score": ("usage_score", "usagescore", "engagement", "engagement_score", "activity_score", "usage", "utilization_score", "usage_frequency"),
    "Support_Tickets": ("support_tickets", "tickets", "ticket_count", "complaints_count", "cases_opened", "support_calls"),
    "Payment_Delay_Days": ("payment_delay_days", "late_payment_days", "days_past_due", "payment_delay", "overdue_days"),
    "Last_Interaction_Days": ("last_interaction_days", "days_since_last_interaction", "last_contact_days", "recency_days", "last_interaction"),
    "Feedback": ("feedback", "customer_feedback", "comment", "comments", "review", "complaint_text"),
    "Sentiment": ("sentiment", "sentiment_score", "feedback_sentiment"),
    "Issue_Category": ("issue_category", "complaint_category", "problem_type", "issue_type"),
    "Churn": ("churn", "target", "label", "is_churned", "churn_status", "attrition", "exit_flag", "churn_label", "churn_value", "attrition_flag"),
}


@dataclass
class DynamicDatasetResult:
    dataframe: pd.DataFrame
    schema_mapping: dict


def normalize_key(value: object) -> str:
    return (
        str(value or "")
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace(".", "_")
    )


def _combine_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    combined = pd.DataFrame(index=df.index)
    for column in df.columns:
        if column in combined.columns:
            combined[column] = combined[column].combine_first(df[column])
        else:
            combined[column] = df[column]
    return combined


def detect_schema_mapping(df: pd.DataFrame) -> dict:
    normalized_lookup = {normalize_key(column): column for column in df.columns}
    mapped_columns: dict[str, str] = {}
    for canonical_name, aliases in CANONICAL_ALIASES.items():
        for alias in aliases:
            if alias in normalized_lookup:
                mapped_columns[canonical_name] = normalized_lookup[alias]
                break

    canonical_sources = set(mapped_columns.values())
    unmapped_columns = [column for column in df.columns if column not in canonical_sources]
    profile = {
        "row_count": int(len(df)),
        "original_columns": list(df.columns),
        "mapped_columns": mapped_columns,
        "unmapped_columns": unmapped_columns,
        "missing_canonical_columns": [column for column in CANONICAL_ALIASES if column not in mapped_columns],
        "data_types": {column: str(dtype) for column, dtype in df.dtypes.items()},
        "numeric_columns": df.select_dtypes(include=["number"]).columns.tolist(),
        "categorical_columns": [column for column in df.columns if column not in df.select_dtypes(include=["number"]).columns],
    }
    return profile


def adapt_company_dataset(df: pd.DataFrame) -> DynamicDatasetResult:
    profile = detect_schema_mapping(df)
    rename_map = {
        source_column: canonical_name
        for canonical_name, source_column in profile["mapped_columns"].items()
        if source_column in df.columns
    }
    standardized = df.rename(columns=rename_map).copy()

    if "Customer_Name" not in standardized.columns:
        first_name = next((column for column in standardized.columns if normalize_key(column) in {"first_name", "firstname"}), None)
        last_name = next((column for column in standardized.columns if normalize_key(column) in {"last_name", "lastname"}), None)
        if first_name and last_name:
            standardized["Customer_Name"] = (
                standardized[first_name].fillna("").astype(str).str.strip()
                + " "
                + standardized[last_name].fillna("").astype(str).str.strip()
            ).str.strip()
            profile["mapped_columns"]["Customer_Name"] = f"{first_name}+{last_name}"

    standardized = _combine_duplicate_columns(standardized)

    core_columns = [column for column in CANONICAL_ALIASES.keys() if column in standardized.columns]
    core_dataset = standardized[core_columns].copy()
    for canonical_name in CANONICAL_ALIASES:
        if canonical_name not in core_dataset.columns:
            core_dataset[canonical_name] = pd.Series([None] * len(core_dataset), index=core_dataset.index)
    cleaned_core = clean_customer_dataset(core_dataset)

    extra_columns = [column for column in standardized.columns if column not in cleaned_core.columns]
    extras = standardized[extra_columns].copy()
    adapted = pd.concat([cleaned_core.reset_index(drop=True), extras.reset_index(drop=True)], axis=1)
    adapted = _combine_duplicate_columns(adapted)

    if "Customer_Name" not in adapted.columns:
        adapted["Customer_Name"] = adapted["Customer_ID"].astype(str)
    adapted["Customer_Name"] = adapted["Customer_Name"].fillna(adapted["Customer_ID"].astype(str)).astype(str)

    profile["adapted_columns"] = list(adapted.columns)
    profile["extra_feature_columns"] = [
        column
        for column in adapted.columns
        if column not in CANONICAL_ALIASES and column not in {"Customer_Name"}
    ]
    profile["retained_extra_feature_count"] = len(profile["extra_feature_columns"])
    return DynamicDatasetResult(dataframe=adapted, schema_mapping=profile)
