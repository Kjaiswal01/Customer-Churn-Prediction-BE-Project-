from __future__ import annotations

from dataclasses import dataclass, field
import io
import joblib
import logging
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None

try:
    import shap  # type: ignore
except Exception:  # pragma: no cover
    shap = None

from enterprise_config import ARTIFACT_PATH


TARGET_ALIASES = [
    "churn",
    "target",
    "label",
    "churn_status",
    "is_churned",
    "attrition",
    "exit_flag",
    "churn_label",
    "churn_value",
    "attrition_flag",
]
EMAIL_ALIASES = ["email", "email_id", "mail", "customer_email", "emailaddress", "email_address"]
ID_ALIASES = ["customer_id", "customerid", "subscriber_id", "account_id", "user_id", "client_id", "cust_id"]
MAX_CARDINALITY_FOR_CATEGORICAL = 120
MAX_CATEGORICAL_UNIQUENESS_RATIO = 0.4
MAX_EXPLANATION_ROWS = 5000


@dataclass
class TrainingArtifacts:
    pipeline: Pipeline
    preprocessor: ColumnTransformer
    best_model_name: str
    metrics: dict
    target_column: str
    model_version: str = "baseline"
    feature_names: list[str] = field(default_factory=list)
    cluster_model: KMeans | None = None
    cluster_labels: dict[int, str] = field(default_factory=dict)
    training_columns: list[str] = field(default_factory=list)
    explainer_type: str = "feature_importance"
    decision_threshold: float = 0.55
    dropped_columns: list[str] = field(default_factory=list)
    evaluation_summary: dict = field(default_factory=dict)
    schema_profile: dict = field(default_factory=dict)


def sanitize_training_artifacts(artifacts: TrainingArtifacts | None) -> TrainingArtifacts | None:
    if artifacts is None:
        return None

    try:
        model = artifacts.pipeline.named_steps.get("model")
    except Exception:
        return artifacts

    if isinstance(model, LogisticRegression) and not hasattr(model, "multi_class"):
        # Older serialized sklearn models can miss deprecated attributes after upgrades.
        model.multi_class = "auto"

    return artifacts


def normalize_column_name(column_name: str) -> str:
    return column_name.strip().lower().replace(" ", "_")


def detect_column(df: pd.DataFrame, aliases: list[str]) -> str | None:
    normalized = {normalize_column_name(col): col for col in df.columns}
    for alias in aliases:
        if alias in normalized:
            return normalized[alias]
    return None


def normalize_target(value) -> int:
    return 1 if str(value).strip().lower() in {"yes", "1", "true", "churned", "y"} else 0


def load_dataset_from_upload(uploaded_file) -> pd.DataFrame:
    file_name = getattr(uploaded_file, "name", None) or getattr(uploaded_file, "filename", "dataset.csv")
    file_name = file_name.lower()
    if hasattr(uploaded_file, "getvalue"):
        content = uploaded_file.getvalue()
    else:
        uploaded_file.file.seek(0)
        content = uploaded_file.file.read()
    if file_name.endswith(".xlsx") or file_name.endswith(".xls"):
        return pd.read_excel(io.BytesIO(content))
    return pd.read_csv(io.BytesIO(content))


def validate_dataset(df: pd.DataFrame, require_target: bool = True) -> dict:
    target_column = detect_column(df, TARGET_ALIASES)
    if require_target and not target_column:
        raise ValueError("Dataset must contain a churn target column like 'Churn' or 'target'.")

    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    if len(numeric_columns) < 2:
        raise ValueError("Dataset should contain at least two numeric columns for training and segmentation.")

    validation = {
        "target_column": target_column,
        "email_column": detect_column(df, EMAIL_ALIASES),
        "id_column": detect_column(df, ID_ALIASES),
    }
    if require_target and target_column:
        target_values = df[target_column].apply(normalize_target)
        class_counts = target_values.value_counts()
        if target_values.nunique() < 2:
            raise ValueError("Dataset target must contain both churn and non-churn examples.")
        if int(class_counts.min()) < 5:
            raise ValueError("Dataset needs at least 5 examples of each class for reliable training.")
    return validation


def build_feature_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    dataset = df.copy()
    drop_candidates = []
    for aliases in (EMAIL_ALIASES, ID_ALIASES):
        column = detect_column(dataset, aliases)
        if column:
            drop_candidates.append(column)

    # Remove likely leakage fields that identify a single customer but do not generalize.
    for column in dataset.columns:
        normalized = normalize_column_name(column)
        if normalized.endswith("_id") and column not in drop_candidates:
            drop_candidates.append(column)

    drop_columns = list(dict.fromkeys(drop_candidates))
    feature_frame = dataset.drop(columns=drop_columns, errors="ignore").copy()
    normalized_columns = {normalize_column_name(col): col for col in feature_frame.columns}

    feedback_column = normalized_columns.get("feedback")
    if feedback_column:
        feedback_text = feature_frame[feedback_column].fillna("").astype(str)
        lower_feedback = feedback_text.str.lower()
        feature_frame["feedback_length"] = feedback_text.str.len().clip(0, 600)
        feature_frame["feedback_word_count"] = feedback_text.str.split().str.len().clip(0, 120)
        feature_frame["feedback_mentions_price"] = lower_feedback.str.contains("price|pricing|bill|billing|cost|charge", regex=True).astype(int)
        feature_frame["feedback_mentions_support"] = lower_feedback.str.contains("support|service|agent|resolution|response", regex=True).astype(int)
        feature_frame["feedback_mentions_network"] = lower_feedback.str.contains("network|signal|speed|latency|disconnect|outage", regex=True).astype(int)
        # Raw free-text feedback creates unstable one-hot dimensions across companies, so keep derived NLP signals instead.
        feature_frame = feature_frame.drop(columns=[feedback_column], errors="ignore")
        drop_columns.append(feedback_column)

    normalized_columns = {normalize_column_name(col): col for col in feature_frame.columns}

    if "tenure" in normalized_columns and "monthly_charges" in normalized_columns:
        tenure_col = normalized_columns["tenure"]
        monthly_col = normalized_columns["monthly_charges"]
        feature_frame["charges_per_tenure"] = (
            pd.to_numeric(feature_frame[monthly_col], errors="coerce")
            / pd.to_numeric(feature_frame[tenure_col], errors="coerce").replace(0, np.nan)
        ).replace([np.inf, -np.inf], np.nan)

    if "total_spend" in normalized_columns and "monthly_charges" in normalized_columns:
        total_col = normalized_columns["total_spend"]
        monthly_col = normalized_columns["monthly_charges"]
        feature_frame["value_to_monthly_ratio"] = (
            pd.to_numeric(feature_frame[total_col], errors="coerce")
            / pd.to_numeric(feature_frame[monthly_col], errors="coerce").replace(0, np.nan)
        ).replace([np.inf, -np.inf], np.nan)

    if "last_interaction_days" in normalized_columns and "usage_score" in normalized_columns:
        interaction_col = normalized_columns["last_interaction_days"]
        usage_col = normalized_columns["usage_score"]
        feature_frame["interaction_usage_gap"] = (
            pd.to_numeric(feature_frame[interaction_col], errors="coerce")
            - pd.to_numeric(feature_frame[usage_col], errors="coerce")
        )

    if "engagement_score" in normalized_columns and "sentiment" in normalized_columns:
        engagement_col = normalized_columns["engagement_score"]
        sentiment_col = normalized_columns["sentiment"]
        feature_frame["engagement_sentiment_index"] = (
            pd.to_numeric(feature_frame[engagement_col], errors="coerce") * 0.7
            + pd.to_numeric(feature_frame[sentiment_col], errors="coerce") * 30
        )

    if "support_tickets" in normalized_columns and "payment_delay_days" in normalized_columns:
        support_col = normalized_columns["support_tickets"]
        delay_col = normalized_columns["payment_delay_days"]
        feature_frame["service_friction_index"] = (
            pd.to_numeric(feature_frame[support_col], errors="coerce") * 12
            + pd.to_numeric(feature_frame[delay_col], errors="coerce") * 1.8
        )

    # Drop extremely high-cardinality text/category columns before one-hot encoding.
    # These fields can explode feature dimensions and memory usage for large company datasets.
    high_cardinality_columns = []
    row_count = max(len(feature_frame), 1)
    for column in feature_frame.select_dtypes(exclude=["number"]).columns:
        unique_count = int(feature_frame[column].astype(str).nunique(dropna=True))
        uniqueness_ratio = unique_count / row_count
        if unique_count > MAX_CARDINALITY_FOR_CATEGORICAL or uniqueness_ratio > MAX_CATEGORICAL_UNIQUENESS_RATIO:
            high_cardinality_columns.append(column)
    if high_cardinality_columns:
        feature_frame = feature_frame.drop(columns=high_cardinality_columns, errors="ignore")
        drop_columns.extend(high_cardinality_columns)

    return feature_frame, drop_columns


def build_preprocessor(X: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [column for column in X.columns if column not in numeric_features]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.01, max_categories=25)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor, numeric_features, categorical_features


def candidate_models():
    models = {
        "logistic_regression": LogisticRegression(max_iter=1500, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=42,
            class_weight="balanced_subsample",
        ),
    }
    if XGBClassifier is not None:
        models["xgboost"] = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42,
        )
    return models


def ensure_2d_array(matrix) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    array = np.asarray(matrix)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    return array


def build_cluster_frame(df: pd.DataFrame, churn_probabilities: np.ndarray | None = None) -> pd.DataFrame:
    cluster_df = pd.DataFrame(index=df.index)
    feature_defaults = {
        "Tenure": 12,
        "Monthly_Charges": 600,
        "Total_Spend": 6000,
        "Usage_Score": 50,
        "Last_Interaction_Days": 15,
        "Support_Tickets": 1,
        "Payment_Delay_Days": 3,
        "Sentiment": 0,
    }
    for column, default in feature_defaults.items():
        cluster_df[column] = pd.to_numeric(df.get(column, default), errors="coerce").fillna(default)
    if churn_probabilities is not None:
        cluster_df["churn_probability"] = np.asarray(churn_probabilities, dtype=float)
    return cluster_df


def default_feature_importance_explanations(model, feature_names: list[str], row_count: int) -> list[list[dict]]:
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        coefficients = getattr(model, "coef_", None)
        if coefficients is not None:
            importances = np.ravel(coefficients)
    if importances is None:
        importances = np.zeros(len(feature_names))
    ranked = sorted(zip(feature_names, importances), key=lambda item: abs(item[1]), reverse=True)[:5]
    fallback = [
        {
            "feature": name,
            "contribution": float(value),
            "direction": "Increase Risk" if value >= 0 else "Reduce Risk",
        }
        for name, value in ranked
    ]
    return [fallback for _ in range(row_count)]


def select_decision_threshold(y_true: pd.Series, probabilities: np.ndarray) -> tuple[float, float]:
    candidate_thresholds = np.linspace(0.25, 0.75, 21)
    best_threshold = 0.5
    best_score = -1.0
    best_recall = -1.0

    for threshold in candidate_thresholds:
        predictions = probabilities >= threshold
        score = f1_score(y_true, predictions, zero_division=0)
        recall = recall_score(y_true, predictions, zero_division=0)
        if score > best_score or (np.isclose(score, best_score) and recall > best_recall):
            best_threshold = float(threshold)
            best_score = float(score)
            best_recall = float(recall)

    return round(best_threshold, 2), round(best_score, 4)


def summarize_predictions(y_true: pd.Series, probabilities: np.ndarray, threshold: float) -> dict:
    predictions = probabilities >= threshold
    summary = {
        "threshold": round(float(threshold), 2),
        "accuracy": round(accuracy_score(y_true, predictions), 4),
        "precision": round(precision_score(y_true, predictions, zero_division=0), 4),
        "recall": round(recall_score(y_true, predictions, zero_division=0), 4),
        "f1_score": round(f1_score(y_true, predictions, zero_division=0), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y_true, predictions), 4),
        "positive_rate": round(float(np.mean(predictions)), 4),
    }
    if len(np.unique(y_true)) > 1:
        summary["roc_auc"] = round(roc_auc_score(y_true, probabilities), 4)
        summary["average_precision"] = round(average_precision_score(y_true, probabilities), 4)
        summary["brier_score"] = round(brier_score_loss(y_true, probabilities), 4)
    return summary


def evaluate_models(df: pd.DataFrame, target_column: str) -> tuple[TrainingArtifacts, pd.DataFrame]:
    dataset = df.copy()
    dataset[target_column] = dataset[target_column].apply(normalize_target)
    X, dropped_columns = build_feature_frame(dataset.drop(columns=[target_column]))
    y = dataset[target_column]

    preprocessor, _, _ = build_preprocessor(X)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.25,
        random_state=42,
        stratify=y_train_full if y_train_full.nunique() > 1 else None,
    )

    model_scores = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for model_name, model in candidate_models().items():
        pipeline = Pipeline(steps=[("preprocessor", clone(preprocessor)), ("model", model)])
        cv_scores = cross_validate(
            pipeline,
            X_train_full,
            y_train_full,
            cv=cv,
            scoring={
                "accuracy": "accuracy",
                "precision": "precision",
                "recall": "recall",
                "f1": "f1",
                "roc_auc": "roc_auc",
            },
            n_jobs=1,
        )
        pipeline.fit(X_train, y_train)
        validation_probabilities = pipeline.predict_proba(X_valid)[:, 1]
        decision_threshold, validation_best_f1 = select_decision_threshold(y_valid, validation_probabilities)
        validation_summary = summarize_predictions(y_valid, validation_probabilities, decision_threshold)
        model_scores[model_name] = {
            "cv_accuracy": round(float(np.mean(cv_scores["test_accuracy"])), 4),
            "cv_precision": round(float(np.mean(cv_scores["test_precision"])), 4),
            "cv_recall": round(float(np.mean(cv_scores["test_recall"])), 4),
            "cv_f1_score": round(float(np.mean(cv_scores["test_f1"])), 4),
            "cv_roc_auc": round(float(np.mean(cv_scores["test_roc_auc"])), 4),
            "validation_accuracy": validation_summary["accuracy"],
            "validation_precision": validation_summary["precision"],
            "validation_recall": validation_summary["recall"],
            "validation_f1_score": validation_best_f1,
            "validation_roc_auc": validation_summary.get("roc_auc", 0.0),
            "validation_average_precision": validation_summary.get("average_precision", 0.0),
            "decision_threshold": decision_threshold,
        }
    best_model_name = max(
        model_scores,
        key=lambda name: (model_scores[name]["validation_f1_score"], model_scores[name]["validation_roc_auc"]),
    )
    best_model = candidate_models()[best_model_name]
    best_pipeline = Pipeline(steps=[("preprocessor", clone(preprocessor)), ("model", best_model)])
    best_pipeline.fit(X_train_full, y_train_full)
    fitted_preprocessor = best_pipeline.named_steps["preprocessor"]
    decision_threshold = float(model_scores[best_model_name]["decision_threshold"])

    test_probabilities = best_pipeline.predict_proba(X_test)[:, 1]
    test_summary = summarize_predictions(y_test, test_probabilities, decision_threshold)
    model_scores[best_model_name]["test_summary"] = test_summary

    transformed = fitted_preprocessor.transform(X)
    feature_names = list(fitted_preprocessor.get_feature_names_out())
    churn_probabilities = best_pipeline.predict_proba(X)[:, 1]

    cluster_input = build_cluster_frame(X, churn_probabilities)
    cluster_model = KMeans(n_clusters=4, n_init=10, random_state=42)
    clusters = cluster_model.fit_predict(cluster_input)
    cluster_labels = label_clusters(churn_probabilities, clusters)

    dataset["predicted_churn_probability"] = churn_probabilities
    dataset["customer_segment"] = [cluster_labels[cluster] for cluster in clusters]

    artifacts = TrainingArtifacts(
        pipeline=best_pipeline,
        preprocessor=fitted_preprocessor,
        best_model_name=best_model_name,
        metrics=model_scores,
        feature_names=feature_names,
        cluster_model=cluster_model,
        cluster_labels=cluster_labels,
        training_columns=X.columns.tolist(),
        target_column=target_column,
        explainer_type="shap" if shap is not None else "feature_importance",
        decision_threshold=decision_threshold,
        dropped_columns=dropped_columns,
        evaluation_summary={
            "dataset_rows": int(len(dataset)),
            "churn_rate": round(float(y.mean()), 4),
            "validation_rows": int(len(X_valid)),
            "test_rows": int(len(X_test)),
            "test_summary": test_summary,
        },
    )
    return artifacts, dataset


def label_clusters(churn_probabilities: np.ndarray, clusters: np.ndarray) -> dict[int, str]:
    labels = {}
    for cluster in sorted(set(clusters)):
        cluster_mean = float(np.mean(churn_probabilities[clusters == cluster]))
        if cluster_mean >= 0.75:
            labels[cluster] = "Critical Churn"
        elif cluster_mean >= 0.5:
            labels[cluster] = "High Attention"
        elif cluster_mean >= 0.25:
            labels[cluster] = "Growth Opportunity"
        else:
            labels[cluster] = "Loyal Base"
    return labels


def persist_artifacts(artifacts: TrainingArtifacts) -> None:
    joblib.dump(artifacts, ARTIFACT_PATH)


def load_artifacts() -> TrainingArtifacts | None:
    if ARTIFACT_PATH.exists():
        try:
            return sanitize_training_artifacts(joblib.load(ARTIFACT_PATH))
        except (OSError, ValueError, TypeError) as exc:
            logger.warning("Failed to load default artifacts from %s: %s", ARTIFACT_PATH, exc)
            return None
    return None


def load_artifacts_from_path(path) -> TrainingArtifacts | None:
    try:
        artifact_path = str(path)
        if not artifact_path:
            return None
        return sanitize_training_artifacts(joblib.load(artifact_path))
    except (OSError, ValueError, TypeError) as exc:
        logger.warning("Failed to load artifacts from %s: %s", path, exc)
        return None


def transform_features(df: pd.DataFrame, artifacts: TrainingArtifacts):
    feature_frame, _ = build_feature_frame(df.copy())
    missing_cols = [column for column in artifacts.training_columns if column not in feature_frame.columns]
    for column in missing_cols:
        feature_frame[column] = np.nan
    ordered = feature_frame[artifacts.training_columns].copy()
    transformed = artifacts.preprocessor.transform(ordered)
    return ordered, transformed


def generate_shap_explanations(artifacts: TrainingArtifacts, raw_df: pd.DataFrame, transformed) -> list[list[dict]]:
    model = artifacts.pipeline.named_steps["model"]
    feature_names = artifacts.feature_names
    explanations = []
    if shap is None:
        return default_feature_importance_explanations(model, feature_names, len(raw_df))

    transformed_shape = getattr(transformed, "shape", (len(raw_df), len(feature_names)))
    row_count = int(transformed_shape[0]) if len(transformed_shape) > 0 else len(raw_df)
    feature_count = int(transformed_shape[1]) if len(transformed_shape) > 1 else len(feature_names)
    if row_count > MAX_EXPLANATION_ROWS or feature_count > 2000 or hasattr(transformed, "toarray"):
        return default_feature_importance_explanations(model, feature_names, len(raw_df))

    try:
        transformed_array = ensure_2d_array(transformed)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(transformed_array)
        if isinstance(shap_values, list):
            values = np.asarray(shap_values[1] if len(shap_values) > 1 else shap_values[0])
        else:
            values = np.asarray(shap_values)
            if values.ndim == 3:
                values = values[:, :, -1]
        for row_values in values:
            ranked = sorted(zip(feature_names, row_values), key=lambda item: abs(item[1]), reverse=True)[:5]
            explanations.append(
                [
                    {
                        "feature": name,
                        "contribution": round(float(value), 4),
                        "direction": "Increase Risk" if value >= 0 else "Reduce Risk",
                    }
                    for name, value in ranked
                ]
            )
        return explanations
    except (ValueError, TypeError, AttributeError) as exc:
        logger.warning("Falling back from SHAP explanations due to: %s", exc)
        return default_feature_importance_explanations(model, feature_names, len(raw_df))


def customer_value_score(row: pd.Series) -> float:
    monthly = float(row.get("monthly_charges", row.get("Monthly_Charges", 0)) or 0)
    total = float(row.get("total_spend", row.get("Total_Spend", 0)) or 0)
    tenure = float(row.get("tenure", row.get("Tenure", 0)) or 0)
    return round((monthly * 0.4) + (total * 0.4) + (tenure * 15), 2)


def heuristic_churn_probability(row: pd.Series) -> float:
    monthly = float(row.get("monthly_charges", row.get("Monthly_Charges", 0)) or 0)
    total = float(row.get("total_spend", row.get("Total_Spend", 0)) or 0)
    tenure = float(row.get("tenure", row.get("Tenure", 0)) or 0)
    usage = float(row.get("usage_score", row.get("Usage_Score", 50)) or 50)
    last_interaction = float(row.get("last_interaction_days", row.get("Last_Interaction_Days", 15)) or 15)
    sentiment = float(row.get("sentiment", row.get("Sentiment", 0)) or 0)
    support_tickets = float(row.get("support_tickets", row.get("Support_Tickets", 1)) or 1)
    payment_delay = float(row.get("payment_delay_days", row.get("Payment_Delay_Days", 0)) or 0)
    issue = str(row.get("issue_category", row.get("Issue_Category", "")) or "").lower()
    contract_type = str(row.get("contract_type", row.get("Contract_Type", "")) or "").lower()
    payment_mode = str(row.get("payment_mode", row.get("Payment_Mode", "")) or "").lower()

    score = -0.75
    score += np.clip((monthly - 600) / 1800, -0.1, 0.45)
    score += np.clip((last_interaction - 18) / 90, -0.08, 0.34)
    score += np.clip((support_tickets - 1) * 0.07, -0.03, 0.35)
    score += np.clip((payment_delay - 3) / 40, -0.02, 0.24)
    score -= np.clip((usage - 55) / 140, -0.22, 0.2)
    score -= np.clip((tenure - 12) / 80, -0.18, 0.16)
    score -= np.clip(sentiment * 0.42, -0.25, 0.25)
    if total >= 20000:
        score += 0.03
    if "pricing" in issue:
        score += 0.12
    elif "network" in issue:
        score += 0.09
    elif "service" in issue:
        score += 0.08
    if contract_type == "annual":
        score -= 0.14
    elif contract_type == "monthly":
        score += 0.05
    if payment_mode == "autopay":
        score -= 0.06
    probability = 1 / (1 + np.exp(-score * 2.4))
    return round(float(np.clip(probability, 0.02, 0.98)), 4)


def business_risk_adjustment(row: pd.Series) -> float:
    tenure = float(row.get("tenure", row.get("Tenure", 0)) or 0)
    monthly = float(row.get("monthly_charges", row.get("Monthly_Charges", 0)) or 0)
    usage = float(row.get("usage_score", row.get("Usage_Score", 50)) or 50)
    last_interaction = float(row.get("last_interaction_days", row.get("Last_Interaction_Days", 15)) or 15)
    sentiment = float(row.get("sentiment", row.get("Sentiment", 0)) or 0)
    support_tickets = float(row.get("support_tickets", row.get("Support_Tickets", 1)) or 1)
    issue = str(row.get("issue_category", row.get("Issue_Category", "")) or "").lower()

    adjustment = 0.0
    if tenure <= 3:
        adjustment += 0.12
    elif tenure <= 6:
        adjustment += 0.06
    elif tenure >= 24:
        adjustment -= 0.08

    if monthly >= 1200:
        adjustment += 0.08
    elif monthly <= 500:
        adjustment -= 0.03

    if usage <= 25:
        adjustment += 0.14
    elif usage <= 45:
        adjustment += 0.07
    elif usage >= 80:
        adjustment -= 0.08

    if last_interaction >= 60:
        adjustment += 0.12
    elif last_interaction >= 30:
        adjustment += 0.06
    elif last_interaction <= 7:
        adjustment -= 0.05

    if sentiment <= -0.4:
        adjustment += 0.14
    elif sentiment <= -0.1:
        adjustment += 0.07
    elif sentiment >= 0.4:
        adjustment -= 0.08

    if support_tickets >= 5:
        adjustment += 0.09
    elif support_tickets == 0:
        adjustment -= 0.02

    if "pricing" in issue:
        adjustment += 0.06
    elif "network" in issue:
        adjustment += 0.05
    elif "service" in issue:
        adjustment += 0.04

    return adjustment


def calibrate_probability(raw_probability: float, row: pd.Series) -> float:
    heuristic = heuristic_churn_probability(row)
    if np.isnan(raw_probability):
        return heuristic
    baseline = float(np.clip(raw_probability, 0.01, 0.99))
    adjusted = baseline + business_risk_adjustment(row)
    clipped_adjusted = float(np.clip(adjusted, 0.01, 0.99))
    divergence = abs(baseline - heuristic)
    ml_weight = 0.78 if divergence <= 0.22 else 0.66
    heuristic_weight = 0.14 if divergence <= 0.22 else 0.22
    adjustment_weight = 0.08 if divergence <= 0.22 else 0.12
    # Preserve model separation when the trained model is confident, while still using business rules as a stabilizer.
    calibrated = (baseline * ml_weight) + (clipped_adjusted * adjustment_weight) + (heuristic * heuristic_weight)
    return round(float(np.clip(calibrated, 0.01, 0.99)), 4)


def retention_strategy(row: pd.Series, churn_probability: float, customer_value: float) -> dict:
    sentiment = float(row.get("sentiment", row.get("Sentiment", 0)) or 0)
    tenure = float(row.get("tenure", row.get("Tenure", 0)) or 0)
    issue = str(row.get("issue_category", row.get("Issue_Category", "")) or "")

    if churn_probability >= 0.8 and customer_value >= 1500:
        return {
            "strategy": "VIP Rescue",
            "offer": "Dedicated account manager + 20% retention discount",
            "action": "Call within 24 hours and assign priority support.",
            "priority": "Critical",
        }
    if sentiment < -0.2:
        return {
            "strategy": "Service Recovery",
            "offer": "Priority support callback and case resolution assurance",
            "action": f"Assign a support specialist to resolve the {issue or 'service'} concern within 24 hours.",
            "priority": "High",
        }
    if tenure <= 6:
        return {
            "strategy": "Onboarding Rescue",
            "offer": "Guided onboarding session and first-quarter success plan",
            "action": "Schedule onboarding coaching and a success manager follow-up.",
            "priority": "Medium",
        }
    if churn_probability >= 0.65:
        return {
            "strategy": "Save Campaign",
            "offer": "Bundled upgrade or loyalty plan with extra benefits",
            "action": "Send personalized offer and track response within 3 days.",
            "priority": "High",
        }
    if churn_probability >= 0.4:
        return {
            "strategy": "Engagement Boost",
            "offer": "Free booster pack or 10% upsell incentive",
            "action": "Trigger re-engagement email with feature education.",
            "priority": "Medium",
        }
    return {
        "strategy": "Loyalty Nurture",
        "offer": "Referral reward or loyalty points booster",
        "action": "Keep warm with value-based engagement messaging.",
        "priority": "Low",
    }


def personalized_offer(row: pd.Series, churn_probability: float) -> dict:
    monthly = float(row.get("monthly_charges", row.get("Monthly_Charges", 0)) or 0)
    tenure = float(row.get("tenure", row.get("Tenure", 0)) or 0)
    usage = float(row.get("usage_score", row.get("Usage_Score", 0)) or 0)
    sentiment = float(row.get("sentiment", row.get("Sentiment", 0)) or 0)
    issue = str(row.get("issue_category", row.get("Issue_Category", "")) or "").lower()

    if "pricing" in issue and monthly >= 800:
        return {"offer_name": "Price Relief Plan", "offer_details": "15% bill reduction for 3 months with premium support."}
    if ("service" in issue and sentiment < -0.05) or sentiment < -0.2:
        return {"offer_name": "White-Glove Recovery", "offer_details": "Dedicated support specialist, priority resolution, and service credit."}
    if "network" in issue and sentiment <= 0.1:
        return {"offer_name": "Network Assurance Pack", "offer_details": "Priority technical review, issue monitoring, and complimentary service add-on."}
    if churn_probability >= 0.6 and usage < 40:
        return {"offer_name": "Usage Booster", "offer_details": "Free extra usage credits and guided onboarding support."}
    if tenure < 6:
        return {"offer_name": "Welcome Back Pack", "offer_details": "Onboarding assistance plus first-upgrade discount."}
    return {"offer_name": "Loyalty Reward", "offer_details": "Referral credits and exclusive member benefits."}


def score_customers(df: pd.DataFrame, artifacts: TrainingArtifacts) -> pd.DataFrame:
    ordered, transformed = transform_features(df.copy(), artifacts)
    raw_probabilities = np.asarray(artifacts.pipeline.predict_proba(ordered)[:, 1], dtype=float)
    heuristic_probabilities = np.array([heuristic_churn_probability(ordered.iloc[index]) for index in range(len(ordered))])
    raw_probabilities = np.where(np.isnan(raw_probabilities), heuristic_probabilities, raw_probabilities)
    probabilities = np.array(
        [calibrate_probability(float(raw_probabilities[index]), ordered.iloc[index]) for index in range(len(ordered))]
    )
    cluster_input = build_cluster_frame(ordered, probabilities)
    try:
        clusters = artifacts.cluster_model.predict(cluster_input) if artifacts.cluster_model is not None else np.zeros(len(ordered), dtype=int)
        cluster_labels = artifacts.cluster_labels or {}
    except (ValueError, AttributeError) as exc:
        logger.warning("Cluster prediction fallback activated: %s", exc)
        clusters = np.zeros(len(ordered), dtype=int)
        cluster_labels = {0: "Critical Churn"}
    explanations = generate_shap_explanations(artifacts, ordered, transformed)

    scored = df.copy()
    scored["raw_model_probability"] = raw_probabilities
    scored["heuristic_probability"] = heuristic_probabilities
    scored["churn_probability"] = probabilities
    scored["churn_prediction"] = probabilities >= float(getattr(artifacts, "decision_threshold", 0.55))
    scored["customer_segment"] = [
        cluster_labels.get(cluster)
        or ("Critical Churn" if probability >= 0.8 else "High Attention" if probability >= 0.6 else "Growth Opportunity" if probability >= 0.35 else "Loyal Base")
        for cluster, probability in zip(clusters, probabilities)
    ]
    scored["customer_value"] = scored.apply(customer_value_score, axis=1)
    scored["retention_strategy"] = scored.apply(
        lambda row: retention_strategy(row, float(row["churn_probability"]), float(row["customer_value"])), axis=1
    )
    scored["personalized_offer"] = scored.apply(
        lambda row: personalized_offer(row, float(row["churn_probability"])), axis=1
    )
    scored["shap_explanation"] = explanations
    return scored
logger = logging.getLogger(__name__)
