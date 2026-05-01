from __future__ import annotations

from datetime import datetime
from typing import Generator

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, JSON, String, Text, create_engine, inspect
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

from enterprise_config import DATABASE_FALLBACK_URL, DATABASE_URL, USE_SQLITE_FALLBACK

Base = declarative_base()


def build_engine():
    primary_url = DATABASE_FALLBACK_URL if USE_SQLITE_FALLBACK else DATABASE_URL
    try:
        engine = create_engine(primary_url, pool_pre_ping=True, future=True)
        with engine.connect() as connection:
            connection.exec_driver_sql("SELECT 1")
        return engine
    except SQLAlchemyError:
        fallback_engine = create_engine(DATABASE_FALLBACK_URL, pool_pre_ping=True, future=True)
        return fallback_engine


engine = build_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


class Company(Base):
    __tablename__ = "companies"
    id = Column(Integer, primary_key=True)
    name = Column(String(255), unique=True, nullable=False)
    industry = Column(String(100), default="Telecom")
    contact_email = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    customers = relationship("CustomerRecord", back_populates="company", cascade="all, delete-orphan")


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    full_name = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(50), default="manager", nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class DatasetUpload(Base):
    __tablename__ = "dataset_uploads"
    id = Column(Integer, primary_key=True)
    company_id = Column(Integer, ForeignKey("companies.id"), nullable=False)
    filename = Column(String(255), nullable=False)
    uploaded_by_email = Column(String(255))
    stored_path = Column(String(500), nullable=False)
    row_count = Column(Integer, default=0)
    dataset_role = Column(String(50), default="scoring")
    model_version = Column(String(100))
    schema_mapping = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class CustomerRecord(Base):
    __tablename__ = "customer_records"
    id = Column(Integer, primary_key=True)
    company_id = Column(Integer, ForeignKey("companies.id"), nullable=False)
    external_customer_id = Column(String(255), nullable=False)
    customer_name = Column(String(255))
    email = Column(String(255))
    raw_payload = Column(JSON, default=dict)
    churn_probability = Column(Float)
    churn_prediction = Column(Boolean, default=False)
    churn_segment = Column(String(100))
    churn_cluster = Column(String(100))
    customer_value = Column(Float)
    recommended_offer = Column(String(255))
    recommended_action = Column(Text)
    explanation = Column(JSON, default=list)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    company = relationship("Company", back_populates="customers")


class PredictionLog(Base):
    __tablename__ = "prediction_logs"
    id = Column(Integer, primary_key=True)
    company_id = Column(Integer, ForeignKey("companies.id"))
    customer_record_id = Column(Integer, ForeignKey("customer_records.id"))
    input_payload = Column(JSON, default=dict)
    prediction_payload = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class RetentionAction(Base):
    __tablename__ = "retention_actions"
    id = Column(Integer, primary_key=True)
    company_id = Column(Integer, ForeignKey("companies.id"), nullable=False)
    customer_record_id = Column(Integer, ForeignKey("customer_records.id"), nullable=False)
    action_type = Column(String(100), nullable=False)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)
    offer_name = Column(String(255))
    priority = Column(String(50), default="Medium")
    email_status = Column(String(50), default="Not Sent")
    status = Column(String(50), default="pending")
    assigned_agent = Column(String(255))
    due_at = Column(DateTime)
    metadata_json = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class ActionLog(Base):
    __tablename__ = "action_logs"
    id = Column(Integer, primary_key=True)
    company_id = Column(Integer, ForeignKey("companies.id"), nullable=False)
    customer_record_id = Column(Integer, ForeignKey("customer_records.id"), nullable=False)
    customer_external_id = Column(String(255))
    retention_action_id = Column(Integer, ForeignKey("retention_actions.id"))
    action_type = Column(String(100), nullable=False, default="manual")
    action_name = Column(String(100), nullable=False)
    actor_email = Column(String(255))
    status = Column(String(50), default="pending")
    notes = Column(Text)
    metadata_json = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class CustomerFeedback(Base):
    __tablename__ = "customer_feedback"
    id = Column(Integer, primary_key=True)
    company_id = Column(Integer, ForeignKey("companies.id"), nullable=False)
    customer_record_id = Column(Integer, ForeignKey("customer_records.id"), nullable=False)
    feedback_text = Column(Text, nullable=False)
    sentiment_label = Column(String(50))
    sentiment_score = Column(Float)
    issue_category = Column(String(100))
    keywords_json = Column(JSON, default=list)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class TimelineEvent(Base):
    __tablename__ = "timeline_events"
    id = Column(Integer, primary_key=True)
    company_id = Column(Integer, ForeignKey("companies.id"), nullable=False)
    customer_record_id = Column(Integer, ForeignKey("customer_records.id"), nullable=False)
    event_type = Column(String(100), nullable=False)
    title = Column(String(255), nullable=False)
    description = Column(Text)
    event_time = Column(DateTime, default=datetime.utcnow, nullable=False)
    metadata_json = Column(JSON, default=dict)


class WorkflowRecord(Base):
    __tablename__ = "workflow_records"
    id = Column(Integer, primary_key=True)
    company_id = Column(Integer, ForeignKey("companies.id"), nullable=False)
    customer_record_id = Column(Integer, ForeignKey("customer_records.id"), nullable=False)
    stage = Column(String(100), nullable=False)
    status = Column(String(50), default="pending")
    owner = Column(String(255))
    deadline = Column(DateTime)
    outcome = Column(String(100))
    metadata_json = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class AuditLog(Base):
    __tablename__ = "audit_logs"
    id = Column(Integer, primary_key=True)
    company_id = Column(Integer, ForeignKey("companies.id"))
    user_email = Column(String(255))
    event_type = Column(String(100), nullable=False)
    entity_type = Column(String(100))
    entity_id = Column(String(255))
    details_json = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


def init_database() -> None:
    Base.metadata.create_all(bind=engine)
    inspector = inspect(engine)
    table_migrations = {
        "retention_actions": {
            "status": "ALTER TABLE retention_actions ADD COLUMN status VARCHAR(50) DEFAULT 'pending'",
            "assigned_agent": "ALTER TABLE retention_actions ADD COLUMN assigned_agent VARCHAR(255)",
            "due_at": "ALTER TABLE retention_actions ADD COLUMN due_at DATETIME",
        },
        "customer_records": {
            "customer_name": "ALTER TABLE customer_records ADD COLUMN customer_name VARCHAR(255)",
        },
        "dataset_uploads": {
            "uploaded_by_email": "ALTER TABLE dataset_uploads ADD COLUMN uploaded_by_email VARCHAR(255)",
            "dataset_role": "ALTER TABLE dataset_uploads ADD COLUMN dataset_role VARCHAR(50) DEFAULT 'scoring'",
            "model_version": "ALTER TABLE dataset_uploads ADD COLUMN model_version VARCHAR(100)",
        },
        "action_logs": {
            "customer_external_id": "ALTER TABLE action_logs ADD COLUMN customer_external_id VARCHAR(255)",
            "action_type": "ALTER TABLE action_logs ADD COLUMN action_type VARCHAR(100) DEFAULT 'manual'",
        },
    }
    with engine.begin() as connection:
        for table_name, required_columns in table_migrations.items():
            if table_name not in inspector.get_table_names():
                continue
            existing_columns = {column["name"] for column in inspector.get_columns(table_name)}
            for column_name, ddl in required_columns.items():
                if column_name not in existing_columns:
                    connection.exec_driver_sql(ddl)


def get_session() -> Generator:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
