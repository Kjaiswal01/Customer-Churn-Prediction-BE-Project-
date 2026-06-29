from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import json
import os

from werkzeug.security import check_password_hash, generate_password_hash

from enterprise_config import JWT_EXPIRE_HOURS, JWT_SECRET, SEED_DEFAULT_USERS
from enterprise_database import SessionLocal, User, init_database


def _b64encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("utf-8")


def _b64decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def hash_password(password: str) -> str:
    return generate_password_hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    return check_password_hash(password_hash, password)


def create_access_token(user: User) -> str:
    payload = {
        "sub": str(user.id),
        "email": user.email,
        "role": user.role,
        "exp": int((datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRE_HOURS)).timestamp()),
    }
    header = {"alg": "HS256", "typ": "JWT"}
    header_part = _b64encode(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    payload_part = _b64encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signing_input = f"{header_part}.{payload_part}".encode("utf-8")
    signature = hmac.new(JWT_SECRET.encode("utf-8"), signing_input, hashlib.sha256).digest()
    signature_part = _b64encode(signature)
    return f"{header_part}.{payload_part}.{signature_part}"


def decode_token(token: str) -> dict:
    try:
        header_part, payload_part, signature_part = token.split(".")
    except ValueError as exc:
        raise ValueError("Invalid token format") from exc

    signing_input = f"{header_part}.{payload_part}".encode("utf-8")
    expected_signature = hmac.new(JWT_SECRET.encode("utf-8"), signing_input, hashlib.sha256).digest()
    actual_signature = _b64decode(signature_part)

    if not hmac.compare_digest(expected_signature, actual_signature):
        raise ValueError("Invalid token signature")

    payload = json.loads(_b64decode(payload_part).decode("utf-8"))
    if int(payload.get("exp", 0)) < int(datetime.now(timezone.utc).timestamp()):
        raise ValueError("Token has expired")
    return payload


def create_user(full_name: str, email: str, password: str, role: str = "manager") -> User:
    session = SessionLocal()
    try:
        existing = session.query(User).filter(User.email == email).one_or_none()
        if existing:
            return existing
        user = User(
            full_name=full_name,
            email=email,
            password_hash=hash_password(password),
            role=role,
            is_active=True,
        )
        session.add(user)
        session.commit()
        session.refresh(user)
        return user
    finally:
        session.close()


def authenticate_user(email: str, password: str) -> User | None:
    session = SessionLocal()
    try:
        user = session.query(User).filter(User.email == email, User.is_active.is_(True)).one_or_none()
        if user and verify_password(password, user.password_hash):
            session.expunge(user)
            return user
        return None
    finally:
        session.close()


def seed_default_user() -> None:
    init_database()
    if not SEED_DEFAULT_USERS:
        return
    admin_password = os.getenv("DEFAULT_ADMIN_PASSWORD")
    analyst_password = os.getenv("DEFAULT_ANALYST_PASSWORD")
    if admin_password:
        create_user("Platform Admin", "admin@retentionos.local", admin_password, role="admin")
    if analyst_password:
        create_user("Business Analyst", "analyst@retentionos.local", analyst_password, role="analyst")
