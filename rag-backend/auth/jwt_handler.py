# auth/jwt_handler.py
# Issue and verify JWT tokens.
# Uses bcrypt directly (passlib dropped — incompatible with bcrypt >= 4.x).

from datetime import datetime, timedelta, timezone

import bcrypt
from jose import JWTError, jwt

from config import settings

ALGORITHM   = "HS256"
EXPIRE_DAYS = 7


# ── Password helpers ──────────────────────────────────────

def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))


# ── Token helpers ─────────────────────────────────────────

def create_access_token(user_id: str, email: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(days=EXPIRE_DAYS)
    payload = {
        "sub"  : user_id,
        "email": email,
        "exp"  : expire,
    }
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    return jwt.decode(token, settings.jwt_secret_key, algorithms=[ALGORITHM])


__all__ = [
    "hash_password", "verify_password",
    "create_access_token", "decode_token",
]