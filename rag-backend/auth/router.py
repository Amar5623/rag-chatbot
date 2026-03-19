# auth/router.py
# POST /auth/signup  — create new account
# POST /auth/login   — authenticate and receive JWT

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr, Field

from auth.user_store  import UserStore
from auth.jwt_handler import hash_password, verify_password, create_access_token

router = APIRouter(prefix="/auth", tags=["auth"])

# UserStore singleton — injected via dependency so it's testable
_user_store: UserStore = None


def get_user_store() -> UserStore:
    return _user_store


def init_user_store(store: UserStore) -> None:
    """Called once from main.py startup."""
    global _user_store
    _user_store = store


# ── Request / Response models ──────────────────────────────

class SignupRequest(BaseModel):
    email   : EmailStr
    password: str = Field(..., min_length=6, max_length=128)


class LoginRequest(BaseModel):
    email   : EmailStr
    password: str = Field(..., min_length=1)


class TokenResponse(BaseModel):
    access_token: str
    token_type  : str = "bearer"
    user_id     : str
    email       : str


# ── Endpoints ──────────────────────────────────────────────

@router.post("/signup", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def signup(
    req  : SignupRequest,
    store: UserStore = Depends(get_user_store),
):
    """
    Create a new account.
    Returns a JWT on success so the user is immediately logged in.
    """
    if store.email_exists(req.email):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists.",
        )

    user_id        = str(uuid.uuid4())
    hashed         = hash_password(req.password)
    store.create_user(user_id, req.email, hashed)

    token = create_access_token(user_id, req.email)
    return TokenResponse(
        access_token=token,
        user_id=user_id,
        email=req.email,
    )


@router.post("/login", response_model=TokenResponse)
def login(
    req  : LoginRequest,
    store: UserStore = Depends(get_user_store),
):
    """
    Authenticate with email + password.
    Returns a JWT on success.
    """
    user = store.get_by_email(req.email)

    # Same error for wrong email OR wrong password — don't leak which one
    invalid = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid email or password.",
    )

    if not user:
        raise invalid
    if not verify_password(req.password, user["hashed_password"]):
        raise invalid

    token = create_access_token(user["id"], user["email"])
    return TokenResponse(
        access_token=token,
        user_id=user["id"],
        email=user["email"],
    )


__all__ = ["router", "init_user_store"]