# auth/dependencies.py
# FastAPI dependency — extracts and validates the current user from the
# Authorization: Bearer <token> header.
# Import get_current_user and use it as Depends(get_current_user) in any route.

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError

from auth.jwt_handler import decode_token

bearer_scheme = HTTPBearer()


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> dict:
    """
    FastAPI dependency.
    Decodes the JWT from the Authorization header.
    Returns {"user_id": str, "email": str} on success.
    Raises HTTP 401 on invalid / missing / expired token.
    """
    token = credentials.credentials
    try:
        payload = decode_token(token)
        user_id: str = payload.get("sub")
        email:   str = payload.get("email")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload",
            )
        return {"user_id": user_id, "email": email}
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token invalid or expired",
            headers={"WWW-Authenticate": "Bearer"},
        )


__all__ = ["get_current_user"]