from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Callable

from fastapi import Header, HTTPException, status

from .config import Settings


@dataclass(slots=True)
class UserContext:
    owner_id: str


def build_auth_dependency(settings: Settings) -> Callable[..., UserContext]:
    """Create a FastAPI dependency that enforces single-owner access."""

    def require_user_context(
        x_owner_id: Annotated[str | None, Header()] = None,
        x_api_key: Annotated[str | None, Header()] = None,
    ) -> UserContext:
        if x_owner_id != settings.app.owner_id or x_api_key != settings.app.api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Unauthorized: owner identity mismatch.",
            )
        return UserContext(owner_id=x_owner_id)

    return require_user_context
