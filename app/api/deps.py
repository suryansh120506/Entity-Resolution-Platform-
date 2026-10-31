"""FastAPI dependencies.

`get_db` is the dependency FIG 2.8's comment refers to:

    -- set once per request/transaction by the FastAPI DB dependency,
    -- from the verified JWT's tenant_id claim — never from a request body

Every route that touches tenant data depends on `get_db`. There is no other
session factory reachable from the API layer, which is what makes that comment
a property of the code rather than an instruction someone has to remember.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import AuthError, Principal, Role, verify_token
from app.db.session import tenant_async_session

_bearer = HTTPBearer(auto_error=False)


async def get_principal(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> Principal:
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        return verify_token(credentials.credentials)
    except AuthError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


async def get_db(
    principal: Annotated[Principal, Depends(get_principal)],
) -> AsyncIterator[AsyncSession]:
    """A request-scoped session already bound to the caller's tenant.

    The binding happens before the route body runs, and is re-applied on every
    transaction the session opens (see app/db/session.py). A route cannot
    obtain an unbound session through this dependency, and a route that forgets
    its own `WHERE tenant_id = ?` gets the RLS policy as a backstop — which is
    exactly the two-layer arrangement NFR-02 asks for, with the application
    filter still expected as the first layer (SAD 3.4: "RLS is the backstop
    for the query that forgets to, not a replacement for writing it").
    """
    async with tenant_async_session(principal.tenant_id) as session:
        yield session


def require_role(*allowed: Role) -> Callable[..., Awaitable[Principal]]:
    """Route guard for the persona boundaries in PRD 1.2.

    Example: only an Org Admin changes the monthly Places cap; only a Data
    Steward submits review feedback.
    """

    async def _guard(
        principal: Annotated[Principal, Depends(get_principal)],
    ) -> Principal:
        if principal.role not in allowed:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="This role may not perform that action",
            )
        return principal

    return _guard


PrincipalDep = Annotated[Principal, Depends(get_principal)]
DbDep = Annotated[AsyncSession, Depends(get_db)]
