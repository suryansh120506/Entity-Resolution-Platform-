"""JWT verification and the request principal.

TRD 2.5, stated as plainly as the document does: "Tenant ID is never accepted
as a client-supplied field — not in the body, not as a query param — it comes
only from the verified token."

`Principal.tenant_id` is therefore the ONLY source of a tenant id anywhere in
the request path. If you find yourself reaching for a `tenant_id` field on a
Pydantic request model, that is the bug this module exists to prevent.
"""

from __future__ import annotations

import uuid
from enum import StrEnum

import jwt
from pydantic import BaseModel, ConfigDict

from app.core.config import get_settings


class Role(StrEnum):
    """PRD 1.2's three personas, as authorisation roles."""

    DATA_ENGINEER = "data_engineer"
    DATA_STEWARD = "data_steward"
    ORG_ADMIN = "org_admin"


class Principal(BaseModel):
    """The verified caller. Immutable on purpose.

    Frozen because a mutable principal is one `principal.tenant_id = ...`
    away from being the client-supplied tenant id that TRD 2.5 forbids.
    """

    model_config = ConfigDict(frozen=True)

    subject: str
    tenant_id: uuid.UUID
    role: Role


class AuthError(Exception):
    """Raised on any token that cannot be trusted. Mapped to 401 by the API."""


def verify_token(token: str) -> Principal:
    """Verify a bearer token and extract the principal.

    HS256 with a shared secret is a development posture. Production should
    move to RS256 with a JWKS endpoint, which is a change to this function's
    key resolution and signature algorithm and to nothing else — every caller
    depends on `Principal`, not on how it was obtained.
    """
    settings = get_settings()

    try:
        claims = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
            audience=settings.jwt_audience,
            issuer=settings.jwt_issuer,
            options={"require": ["exp", "sub"]},
        )
    except jwt.InvalidTokenError as exc:
        # Deliberately not echoing the library's message to the client: the
        # difference between "expired" and "bad signature" is useful to an
        # attacker and not to a legitimate caller.
        raise AuthError("Invalid or expired token") from exc

    raw_tenant = claims.get("tenant_id")
    if not raw_tenant:
        raise AuthError("Token is missing the tenant_id claim")

    try:
        tenant_id = uuid.UUID(str(raw_tenant))
    except (ValueError, TypeError) as exc:
        # This value goes on to be bound into the RLS session variable. A
        # non-UUID must be rejected here, at the edge, rather than becoming a
        # database error deeper in.
        raise AuthError("tenant_id claim is not a valid UUID") from exc

    raw_role = claims.get("role")
    try:
        role = Role(str(raw_role))
    except ValueError as exc:
        raise AuthError("Token carries an unrecognised role") from exc

    return Principal(subject=str(claims["sub"]), tenant_id=tenant_id, role=role)
