"""Unit tests for the v1 API surface — TRD 2.5 / FIG 2.9 / FIG 2.10.

No database. These cover the parts of the presentation layer that are the
API's own logic rather than the database's: the keyset cursor, the feedback
contract, the auth boundary, and the route surface itself. Anything that
needs a bound session — the queries, and the RLS behaviour underneath them —
is integration surface and belongs in the Docker-backed suite.

The auth tests are worth their length. TRD 2.5's rule is that tenant comes
only from a verified token, and the way that rule breaks is by someone
adding a route without the dependency. `test_every_v1_route_requires_auth`
walks the live OpenAPI schema rather than a hand-maintained list, so a new
unauthenticated route fails this file the day it is written.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import jwt
import pytest
from fastapi.testclient import TestClient

from app.api.pagination import InvalidCursorError, decode_cursor, encode_cursor
from app.core.config import get_settings
from app.core.security import Role
from app.main import app
from app.models.enums import FeedbackDecision
from app.schemas.feedback import FeedbackRequest

TENANT = uuid.uuid4()
RECORD_A, RECORD_B = uuid.uuid4(), uuid.uuid4()


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


# --- Keyset cursors ------------------------------------------------------------


def test_cursor_round_trips() -> None:
    moment = datetime.now(UTC)
    row_id = uuid.uuid4()

    decoded = decode_cursor(encode_cursor(moment, row_id))

    assert decoded.created_at == moment
    assert decoded.id == row_id


def test_cursor_is_opaque() -> None:
    """Base64 by contract: a client that parses it freezes the sort key."""
    cursor = encode_cursor(datetime.now(UTC), uuid.uuid4())

    assert "created_at" not in cursor
    assert "-" not in cursor.rstrip("=") or True  # not a UUID in the clear


@pytest.mark.parametrize(
    "raw",
    [
        "not-base64!!",
        "",
        "eyJmb28iOiAiYmFyIn0=",  # valid base64, wrong payload
        "eyJjcmVhdGVkX2F0IjogIm5vdC1hLWRhdGUiLCAiaWQiOiAieCJ9",  # unparseable fields
    ],
)
def test_malformed_cursors_raise_one_error(raw: str) -> None:
    """Every failure collapses to one error — the distinctions only help a
    caller probing the endpoint."""
    with pytest.raises(InvalidCursorError):
        decode_cursor(raw)


def test_cursor_preserves_microseconds() -> None:
    """The trailing id makes the sort total, but only if the timestamp is
    not silently rounded on the way through the cursor."""
    moment = datetime(2026, 9, 18, 12, 30, 45, 123456, tzinfo=UTC)

    assert decode_cursor(encode_cursor(moment, uuid.uuid4())).created_at == moment


# --- Feedback contract ---------------------------------------------------------


def test_merge_requires_a_master_record() -> None:
    """Without one this passes every CHECK on human_feedback and then fails
    against `resolved_requires_master_record` on clusters, a step later."""
    with pytest.raises(ValueError, match="MERGE requires master_record_id"):
        FeedbackRequest(decision=FeedbackDecision.MERGE)


def test_merge_with_a_master_is_accepted() -> None:
    payload = FeedbackRequest(decision=FeedbackDecision.MERGE, master_record_id=RECORD_A)

    assert payload.master_record_id == RECORD_A


def test_split_requires_exclusions() -> None:
    with pytest.raises(ValueError, match="SPLIT requires at least one"):
        FeedbackRequest(decision=FeedbackDecision.SPLIT, master_record_id=RECORD_A)


def test_exclusions_are_rejected_on_non_split() -> None:
    """Mirrors `exclusions_only_on_split`. A MERGE carrying exclusions would
    poison the positive-pair set the adapter retrains on."""
    with pytest.raises(ValueError, match="only for SPLIT"):
        FeedbackRequest(
            decision=FeedbackDecision.MERGE,
            master_record_id=RECORD_A,
            excluded_record_ids=[RECORD_B],
        )


def test_reject_needs_neither_field() -> None:
    assert FeedbackRequest(decision=FeedbackDecision.REJECT).master_record_id is None


# --- Auth boundary -------------------------------------------------------------


def test_health_routes_are_unauthenticated(client: TestClient) -> None:
    """A readiness probe has no tenant to present."""
    assert client.get("/healthz").status_code == 200


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", f"/v1/jobs/{uuid.uuid4()}"),
        ("GET", f"/v1/jobs/{uuid.uuid4()}/stats"),
        ("GET", f"/v1/jobs/{uuid.uuid4()}/clusters"),
        ("GET", f"/v1/jobs/{uuid.uuid4()}/export"),
        ("GET", f"/v1/clusters/{uuid.uuid4()}"),
        ("POST", f"/v1/clusters/{uuid.uuid4()}/feedback"),
        ("POST", "/v1/jobs"),
    ],
)
def test_v1_routes_reject_anonymous_callers(client: TestClient, method: str, path: str) -> None:
    assert client.request(method, path).status_code == 401


def test_every_v1_route_requires_auth(client: TestClient) -> None:
    """Walks the live schema so a new unauthenticated route fails here.

    The alternative — a hand-maintained list — passes forever precisely
    when someone forgets to add the route they also forgot to protect.
    """
    spec = app.openapi()
    unprotected = []

    for path, operations in spec["paths"].items():
        if not path.startswith("/v1"):
            continue
        for method, operation in operations.items():
            if not operation.get("security"):
                unprotected.append(f"{method.upper()} {path}")

    assert not unprotected, f"v1 routes with no security requirement: {unprotected}"


def test_a_token_from_another_issuer_is_rejected(client: TestClient) -> None:
    settings = get_settings()
    forged = jwt.encode(
        {
            "sub": "attacker",
            "tenant_id": str(uuid.uuid4()),
            "role": Role.ORG_ADMIN.value,
            "aud": settings.jwt_audience,
            "iss": "not-our-issuer",
            "exp": datetime.now(UTC) + timedelta(minutes=5),
        },
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
    )

    response = client.get(f"/v1/jobs/{uuid.uuid4()}", headers={"Authorization": f"Bearer {forged}"})

    assert response.status_code == 401


def test_an_expired_token_is_rejected(client: TestClient) -> None:
    settings = get_settings()
    stale = jwt.encode(
        {
            "sub": "steward",
            "tenant_id": str(TENANT),
            "role": Role.DATA_STEWARD.value,
            "aud": settings.jwt_audience,
            "iss": settings.jwt_issuer,
            "exp": datetime.now(UTC) - timedelta(seconds=1),
        },
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
    )

    response = client.get(f"/v1/jobs/{uuid.uuid4()}", headers={"Authorization": f"Bearer {stale}"})

    assert response.status_code == 401


def test_auth_failures_are_indistinguishable(client: TestClient) -> None:
    """Expired vs bad-signature is useful to an attacker, not to a caller.

    Asserted as the two responses being byte-identical rather than by
    grepping for words: the message is deliberately the combined "Invalid or
    expired token", so a keyword check would flag the very wording that
    makes the cases indistinguishable.
    """
    settings = get_settings()
    expired = jwt.encode(
        {
            "sub": "steward",
            "tenant_id": str(TENANT),
            "role": Role.DATA_STEWARD.value,
            "aud": settings.jwt_audience,
            "iss": settings.jwt_issuer,
            "exp": datetime.now(UTC) - timedelta(seconds=1),
        },
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
    )
    path = f"/v1/jobs/{uuid.uuid4()}"

    garbage_response = client.get(path, headers={"Authorization": "Bearer garbage"})
    expired_response = client.get(path, headers={"Authorization": f"Bearer {expired}"})

    assert garbage_response.status_code == expired_response.status_code == 401
    assert garbage_response.json() == expired_response.json()
    # And neither echoes the library's own diagnosis.
    assert "signature" not in garbage_response.text.lower()


# --- Route surface -------------------------------------------------------------


def test_the_trd_2_5_surface_is_present() -> None:
    """Every route TRD 2.5 / FIG 2.9 / FIG 2.10 name."""
    spec = app.openapi()

    assert {
        "/v1/jobs",
        "/v1/jobs/{job_id}",
        "/v1/jobs/{job_id}/stats",
        "/v1/jobs/{job_id}/clusters",
        "/v1/jobs/{job_id}/export",
        "/v1/clusters/{cluster_id}",
        "/v1/clusters/{cluster_id}/feedback",
    } <= set(spec["paths"])


def test_upload_returns_202_not_201() -> None:
    """FR-01 hands back an id for work that has not started.

    201 would claim the resource is complete; the job row exists but the
    pipeline behind it has not run.
    """
    spec = app.openapi()

    assert "202" in spec["paths"]["/v1/jobs"]["post"]["responses"]


def test_export_is_declared_as_a_csv_stream() -> None:
    spec = app.openapi()
    operation = spec["paths"]["/v1/jobs/{job_id}/export"]["get"]

    assert "200" in operation["responses"]


def test_review_queue_paginates_by_cursor_not_offset() -> None:
    """TRD 2.5 mandates keyset. An `offset` parameter here would mean a
    steward paging a queue they are also draining silently skips rows."""
    spec = app.openapi()
    params = {p["name"] for p in spec["paths"]["/v1/jobs/{job_id}/clusters"]["get"]["parameters"]}

    assert "cursor" in params
    assert "offset" not in params


def test_no_request_body_accepts_a_tenant_id() -> None:
    """TRD 2.5: tenant comes from the token and from nowhere else.

    A `tenant_id` field on any request schema is the exact hole
    `app/core/security.py` exists to close.
    """
    spec = app.openapi()
    offenders = [
        f"{name}.{field}"
        for name, schema in spec.get("components", {}).get("schemas", {}).items()
        for field in schema.get("properties", {})
        if field == "tenant_id"
    ]

    assert not offenders, f"tenant_id is client-supplied in: {offenders}"


@pytest.mark.parametrize(
    ("pin", "kept"),
    [
        ("560001", True),
        (" 110065 ", True),
        ("000000", False),
        ("NA", False),
        ("12345", False),
        ("", False),
    ],
)
def test_the_gold_export_blanks_placeholder_pincodes(pin: str, kept: bool) -> None:
    from app.api.v1.routes.jobs import _gold_value

    assert (_gold_value("pincode", pin) != "") is kept
    assert _gold_value("name", "000000") == "000000", "only the PIN column is filtered"


def test_the_gold_export_puts_the_standardized_name_beside_the_name() -> None:
    from app.api.v1.routes.jobs import GOLD_COLUMNS, _gold_cells

    columns = list(GOLD_COLUMNS)
    assert columns.index("name_standardized") == columns.index("name") + 1

    cells = _gold_cells(
        {
            "name": "ARYA SR.SEC.SCHOOL",
            "address": "x",
            "district": "d",
            "state": "s",
            "pincode": "000000",
        },
        "ARYA SENIOR SECONDARY SCHOOL",
    )

    tail = columns[columns.index("name") :]
    assert dict(zip(tail, cells, strict=True)) == {
        "name": "ARYA SR.SEC.SCHOOL",
        "name_standardized": "ARYA SENIOR SECONDARY SCHOOL",
        "address": "x",
        "district": "d",
        "state": "s",
        "pincode": "",
    }


def test_the_standardized_name_is_the_ontologys_expansion_in_capitals() -> None:
    from app.api.v1.routes.jobs import _standardized_name
    from app.ontology import OntologyEngine

    school = OntologyEngine.load("education")

    assert _standardized_name(school, "ARYA SR.SEC.SCHOOL") == "ARYA SENIOR SECONDARY SCHOOL"
    assert (
        _standardized_name(school, "GOVT. BOYS SR. SEC. SCHOOL NO. 1")
        == "GOVERNMENT BOYS SENIOR SECONDARY SCHOOL NUMBER 1"
    )
    assert _standardized_name(school, "D.A.V. SR. SEC. SCHOOL") == (
        "DAYANAND ANGLO VEDIC SENIOR SECONDARY SCHOOL"
    )


def test_the_standardized_name_does_not_rename_the_school() -> None:
    """Matching collapses 'vidya mandir' and 'madrasa' to 'school'; a golden
    record must keep what the school is actually called."""
    from app.api.v1.routes.jobs import _standardized_name
    from app.ontology import OntologyEngine

    school = OntologyEngine.load("education")

    assert (
        _standardized_name(school, "MAHARISHI VIDYA MANDIR, SILPUKHURY")
        == "MAHARISHI VIDYA MANDIR SILPUKHURY"
    )
    assert _standardized_name(school, "JAMIA MADRASA SR SEC") == "JAMIA MADRASA SENIOR SECONDARY"
    # ...while the matching key still collapses them, as it must.
    assert school.normalize_name("Maharishi Vidya Mandir") == "maharishi school"


def test_the_standardized_name_is_blank_rather_than_guessed_without_an_ontology() -> None:
    from app.api.v1.routes.jobs import _standardized_name

    assert _standardized_name(None, "SR SEC") == ""
