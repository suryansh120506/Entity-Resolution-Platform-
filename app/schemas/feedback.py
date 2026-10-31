"""Steward feedback — FIG 2.10 / FR-08.

The validation here is the interesting part. `human_feedback` carries three
CHECK constraints that encode which field combinations are meaningful, and
the same rules are enforced at the edge so a steward gets a 422 explaining
what was wrong instead of a 500 from a constraint violation. The database
keeps its constraints regardless: this layer is for the message, not for the
guarantee.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field, model_validator

from app.ml_pipeline.fields import REQUIRED_FIELDS as CORRECTABLE_FIELDS
from app.models.enums import FeedbackDecision


class FeedbackRequest(BaseModel):
    """A steward's verdict on one cluster (FIG 2.10 request body)."""

    decision: FeedbackDecision
    master_record_id: uuid.UUID | None = Field(
        default=None,
        description="The surviving record. Required for MERGE.",
    )
    excluded_record_ids: list[uuid.UUID] | None = Field(
        default=None,
        description="Records pulled out of the cluster. SPLIT only.",
    )
    field_corrections: dict[str, str] | None = Field(
        default=None,
        description=(
            "Corrected values for the surviving record — any of name, address, "
            "district, state, pincode. MERGE only. Stored beside the record; the "
            "uploaded row itself is never edited."
        ),
    )

    @model_validator(mode="after")
    def _check_shape(self) -> FeedbackRequest:
        """Mirror the three CHECKs on `human_feedback`.

        `exclusions_only_on_split` and `split_requires_exclusions` are the
        database's; the MERGE rule is this layer's own. A MERGE with no
        master would satisfy every constraint on the feedback table and then
        fail later against `resolved_requires_master_record` on `clusters`,
        one step too far from the request that caused it.
        """
        if self.decision is FeedbackDecision.MERGE and self.master_record_id is None:
            raise ValueError("MERGE requires master_record_id: something has to survive.")

        if self.decision is FeedbackDecision.SPLIT and not self.excluded_record_ids:
            raise ValueError("SPLIT requires at least one entry in excluded_record_ids.")

        if self.decision is not FeedbackDecision.SPLIT and self.excluded_record_ids:
            raise ValueError(
                "excluded_record_ids is meaningful only for SPLIT — on a MERGE it "
                "would poison the positive-pair set the adapter retrains on."
            )

        if self.field_corrections:
            if self.decision is not FeedbackDecision.MERGE:
                raise ValueError(
                    "field_corrections apply to the record that survives a MERGE; "
                    "a REJECT or SPLIT has no single survivor to correct."
                )
            unknown = set(self.field_corrections) - set(CORRECTABLE_FIELDS)
            if unknown:
                raise ValueError(
                    f"Cannot correct {sorted(unknown)}; correctable fields are "
                    f"{list(CORRECTABLE_FIELDS)}."
                )
        return self


class FeedbackResponse(BaseModel):
    """FIG 2.10's response: what was recorded, and where the cluster landed."""

    feedback_id: uuid.UUID
    cluster_id: uuid.UUID
    decision: FeedbackDecision
    cluster_status: str
    created_at: datetime
