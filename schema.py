"""
Structured extraction schema for SAM vendor contracts.

This defines exactly what fields the LLM must return. Every field is
Optional because contracts vary wildly -- a field genuinely not present in
the source document should come back as null, never guessed.

Each extracted fact also carries an `evidence` object (page number + a short
verbatim snippet) so the field can be traced back to the source text. This
is what lets the tool avoid silently hallucinating a renewal date or a
dollar figure.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class Evidence(BaseModel):
    page: Optional[int] = Field(None, description="Page number the fact was found on")
    quote: Optional[str] = Field(
        None,
        description="Short verbatim snippet (<= 25 words) from the source text supporting this field",
    )


class ExtractedField(BaseModel):
    """Wraps any extracted value together with its source evidence."""
    value: Optional[str] = None
    evidence: Optional[Evidence] = None


class RiskFlag(BaseModel):
    clause: str = Field(..., description="Short name of the clause, e.g. 'Auto-renewal'")
    severity: str = Field(..., description="One of: low, medium, high")
    explanation: str = Field(..., description="One sentence on why this is flagged")
    evidence: Optional[Evidence] = None


class ProductLineItem(BaseModel):
    """
    One line item from a contract's order form / license schedule / SKU
    table -- the actual products/licenses being purchased, as opposed to
    the top-level contract terms captured elsewhere in this schema.

    Uses a single evidence reference per row (rather than per field, like
    the top-level fields do) since a line item's fields typically all come
    from the same table row -- one page/quote pointing at that row is
    enough to verify the whole line, and asking for 9 separate evidence
    quotes per product would balloon token usage for little extra value.
    """
    publisher_part_number: Optional[str] = None
    product_name: Optional[str] = None
    license_type: Optional[str] = Field(None, description="'subscription' | 'perpetual' | 'unclear'")
    license_metric: Optional[str] = Field(None, description="e.g. per-user, per-core, concurrent")
    purchased_rights: Optional[str] = Field(None, description="quantity/entitlement, e.g. '500 users', '100 cores'")
    unit_cost: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    country_of_agreement: Optional[str] = None
    evidence: Optional[Evidence] = None


class ContractExtraction(BaseModel):
    # --- Identification ---
    vendor_name: Optional[ExtractedField] = None
    customer_name: Optional[ExtractedField] = None
    contract_title: Optional[ExtractedField] = None

    # --- Term & renewal ---
    effective_date: Optional[ExtractedField] = None
    term_end_date: Optional[ExtractedField] = None
    term_length: Optional[ExtractedField] = None
    auto_renewal: Optional[ExtractedField] = None            # "yes" / "no" / "unclear"
    renewal_notice_period_days: Optional[ExtractedField] = None

    # --- Commercial terms ---
    contract_value: Optional[ExtractedField] = None
    pricing_model: Optional[ExtractedField] = None            # per-seat, per-core, flat fee, etc.
    price_escalation_cap: Optional[ExtractedField] = None
    payment_terms: Optional[ExtractedField] = None

    # --- License / entitlement ---
    license_metric: Optional[ExtractedField] = None           # per-user, per-device, concurrent, etc.
    true_up_rights: Optional[ExtractedField] = None

    # --- Compliance & risk ---
    audit_rights_present: Optional[ExtractedField] = None     # "yes" / "no" / "unclear"
    audit_notice_period_days: Optional[ExtractedField] = None
    audit_frequency: Optional[ExtractedField] = None

    # --- Termination ---
    termination_for_convenience: Optional[ExtractedField] = None
    termination_for_cause: Optional[ExtractedField] = None
    early_termination_fee: Optional[ExtractedField] = None

    # --- SLA & exit ---
    sla_summary: Optional[ExtractedField] = None
    data_exit_transition_period: Optional[ExtractedField] = None

    # --- Product / line-item table ---
    products: list[ProductLineItem] = Field(
        default_factory=list,
        description="Line items from an order form, license schedule, or SKU table, if present",
    )

    # --- Narrative + risk ---
    plain_english_summary: str = Field(..., description="3-5 sentence plain-English summary of the contract")
    risk_flags: list[RiskFlag] = Field(default_factory=list)
    fields_not_found: list[str] = Field(
        default_factory=list,
        description="Names of schema fields that could not be found in the document at all",
    )
