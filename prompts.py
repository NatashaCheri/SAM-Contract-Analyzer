"""
Prompt design for SAM vendor contract extraction.

Two prompts, two jobs:
  - SYSTEM_PROMPT: sets the rules of engagement ONCE (role, grounding
    requirements, how to handle missing info, output format). This rarely
    changes between calls.
  - build_user_prompt(): carries the actual schema + the document text for
    THIS specific contract. Changes every call.

Keeping them separate means you can tune extraction behavior (system prompt)
independently of what you're asking it to extract (schema/user prompt).
"""

import json
from schema import ContractExtraction

# ---------------------------------------------------------------------------
# SYSTEM PROMPT
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a contract data extraction engine specialised in Software \
Asset Management (SAM) vendor contracts -- software license agreements, SaaS \
subscription terms, and enterprise service agreements.

Your job is to read the contract text provided by the user and extract a fixed \
set of structured fields as JSON. You are not a lawyer and you do not give legal \
advice or interpretation beyond what the text literally states.

## Grounding rules (critical)

1. Every fact you extract MUST be traceable to the source text. For each field, \
provide the page number and a short verbatim quote (<=25 words) that supports it.
2. If a field is not present in the document, or you are not confident it is \
correct, set its value to null. Do NOT infer, estimate, or guess a plausible \
value. A missing field is a normal and expected outcome -- most contracts will \
NOT mention most fields.
3. Never invent a quote. If you cannot find a supporting quote in the text, do \
not populate that field.
4. Do not average, calculate, or extrapolate numbers (e.g. do not infer an \
annual price from a monthly one unless the text states the annual figure).
5. If the document is a general Terms of Use / consumer agreement rather than a \
negotiated enterprise contract, many enterprise-specific fields (contract \
value, true-up rights, SLA) will legitimately be absent -- return null for \
those rather than fabricating typical values.

## Field shape

Every extracted field (except plain_english_summary, risk_flags, products and \
fields_not_found) uses this exact shape:
    {"value": <string or null>, "evidence": {"page": <int or null>, "quote": <string or null>} or null}
If value is null, evidence should also be null.

## Product / line-item table

Many contracts include an order form, license schedule, quote, or SKU table \
listing the specific products/licenses being purchased -- separate from the \
general contract terms. If ANY such table or list is present, extract EVERY \
row into the `products` array, one entry per distinct product/SKU line. \
Do not summarise or merge rows together -- if the source lists 5 line items, \
return 5 entries. If no such table exists in the document, return an empty \
array -- do not invent line items from general contract language.
For `license_type`, use exactly 'subscription', 'perpetual', or 'unclear'.
Each product entry needs only ONE evidence reference (not one per field), \
pointing at the table row it came from.

## Output rules

- Respond with ONLY a single JSON object. No preamble, no markdown code fences, \
no explanation before or after.
- Be concise: evidence quotes must stay under 25 words, and \
plain_english_summary must stay under 5 sentences. Do not pad output length.
- The JSON must conform exactly to the schema provided in the user message \
(same field names, same nesting).
- `plain_english_summary` and `risk_flags` are the only fields that involve \
your own synthesis/judgment rather than direct extraction -- everything else \
must be grounded in an exact quote.
- For `risk_flags`, only flag clauses that are genuinely present in the text \
(e.g. auto-renewal with a short notice window, broad audit rights, uncapped \
price escalation, high early-termination fees). Do not flag the absence of a \
clause as a risk unless the absence itself is unusual for this contract type. \
Limit to the 5 most significant risks.
- List every schema field you could not find any evidence for in \
`fields_not_found`, using its exact field name.
"""


# ---------------------------------------------------------------------------
# USER PROMPT (schema + document, built per-call)
# ---------------------------------------------------------------------------

def _schema_skeleton() -> dict:
    """
    Compact skeleton: shows the {value, evidence} shape fully worked out ONCE
    (vendor_name), then lists the remaining fields as bare names -- the
    system prompt's "Field shape" section tells the model to apply the same
    shape to all of them. Repeating the full nested example 20+ times was
    burning a large, unnecessary chunk of the prompt token budget for zero
    extra clarity.
    """
    worked_example = {
        "value": "string or null",
        "evidence": {"page": "int or null", "quote": "verbatim quote, <=25 words, or null"},
    }

    same_shape_fields = [
        "customer_name", "contract_title", "effective_date", "term_end_date",
        "term_length", "auto_renewal (value: 'yes'|'no'|'unclear')",
        "renewal_notice_period_days", "contract_value", "pricing_model",
        "price_escalation_cap", "payment_terms", "license_metric",
        "true_up_rights", "audit_rights_present (value: 'yes'|'no'|'unclear')",
        "audit_notice_period_days", "audit_frequency",
        "termination_for_convenience", "termination_for_cause",
        "early_termination_fee", "sla_summary", "data_exit_transition_period",
    ]

    return {
        "vendor_name": worked_example,
        "<all fields below use the SAME {value, evidence} shape as vendor_name above>": same_shape_fields,
        "products": [
            {
                "publisher_part_number": "string or null",
                "product_name": "string or null",
                "license_type": "'subscription' | 'perpetual' | 'unclear' | null",
                "license_metric": "string or null",
                "purchased_rights": "string or null, e.g. '500 users'",
                "unit_cost": "string or null",
                "start_date": "string or null",
                "end_date": "string or null",
                "country_of_agreement": "string or null",
                "evidence": {"page": "int or null", "quote": "string or null"},
            }
        ],
        "plain_english_summary": "string, <=5 sentences",
        "risk_flags": [
            {
                "clause": "short clause name",
                "severity": "'low' | 'medium' | 'high'",
                "explanation": "one sentence",
                "evidence": {"page": "int or null", "quote": "string or null"},
            }
        ],
        "fields_not_found": ["exact field names with no evidence in the document"],
    }


# If the document text is very long, truncate it before building the prompt.
# This is a blunt safety net for free-tier token limits -- a proper chunking
# strategy (splitting by section and extracting per-chunk) is the correct
# long-term fix for large enterprise contracts, but for v1 this at least
# prevents a single huge document from silently blowing the request budget.
MAX_DOCUMENT_CHARS = 14000


def build_user_prompt(document_text: str, document_filename: str = "uploaded contract") -> str:
    schema_json = json.dumps(_schema_skeleton(), separators=(",", ":"))

    truncated = False
    if len(document_text) > MAX_DOCUMENT_CHARS:
        document_text = document_text[:MAX_DOCUMENT_CHARS]
        truncated = True

    truncation_note = (
        "\n\n[NOTE: document truncated to fit token limits -- content after this "
        "point was not analyzed. Fields only found later in the document may be "
        "incorrectly reported as not found.]"
        if truncated else ""
    )

    return f"""Extract structured data from the following contract: {document_filename}

## Required JSON output shape

{schema_json}

## Contract text (page markers preserved as "--- Page N ---")

{document_text}{truncation_note}

Return only the JSON object described above, populated from the contract text.
"""


# ---------------------------------------------------------------------------
# Convenience: assemble the full message list for an API call
# ---------------------------------------------------------------------------

def build_messages(document_text: str, document_filename: str = "uploaded contract") -> list[dict]:
    """Returns a messages list ready to hand to Gemini's OpenAI-compatible
    chat completions endpoint."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(document_text, document_filename)},
    ]


# ---------------------------------------------------------------------------
# REDUCE-STEP PROMPT (map-reduce synthesis for chunked/large documents)
# ---------------------------------------------------------------------------

REDUCE_SYSTEM_PROMPT = """You write a single coherent plain-English summary of a \
vendor contract from a structured list of facts already extracted from it. You \
are not extracting anything new -- only synthesising the facts you're given \
into one clear summary, in 3-5 sentences. Do not mention that the facts came \
from multiple sections or a merge process; write it as one normal summary. \
Do not add any fact not present in the input. Respond with plain text only, \
no JSON, no markdown."""


def build_reduce_messages(merged_facts: dict, risk_flags: list[dict]) -> list[dict]:
    """
    Builds the prompt for the reduce step: a short synthesis call over
    already-merged structured facts (cheap -- no raw document text here),
    used when a document was large enough to require chunking.
    """
    facts_lines = [f"- {k.replace('_', ' ')}: {v}" for k, v in merged_facts.items() if v]
    risk_lines = [f"- [{r['severity']}] {r['clause']}: {r['explanation']}" for r in risk_flags]

    content = "Facts extracted from the contract:\n" + "\n".join(facts_lines)
    if risk_lines:
        content += "\n\nRisk flags identified:\n" + "\n".join(risk_lines)
    content += "\n\nWrite the 3-5 sentence summary now."

    return [
        {"role": "system", "content": REDUCE_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


if __name__ == "__main__":
    # Quick sanity check: print prompt length and confirm schema round-trips
    sample_text = "--- Page 1 ---\nThis is a placeholder contract body for prompt length testing."
    messages = build_messages(sample_text, "sample.pdf")
    print(f"System prompt: {len(SYSTEM_PROMPT)} chars")
    print(f"User prompt:   {len(messages[1]['content'])} chars")
    print("\n--- Rendered user prompt preview ---\n")
    print(messages[1]["content"][:600], "...")
