"""
Calls the Gemini API to extract structured contract data from parsed
document text, using the schema + prompts defined in schema.py / prompts.py.

Uses Gemini's OpenAI-compatible endpoint via the `openai` client, so no
Google-specific SDK is required.

Setup required before this will run:
    1. Get a free API key at https://aistudio.google.com/apikey (no credit
       card required)
    2. Copy .env.example to .env and paste your key in as GEMINI_API_KEY
    3. pip install -r requirements.txt

This module does NOT read files or handle uploads itself -- it expects
already-parsed text (e.g. from parser.ParsedDocument.full_text).
"""

from __future__ import annotations

import json
import os

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import ValidationError

from prompts import build_messages, build_reduce_messages
from schema import ContractExtraction
from chunking import chunk_document, DocumentChunk
from merge import merge_extractions, MergeResult
from rate_limiter import TokenRateLimiter, estimate_tokens

load_dotenv()  # reads .env into environment variables if present

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
GEMINI_SIGNUP_HINT = "https://aistudio.google.com/apikey"

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

# How large a single chunk is allowed to be before chunking kicks in at all.
# Gemini's free-tier context window and token budget are generous enough
# that most real contracts (even 50+ pages) fit in ONE call -- chunking
# only activates past this, as a safety net for unusually long documents.
DEFAULT_CHUNK_MAX_CHARS = 180000

# Conservative token-per-minute ceiling used by the rate limiter. Free-tier
# numbers reported for Gemini vary noticeably across sources (roughly
# 250K-1M TPM depending on model/source, and Google has changed these more
# than once in 2026) -- 200,000 is set well below even the most
# conservative figure found, as headroom against both inaccurate reporting
# and future reductions. Check
# https://ai.google.dev/gemini-api/docs/rate-limits for current numbers.
DEFAULT_TPM_BUDGET = 200000

# Caps the model's OUTPUT length. A full extraction (~21 fields, each with
# an evidence quote, plus risk flags and a summary) can legitimately run
# past a couple thousand tokens on a dense contract -- too low a cap here
# causes the response to be cut off mid-string, which then fails JSON
# parsing (a truncated response is not valid JSON). 4000 leaves real
# headroom; Gemini's free-tier TPM budget comfortably absorbs it.
MAX_OUTPUT_TOKENS = 4000


class ExtractionError(Exception):
    """Raised when the LLM call fails or its output doesn't match the schema."""


def _get_client() -> OpenAI:
    api_key = os.getenv(GEMINI_API_KEY_ENV)
    if not api_key:
        raise ExtractionError(
            f"{GEMINI_API_KEY_ENV} is not set. Copy .env.example to .env and add "
            f"your key from {GEMINI_SIGNUP_HINT}."
        )
    return OpenAI(api_key=api_key, base_url=GEMINI_BASE_URL)


# ---------------------------------------------------------------------------
# Response sanitization
# ---------------------------------------------------------------------------
# Models occasionally deviate slightly from the requested JSON shape --
# e.g. an empty string ('') showing up inside a list of objects, or a bare
# string where {"value": ..., "evidence": ...} was expected. Rather than
# letting one malformed entry fail the entire extraction, we clean up
# known-safe deviations before schema validation. Anything that's still
# broken after this still surfaces as a normal ExtractionError.

_EXTRACTED_FIELD_NAMES = [
    "vendor_name", "customer_name", "contract_title",
    "effective_date", "term_end_date", "term_length", "auto_renewal",
    "renewal_notice_period_days", "contract_value", "pricing_model",
    "price_escalation_cap", "payment_terms", "license_metric",
    "true_up_rights", "audit_rights_present", "audit_notice_period_days",
    "audit_frequency", "termination_for_convenience", "termination_for_cause",
    "early_termination_fee", "sla_summary", "data_exit_transition_period",
]


def _sanitize_extraction_payload(data: dict) -> dict:
    if not isinstance(data, dict):
        return data

    # risk_flags must be a list of dicts -- drop anything else (e.g. stray
    # empty strings) rather than failing the whole response over it
    risk_flags = data.get("risk_flags")
    if isinstance(risk_flags, list):
        cleaned = []
        for item in risk_flags:
            if not isinstance(item, dict):
                continue  # drop malformed entries like ''
            item.setdefault("severity", "medium")
            item.setdefault("explanation", "")
            item.setdefault("clause", "Unspecified clause")
            if not isinstance(item.get("evidence"), dict):
                item["evidence"] = None
            cleaned.append(item)
        data["risk_flags"] = cleaned

    # fields_not_found must be a list of strings
    fnf = data.get("fields_not_found")
    if isinstance(fnf, list):
        data["fields_not_found"] = [f for f in fnf if isinstance(f, str)]

    # a field expected as {"value":..., "evidence":...} sometimes comes back
    # as a bare string -- wrap it rather than reject the whole payload
    for name in _EXTRACTED_FIELD_NAMES:
        val = data.get(name)
        if isinstance(val, str):
            data[name] = {"value": val, "evidence": None}
        elif isinstance(val, dict) and not isinstance(val.get("evidence"), (dict, type(None))):
            val["evidence"] = None

    return data


def extract_contract(
    document_text: str,
    document_filename: str = "uploaded contract",
    model: str = DEFAULT_MODEL,
    max_retries: int = 1,
) -> ContractExtraction:
    """
    Sends parsed contract text to Gemini and returns a validated
    ContractExtraction object.

    Raises ExtractionError if the API call fails, or if the model's output
    doesn't validate against the schema after retries and sanitization.
    """
    client = _get_client()
    messages = build_messages(document_text, document_filename)

    last_error: Exception | None = None
    raw_content = ""

    for attempt in range(1, max_retries + 2):  # e.g. max_retries=1 -> try twice total
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,               # deterministic extraction, not creative writing
                max_tokens=MAX_OUTPUT_TOKENS,  # see comment above -- avoids mid-string truncation
                response_format={"type": "json_object"},  # forces valid JSON output
            )
            raw_content = response.choices[0].message.content
            # strict=False tolerates literal control characters (e.g. raw
            # newlines) inside JSON string values, which some models emit
            # unescaped despite being asked for valid JSON.
            parsed_json = json.loads(raw_content, strict=False)
            parsed_json = _sanitize_extraction_payload(parsed_json)
            return ContractExtraction.model_validate(parsed_json)

        except json.JSONDecodeError as e:
            last_error = ExtractionError(f"Model did not return valid JSON: {e}")
        except ValidationError as e:
            last_error = ExtractionError(f"Model output didn't match the expected schema: {e}")
        except Exception as e:  # covers API/network errors from the SDK
            error_str = str(e)
            if "model_not_found" in error_str or "does not exist" in error_str or "404" in error_str:
                last_error = ExtractionError(
                    f"The model '{model}' is not available. Update GEMINI_MODEL in "
                    f"your .env file -- check https://ai.google.dev/gemini-api/docs/models "
                    f"for the current list. (Original error: {e})"
                )
            elif "rate_limit_exceeded" in error_str or "tokens per minute" in error_str or "429" in error_str:
                last_error = ExtractionError(
                    f"Rate limit hit on Gemini. Try again in a minute, or try a "
                    f"shorter document. (Original error: {e})"
                )
            else:
                last_error = ExtractionError(f"Gemini API call failed: {e}")

        if attempt <= max_retries:
            # On retry, nudge the model to fix its own output rather than
            # resending the whole prompt from scratch.
            messages = messages + [
                {"role": "assistant", "content": raw_content},
                {"role": "user", "content": (
                    "That response was invalid. Return ONLY a single valid JSON "
                    "object matching the required schema -- no markdown, no "
                    "commentary, no trailing text, and every list item must be "
                    "a proper object, never an empty string."
                )},
            ]

    raise last_error  # all attempts exhausted


def _run_reduce_summary(merged_extraction: ContractExtraction, model: str, limiter: TokenRateLimiter) -> str:
    """
    Reduce step: one cheap LLM call over the already-merged structured facts
    (not the raw document again) to produce one coherent summary, instead of
    just concatenating each chunk's separate summary.
    """
    client = _get_client()

    facts = {
        name: (getattr(merged_extraction, name).value if getattr(merged_extraction, name, None) else None)
        for name in _EXTRACTED_FIELD_NAMES
    }
    risk_flags = [
        {"severity": f.severity, "clause": f.clause, "explanation": f.explanation}
        for f in merged_extraction.risk_flags
    ]

    messages = build_reduce_messages(facts, risk_flags)
    estimated = sum(estimate_tokens(m["content"]) for m in messages) + 300  # + output headroom
    limiter.reserve(estimated)

    response = client.chat.completions.create(
        model=model, messages=messages, temperature=0, max_tokens=300,
    )
    return response.choices[0].message.content.strip()


def extract_contract_chunked(
    document_text_or_doc,
    document_filename: str = "uploaded contract",
    model: str = DEFAULT_MODEL,
    tpm_budget: int = DEFAULT_TPM_BUDGET,
    chunk_max_chars: int = DEFAULT_CHUNK_MAX_CHARS,
    progress_callback=None,
) -> tuple[ContractExtraction, MergeResult]:
    """
    Map-reduce extraction for documents of any size.

    - MAP: the document is split into overlapping chunks (chunking.py).
      Given Gemini's generous free-tier budget, most real contracts fit in
      a single chunk -- chunking only activates for unusually long
      documents. Each chunk is sent through extract_contract(), spaced out
      by a rate limiter so consecutive chunk calls don't collectively bust
      the per-minute token budget.
    - REDUCE: per-chunk results are merged structurally (merge.py) -- most
      fields are simple "first non-null value wins" since a fact either
      appears once in the document or it doesn't. Risk flags are deduped.
      Only the final summary gets a dedicated LLM synthesis call, run over
      the compact merged facts rather than the raw text again.

    For a document that fits in a single chunk, this is exactly one
    extract_contract() call plus no reduce call -- no overhead added for
    the common case.

    `progress_callback(current_chunk: int, total_chunks: int)` is optional,
    for a UI to show "Extracting section 2 of 4...".

    Accepts either a parser.ParsedDocument (preferred -- enables page-aware
    chunking) or a plain string (falls back to paragraph-based chunking).
    """
    from parser import ParsedDocument, PageResult  # local import avoids a hard dependency for text-only callers

    if isinstance(document_text_or_doc, ParsedDocument):
        doc = document_text_or_doc
    else:
        # wrap a plain string as a single-page ParsedDocument so chunk_document
        # can treat it uniformly
        doc = ParsedDocument(filename=document_filename, file_type="text")
        doc.pages.append(PageResult(page_number=1, text=document_text_or_doc, source="native"))

    chunks: list[DocumentChunk] = chunk_document(doc, max_chars=chunk_max_chars)
    limiter = TokenRateLimiter(tpm_budget=tpm_budget)

    chunk_results: list[ContractExtraction] = []
    for chunk in chunks:
        if progress_callback:
            progress_callback(chunk.index + 1, chunk.total_chunks)

        messages = build_messages(chunk.text, document_filename)
        estimated = sum(estimate_tokens(m["content"]) for m in messages) + MAX_OUTPUT_TOKENS
        limiter.reserve(estimated)

        result = extract_contract(chunk.text, document_filename, model=model)
        chunk_results.append(result)

    merge_result = merge_extractions(chunk_results)

    if merge_result.chunk_count > 1:
        # reduce step: one extra call for a coherent final summary
        try:
            merge_result.merged.plain_english_summary = _run_reduce_summary(
                merge_result.merged, model, limiter
            )
        except Exception:
            # non-fatal -- fall back to the concatenated summary merge.py
            # already produced rather than failing the whole extraction
            pass

    return merge_result.merged, merge_result


if __name__ == "__main__":
    # Manual smoke test -- only runs if you execute this file directly AND
    # have a real GEMINI_API_KEY set. Not run automatically anywhere else.
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from parser import parse_document

    sample_pdf = Path(__file__).parent / "tests" / "fixtures" / "microsoft_native.pdf"
    if not sample_pdf.exists():
        print("Run tests/make_test_files.py first to generate sample fixtures.")
        sys.exit(1)

    print(f"Model: {DEFAULT_MODEL}  |  TPM budget: {DEFAULT_TPM_BUDGET}")
    doc = parse_document(sample_pdf)
    result, merge_info = extract_contract_chunked(doc, document_filename=sample_pdf.name)
    print(f"Chunks used: {merge_info.chunk_count}")
    print(result.model_dump_json(indent=2))
