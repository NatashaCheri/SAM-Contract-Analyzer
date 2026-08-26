"""
SAM Vendor Contract Analyzer -- Streamlit app.

Flow: upload -> parse -> extract -> review -> download PDF -> done.

Design notes:
  - No persistent storage anywhere. The uploaded file lives in a per-session
    temp directory (parser.SessionWorkspace) that is deleted immediately
    after the text is extracted from it. Everything after that point lives
    only in st.session_state, which is wiped when the browser tab closes.
  - Every external failure point (bad file, parsing error, missing API key,
    LLM/network error, schema mismatch) is caught and shown to the user as
    a plain-language message with a next action, never a raw traceback.
"""

from __future__ import annotations

import os

import streamlit as st

# ---------------------------------------------------------------------------
# Bridge Streamlit Cloud's secrets manager into environment variables
# ---------------------------------------------------------------------------
# Locally, extract.py reads GEMINI_API_KEY / GEMINI_MODEL via os.getenv(),
# populated from a local .env file (see extract.py's load_dotenv() call).
# Streamlit Community Cloud has no .env file -- secrets are set through its
# own dashboard (Settings -> Secrets) and exposed via st.secrets instead.
# Copying them into os.environ here means extract.py's os.getenv() calls
# work identically in both places, with zero branching logic needed.
# Wrapped in try/except because st.secrets raises if no secrets.toml exists
# at all (the normal case for local development, where .env is used instead).
try:
    for _key in ("GEMINI_API_KEY", "GEMINI_MODEL"):
        if _key in st.secrets:
            os.environ[_key] = st.secrets[_key]
except Exception:
    pass  # no secrets.toml locally -- .env (loaded by extract.py) is used instead

from parser import SessionWorkspace, parse_document
from extract import extract_contract_chunked, ExtractionError
from report_generator import generate_pdf_report
from schema import ContractExtraction

# ---------------------------------------------------------------------------
# Page setup + visual identity
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="SAM Contract Analyzer",
    page_icon="📑",
    layout="centered",
    initial_sidebar_state="collapsed",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

h1, h2, h3 {
    font-family: 'Source Serif 4', serif !important;
    color: #1C2B33 !important;
    letter-spacing: -0.01em;
}

.app-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #2B6E63;
    margin-bottom: 0.15rem;
}

.app-subtitle {
    color: #5B6B66;
    font-size: 0.95rem;
    margin-top: -0.4rem;
    margin-bottom: 1.6rem;
}

/* Step indicator */
.step-row { display: flex; gap: 0.5rem; margin-bottom: 1.8rem; }
.step-pill {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.06em;
    padding: 0.3rem 0.7rem;
    border-radius: 999px;
    border: 1px solid #D7DED9;
    color: #5B6B66;
    background: #FFFFFF;
}
.step-pill.active {
    background: #2B6E63;
    border-color: #2B6E63;
    color: #FFFFFF;
}
.step-pill.done {
    background: #EFF2EE;
    border-color: #2B6E63;
    color: #2B6E63;
}

/* Field cards */
.field-card {
    background: #FFFFFF;
    border: 1px solid #D7DED9;
    border-radius: 6px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.6rem;
}
.field-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #5B6B66;
    margin-bottom: 0.2rem;
}
.field-value {
    font-size: 1rem;
    color: #1C2B33;
    font-weight: 500;
}
.field-evidence {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.74rem;
    color: #8A968F;
    margin-top: 0.35rem;
    font-style: italic;
}

/* Risk stamp badges */
.stamp {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.15rem 0.55rem;
    border: 1.5px solid currentColor;
    border-radius: 3px;
    transform: rotate(-1.5deg);
    margin-right: 0.6rem;
}
.stamp-high { color: #B23A32; }
.stamp-medium { color: #C98A2C; }
.stamp-low { color: #3F7D5C; }

.risk-row {
    display: flex;
    align-items: flex-start;
    gap: 0.3rem;
    padding: 0.6rem 0;
    border-bottom: 1px solid #D7DED9;
}

section[data-testid="stFileUploaderDropzone"] {
    background: #FFFFFF;
    border: 1.5px dashed #2B6E63 !important;
}

div.stButton > button, div.stDownloadButton > button {
    background-color: #2B6E63;
    color: #FFFFFF;
    border-radius: 6px;
    border: none;
    font-weight: 500;
    padding: 0.55rem 1.4rem;
}
div.stButton > button:hover, div.stDownloadButton > button:hover {
    background-color: #23584F;
    color: #FFFFFF;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "stage" not in st.session_state:
    st.session_state.stage = "upload"       # upload -> analyzing -> results -> error
if "extraction" not in st.session_state:
    st.session_state.extraction = None
if "source_filename" not in st.session_state:
    st.session_state.source_filename = None
if "error_message" not in st.session_state:
    st.session_state.error_message = None
if "merge_conflicts" not in st.session_state:
    st.session_state.merge_conflicts = None


def reset_session():
    st.session_state.stage = "upload"
    st.session_state.extraction = None
    st.session_state.source_filename = None
    st.session_state.error_message = None
    st.session_state.merge_conflicts = None


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown('<div class="app-eyebrow">Software Asset Management</div>', unsafe_allow_html=True)
st.markdown("# Vendor Contract Analyzer")
st.markdown(
    '<div class="app-subtitle">Upload a vendor contract to extract renewal terms, '
    "audit rights, and risk flags. Nothing is stored -- your file and results exist "
    "only for this session.</div>",
    unsafe_allow_html=True,
)

steps = ["upload", "analyzing", "results"]
step_labels = {"upload": "1 · Upload", "analyzing": "2 · Analyze", "results": "3 · Review & Download"}
current_index = steps.index(st.session_state.stage) if st.session_state.stage in steps else 0
pills_html = '<div class="step-row">'
for i, s in enumerate(steps):
    cls = "active" if i == current_index else ("done" if i < current_index else "")
    pills_html += f'<div class="step-pill {cls}">{step_labels[s]}</div>'
pills_html += "</div>"
st.markdown(pills_html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Stage: UPLOAD
# ---------------------------------------------------------------------------

if st.session_state.stage == "upload":
    uploaded_file = st.file_uploader(
        "Contract file (PDF or DOCX, max 25MB)",
        type=["pdf", "docx"],
        help="Scanned/image-only PDFs are supported -- OCR runs automatically.",
    )

    analyze_clicked = st.button("Analyze contract", disabled=uploaded_file is None, type="primary")

    if analyze_clicked and uploaded_file is not None:
        st.session_state.stage = "analyzing"
        st.session_state.source_filename = uploaded_file.name
        st.session_state._pending_bytes = uploaded_file.getvalue()
        st.rerun()


# ---------------------------------------------------------------------------
# Stage: ANALYZING (parse -> extract, with error handling at each step)
# ---------------------------------------------------------------------------

elif st.session_state.stage == "analyzing":
    filename = st.session_state.source_filename
    file_bytes = st.session_state.get("_pending_bytes")

    with st.status(f"Analyzing {filename}...", expanded=True) as status:
        try:
            # --- Parse (file touches disk only inside this temp workspace) ---
            st.write("Reading document...")
            with SessionWorkspace() as ws:
                saved_path = ws.save_upload(filename, file_bytes)
                parsed = parse_document(saved_path)
            # workspace is deleted here -- original file no longer exists on disk

            if parsed.char_count < 20:
                raise ValueError(
                    "No readable text could be found in this document, even after "
                    "OCR. The file may be empty, corrupted, or a type of scan OCR "
                    "couldn't handle."
                )

            if parsed.ocr_page_count > 0:
                st.write(f"Used OCR on {parsed.ocr_page_count} scanned page(s).")

            # --- Extract via LLM (map-reduce for large documents) ---
            def _progress(current, total):
                if total > 1:
                    st.write(f"Extracting section {current} of {total}...")

            extraction: ContractExtraction
            extraction, merge_info = extract_contract_chunked(
                parsed, filename, progress_callback=_progress
            )

            if merge_info.chunk_count > 1:
                st.write(f"Merged results from {merge_info.chunk_count} sections.")
                if merge_info.conflicts:
                    st.session_state.merge_conflicts = merge_info.conflicts

            st.session_state.extraction = extraction
            st.session_state.stage = "results"
            status.update(label="Analysis complete", state="complete")
            st.rerun()

        except ValueError as e:
            # Bad/unreadable file
            st.session_state.error_message = (
                f"**Couldn't read this file.** {e}\n\n"
                "Try re-exporting the PDF or uploading a different copy."
            )
            st.session_state.stage = "error"
            status.update(label="Analysis failed", state="error")
            st.rerun()

        except ExtractionError as e:
            # Missing API key / LLM call failed / schema mismatch
            st.session_state.error_message = (
                f"**Contract extraction failed.** {e}\n\n"
                "If this is your first time running the app, check that `GEMINI_API_KEY` "
                "is set in your `.env` file. If the key is set, this may be a temporary "
                "issue with the extraction service -- try again in a moment."
            )
            st.session_state.stage = "error"
            status.update(label="Analysis failed", state="error")
            st.rerun()

        except Exception as e:
            # Catch-all so the user never sees a raw traceback
            st.session_state.error_message = (
                f"**Something unexpected went wrong.** ({type(e).__name__}: {e})\n\n"
                "Please try again. If this keeps happening, try a different file."
            )
            st.session_state.stage = "error"
            status.update(label="Analysis failed", state="error")
            st.rerun()


# ---------------------------------------------------------------------------
# Stage: ERROR
# ---------------------------------------------------------------------------

elif st.session_state.stage == "error":
    st.error(st.session_state.error_message)
    if st.button("Try again"):
        reset_session()
        st.rerun()


# ---------------------------------------------------------------------------
# Stage: RESULTS
# ---------------------------------------------------------------------------

elif st.session_state.stage == "results":
    extraction: ContractExtraction = st.session_state.extraction
    filename = st.session_state.source_filename

    vendor_label = extraction.vendor_name.value if extraction.vendor_name and extraction.vendor_name.value else "Vendor"
    st.markdown(f"### {vendor_label}")
    st.caption(f"Source: {filename}")

    st.markdown(extraction.plain_english_summary)

    if st.session_state.merge_conflicts:
        with st.expander("⚠️ Some fields had conflicting values across sections", expanded=False):
            st.caption(
                "This document was long enough to be analyzed in sections. The fields below "
                "returned different values in different sections -- the first value found is "
                "shown above; double-check these against the original document."
            )
            for field_name, values in st.session_state.merge_conflicts.items():
                st.write(f"**{field_name.replace('_', ' ')}**: {', '.join(values)}")

    # --- Risk flags ---
    if extraction.risk_flags:
        st.markdown("#### Risk Flags")
        for flag in extraction.risk_flags:
            sev = flag.severity.lower()
            stamp_class = f"stamp stamp-{sev}" if sev in ("high", "medium", "low") else "stamp"
            st.markdown(
                f'<div class="risk-row"><span class="{stamp_class}">{sev}</span>'
                f"<div><b>{flag.clause}</b> &mdash; {flag.explanation}</div></div>",
                unsafe_allow_html=True,
            )
        st.write("")

    # --- Field sections ---
    def render_field(label: str, field):
        if field is None or not field.value:
            return
        evidence_html = ""
        if field.evidence and field.evidence.quote:
            page_note = f" &middot; p.{field.evidence.page}" if field.evidence.page else ""
            evidence_html = f'<div class="field-evidence">"{field.evidence.quote}"{page_note}</div>'
        st.markdown(
            f'<div class="field-card"><div class="field-label">{label}</div>'
            f'<div class="field-value">{field.value}</div>{evidence_html}</div>',
            unsafe_allow_html=True,
        )

    sections = [
        ("Term & Renewal", [
            ("Effective date", extraction.effective_date),
            ("Term end date", extraction.term_end_date),
            ("Term length", extraction.term_length),
            ("Auto-renewal", extraction.auto_renewal),
            ("Renewal notice period", extraction.renewal_notice_period_days),
        ]),
        ("Commercial Terms", [
            ("Contract value", extraction.contract_value),
            ("Pricing model", extraction.pricing_model),
            ("Price escalation cap", extraction.price_escalation_cap),
            ("Payment terms", extraction.payment_terms),
        ]),
        ("License & Entitlement", [
            ("License metric", extraction.license_metric),
            ("True-up rights", extraction.true_up_rights),
        ]),
        ("Compliance & Audit", [
            ("Audit rights present", extraction.audit_rights_present),
            ("Audit notice period", extraction.audit_notice_period_days),
            ("Audit frequency", extraction.audit_frequency),
        ]),
        ("Termination", [
            ("Termination for convenience", extraction.termination_for_convenience),
            ("Termination for cause", extraction.termination_for_cause),
            ("Early termination fee", extraction.early_termination_fee),
        ]),
        ("SLA & Exit", [
            ("SLA summary", extraction.sla_summary),
            ("Data exit / transition period", extraction.data_exit_transition_period),
        ]),
    ]

    any_rendered_overall = False
    for section_name, fields in sections:
        has_content = any(f is not None and f.value for _, f in fields)
        if not has_content:
            continue
        any_rendered_overall = True
        with st.expander(section_name, expanded=True):
            for label, field in fields:
                render_field(label, field)

    if extraction.fields_not_found:
        with st.expander("Not found in this document"):
            st.write(", ".join(f.replace("_", " ") for f in extraction.fields_not_found))

    st.divider()

    # --- Download + reset ---
    col1, col2 = st.columns([1, 1])
    with col1:
        try:
            pdf_bytes = generate_pdf_report(extraction, filename)
            st.download_button(
                "Download report (PDF)",
                data=pdf_bytes,
                file_name=f"{vendor_label.replace(' ', '_')}_contract_analysis.pdf",
                mime="application/pdf",
                type="primary",
            )
        except Exception as e:
            st.error(f"Couldn't generate the PDF report ({e}). Your results are still shown above.")

    with col2:
        if st.button("Analyze another contract"):
            reset_session()
            st.rerun()

    st.caption(
        "This analysis is AI-generated and not legal advice. Verify extracted terms "
        "against the original signed contract."
    )
