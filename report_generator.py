"""
Generates a clean, formatted PDF report from a ContractExtraction result.

This is the "download your results" deliverable -- a polished single PDF a
person can save, forward, or drop into a vendor file, rather than raw JSON.
Built with reportlab's platypus layer (not raw canvas drawing) so sections,
tables and page breaks are handled for us.
"""

from __future__ import annotations

import io
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable,
)

from schema import ContractExtraction, ExtractedField, ProductLineItem

# ---------------------------------------------------------------------------
# Palette (matches the Streamlit app's ledger/audit theme)
# ---------------------------------------------------------------------------

INK = colors.HexColor("#1C2B33")
MUTED = colors.HexColor("#5B6B66")
ACCENT = colors.HexColor("#2B6E63")
BORDER = colors.HexColor("#D7DED9")
RISK_HIGH = colors.HexColor("#B23A32")
RISK_MEDIUM = colors.HexColor("#C98A2C")
RISK_LOW = colors.HexColor("#3F7D5C")

RISK_COLOR = {"high": RISK_HIGH, "medium": RISK_MEDIUM, "low": RISK_LOW}


def _styles():
    ss = getSampleStyleSheet()
    ss.add(ParagraphStyle(
        name="ReportTitle", fontName="Helvetica-Bold", fontSize=20,
        textColor=INK, spaceAfter=10, leading=24,
    ))
    ss.add(ParagraphStyle(
        name="ReportSubtitle", fontName="Helvetica", fontSize=9,
        textColor=MUTED, spaceAfter=18, leading=12,
    ))
    ss.add(ParagraphStyle(
        name="SectionHeading", fontName="Helvetica-Bold", fontSize=12,
        textColor=ACCENT, spaceBefore=16, spaceAfter=8,
    ))
    ss.add(ParagraphStyle(
        name="FieldLabel", fontName="Helvetica-Bold", fontSize=8.5,
        textColor=MUTED,
    ))
    ss.add(ParagraphStyle(
        name="FieldValue", fontName="Helvetica", fontSize=10,
        textColor=INK, spaceAfter=6,
    ))
    ss.add(ParagraphStyle(
        name="Evidence", fontName="Helvetica-Oblique", fontSize=8,
        textColor=MUTED, spaceAfter=10,
    ))
    ss.add(ParagraphStyle(
        name="BodyTextTight", fontName="Helvetica", fontSize=10,
        textColor=INK, leading=14,
    ))
    ss.add(ParagraphStyle(
        name="Disclaimer", fontName="Helvetica-Oblique", fontSize=7.5,
        textColor=MUTED,
    ))
    return ss


def _field_block(styles, label: str, field: ExtractedField | None):
    """Renders one labeled field + its evidence quote, or nothing if empty."""
    flow = []
    if field is None or not field.value:
        return flow
    flow.append(Paragraph(label.upper(), styles["FieldLabel"]))
    flow.append(Paragraph(str(field.value), styles["FieldValue"]))
    if field.evidence and field.evidence.quote:
        page_note = f" (p. {field.evidence.page})" if field.evidence.page else ""
        flow.append(Paragraph(f'&ldquo;{field.evidence.quote}&rdquo;{page_note}', styles["Evidence"]))
    return flow


def _products_table(products: list[ProductLineItem]):
    """
    Renders the product/license line-item table. Uses Paragraph cells
    (rather than plain strings) so long product names wrap instead of
    overflowing -- 9 columns on a letter-width page leaves little room per
    column, so wrapping is essential rather than cosmetic.
    """
    header_style = ParagraphStyle(
        name="TableHeader", fontName="Helvetica-Bold", fontSize=6.5,
        textColor=colors.white, leading=8,
    )
    cell_style = ParagraphStyle(
        name="TableCell", fontName="Helvetica", fontSize=6.5,
        textColor=INK, leading=8,
    )

    headers = [
        "Part #", "Product Name", "License Type", "License Metric",
        "Purchased Rights", "Unit Cost", "Start Date", "End Date", "Country",
    ]
    table_data = [[Paragraph(h, header_style) for h in headers]]

    for p in products:
        row_values = [
            p.publisher_part_number, p.product_name, p.license_type,
            p.license_metric, p.purchased_rights, p.unit_cost,
            p.start_date, p.end_date, p.country_of_agreement,
        ]
        table_data.append([Paragraph(v or "—", cell_style) for v in row_values])

    col_widths = [
        0.7 * inch, 1.3 * inch, 0.65 * inch, 0.75 * inch, 0.75 * inch,
        0.6 * inch, 0.55 * inch, 0.55 * inch, 0.55 * inch,
    ]
    table = Table(table_data, colWidths=col_widths, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
        ("GRID", (0, 0), (-1, -1), 0.4, BORDER),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F9F7")]),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
    ]))
    return [table, Spacer(1, 10)]


def generate_pdf_report(extraction: ContractExtraction, source_filename: str) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=letter,
        topMargin=0.75 * inch, bottomMargin=0.75 * inch,
        leftMargin=0.85 * inch, rightMargin=0.85 * inch,
    )
    styles = _styles()
    story = []

    # --- Header ---
    title = extraction.vendor_name.value if extraction.vendor_name and extraction.vendor_name.value else "Vendor"
    story.append(Paragraph(f"Contract Analysis: {title}", styles["ReportTitle"]))
    generated = datetime.now().strftime("%d %b %Y, %H:%M")
    story.append(Paragraph(
        f"Source document: {source_filename} &nbsp;|&nbsp; Generated {generated} &nbsp;|&nbsp; "
        f"SAM Contract Analyzer",
        styles["ReportSubtitle"],
    ))
    story.append(HRFlowable(width="100%", color=BORDER, thickness=1))

    # --- Summary ---
    story.append(Paragraph("Summary", styles["SectionHeading"]))
    story.append(Paragraph(extraction.plain_english_summary, styles["BodyTextTight"]))

    # --- Risk flags ---
    if extraction.risk_flags:
        story.append(Paragraph("Risk Flags", styles["SectionHeading"]))
        for flag in extraction.risk_flags:
            sev = flag.severity.lower()
            color = RISK_COLOR.get(sev, MUTED)
            row = Table(
                [[Paragraph(f'<font color="{color.hexval()}"><b>{sev.upper()}</b></font>', styles["FieldValue"]),
                  Paragraph(f"<b>{flag.clause}</b> &mdash; {flag.explanation}", styles["FieldValue"])]],
                colWidths=[0.9 * inch, 5.35 * inch],
            )
            row.setStyle(TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LINEBELOW", (0, 0), (-1, -1), 0.5, BORDER),
            ]))
            story.append(row)
        story.append(Spacer(1, 8))

    # --- Products / license line items ---
    if extraction.products:
        story.append(Paragraph("Products / License Line Items", styles["SectionHeading"]))
        story.extend(_products_table(extraction.products))

    # --- Sections of extracted fields ---
    sections = [
        ("Identification", [
            ("Vendor", extraction.vendor_name),
            ("Customer", extraction.customer_name),
            ("Contract title", extraction.contract_title),
        ]),
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

    for section_name, fields in sections:
        rendered = []
        for label, field in fields:
            rendered.extend(_field_block(styles, label, field))
        if rendered:  # skip sections where every field was empty
            story.append(Paragraph(section_name, styles["SectionHeading"]))
            story.extend(rendered)

    # --- Fields not found ---
    if extraction.fields_not_found:
        story.append(Paragraph("Not Found in This Document", styles["SectionHeading"]))
        story.append(Paragraph(
            "The following were not mentioned or could not be confidently located in the "
            "source text: " + ", ".join(f.replace("_", " ") for f in extraction.fields_not_found),
            styles["BodyTextTight"],
        ))

    # --- Footer / disclaimer ---
    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", color=BORDER, thickness=0.5))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "This report was generated by an AI extraction tool and is provided for informational "
        "purposes only. It is not legal advice. Always verify extracted terms against the "
        "original signed contract before making decisions.",
        styles["Disclaimer"],
    ))

    doc.build(story)
    return buffer.getvalue()
