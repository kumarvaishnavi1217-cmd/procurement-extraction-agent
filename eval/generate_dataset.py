"""Generate the labelled evaluation set: synthetic supplier quotes and invoices with ground truth.

    python eval/generate_dataset.py

Writes eval/dataset/<id>.pdf + <id>.json (labels) and eval/fixtures/<id>.json. The fixtures are
*simulated* model outputs with typical LLM mistakes injected (buyer taken as supplier, PO number taken as
invoice number, misread quantity, ambiguous date, skipped line). They let tests and CI exercise the
validation rules and the scoring offline; real accuracy numbers come from `procex-eval --extractor claude`.
"""

import json
import random
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

ROOT = Path(__file__).parent
BUYER = "Meridian Manufacturing GmbH"

SUPPLIERS = [
    ("Nordwerk Industrietechnik GmbH", "Hafenstrasse 12, 20457 Hamburg"),
    ("Apex Fasteners Ltd", "Unit 4, Riverside Park, Leeds LS9 0RQ"),
    ("Sierra Components Inc.", "2201 Mission Blvd, Fremont, CA 94539"),
    ("Baltic Hydraulics AB", "Industrivägen 8, 211 24 Malmö"),
    ("Rhine Precision Parts GmbH", "Rheinallee 40, 55118 Mainz"),
    ("Keystone Electrical Supply Co.", "88 Commerce Dr, Pittsburgh, PA 15222"),
]
PARTS = [
    ("HX-M8-25-ZN", "Hex bolt M8x25, zinc plated"),
    ("BRG-6204-2RS", "Deep groove ball bearing 6204-2RS"),
    ("VLV-BALL-DN25", "Ball valve DN25 stainless"),
    ("SNS-PT100-A", "PT100 temperature sensor class A"),
    ("CBL-NYY-3G2.5", "Power cable NYY-J 3G2.5 (per m)"),
    ("FLT-HYD-10U", "Hydraulic filter element 10 micron"),
    ("GSK-NBR-50", "NBR gasket 50 mm"),
    ("CON-M12-4P", "M12 connector 4-pin female"),
    ("MTR-0.75KW-4P", "Induction motor 0.75 kW 4-pole"),
    ("RLY-24VDC-2CO", "Relay 24 V DC 2 CO"),
]


@dataclass
class Spec:
    doc_id: str
    kind: str  # invoice | quote
    supplier: int
    number: str
    po_number: str
    issued: date
    date_format: str  # iso | eu | long | us
    currency: str
    number_format: str  # us | eu
    tax_rate: float
    lines: list[tuple[str, str, int, float]]


def fmt_number(value: float, style: str) -> str:
    s = f"{value:,.2f}"
    return s if style == "us" else s.replace(",", "_").replace(".", ",").replace("_", ".")


def fmt_date(d: date, style: str) -> str:
    return {
        "iso": d.isoformat(),
        "eu": d.strftime("%d.%m.%Y"),
        "long": f"{d.day} {d.strftime('%B %Y')}",
        "us": d.strftime("%m/%d/%Y"),
    }[style]


def totals(spec: Spec) -> tuple[list[float], float, float]:
    line_totals = [round(q * p, 2) for _, _, q, p in spec.lines]
    subtotal = round(sum(line_totals), 2)
    return line_totals, subtotal, round(subtotal * (1 + spec.tax_rate), 2)


def make_specs() -> list[Spec]:
    rng = random.Random(42)  # noqa: S311 - deterministic test data, not security
    combos = [
        ("invoice", "EUR", "eu", "eu"),
        ("quote", "USD", "us", "iso"),
        ("invoice", "GBP", "us", "long"),
        ("invoice", "EUR", "eu", "iso"),
        ("quote", "SEK", "eu", "eu"),
        ("invoice", "USD", "us", "us"),
        ("quote", "EUR", "eu", "long"),
        ("invoice", "EUR", "us", "eu"),
        ("quote", "GBP", "us", "iso"),
        ("invoice", "USD", "us", "long"),
        ("invoice", "CHF", "us", "eu"),
        ("quote", "EUR", "eu", "iso"),
    ]
    specs = []
    for i, (kind, currency, num_fmt, date_fmt) in enumerate(combos, start=1):
        parts = rng.sample(PARTS, rng.randint(2, 5))
        lines = [
            (pn, desc, rng.choice([1, 2, 5, 10, 25, 50, 100, 250]), round(rng.uniform(0.3, 900), 2))
            for pn, desc in parts
        ]
        prefix = "INV" if kind == "invoice" else "Q"
        specs.append(Spec(
            doc_id=f"doc{i:02d}",
            kind=kind,
            supplier=(i - 1) % len(SUPPLIERS),
            number=f"{prefix}-2026-{rng.randint(1000, 9999)}",
            po_number=f"PO-{rng.randint(10000, 99999)}",
            issued=date(2026, rng.randint(1, 9), rng.randint(1, 12)),  # day <= 12 keeps us/eu dates ambiguous
            date_format=date_fmt,
            currency=currency,
            number_format=num_fmt,
            tax_rate={"EUR": 0.19, "GBP": 0.20, "SEK": 0.25, "CHF": 0.081}.get(currency, 0.0),
            lines=lines,
        ))  # fmt: skip
    return specs


# ---------- rendering ----------


def render(spec: Spec, path: Path) -> dict[str, str]:
    """Draws the PDF and returns the exact printed strings, used as evidence in the fixtures."""
    name, address = SUPPLIERS[spec.supplier]
    n = lambda v: fmt_number(v, spec.number_format)  # noqa: E731
    line_totals, subtotal, total = totals(spec)
    printed: dict[str, str] = {}
    c = canvas.Canvas(str(path), pagesize=A4)
    w, h = A4
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, h - 50, name)
    printed["supplier_name"] = name
    c.setFont("Helvetica", 9)
    c.drawString(40, h - 64, address)

    title = "INVOICE" if spec.kind == "invoice" else "QUOTATION"
    number_label = "Invoice No:" if spec.kind == "invoice" else "Quote #:"
    c.setFont("Helvetica-Bold", 16)
    c.drawRightString(w - 40, h - 50, title)
    c.setFont("Helvetica", 10)
    for i, text in enumerate([
        f"{number_label} {spec.number}",
        f"Date: {fmt_date(spec.issued, spec.date_format)}",
        f"Your PO: {spec.po_number}",
    ]):  # fmt: skip
        c.drawRightString(w - 40, h - 70 - i * 14, text)
    printed["document_number"] = f"{number_label} {spec.number}"
    printed["document_date"] = f"Date: {fmt_date(spec.issued, spec.date_format)}"
    printed["po_number"] = f"Your PO: {spec.po_number}"

    c.drawString(40, h - 110, f"Bill To: {BUYER}")
    printed["buyer"] = f"Bill To: {BUYER}"
    c.drawString(40, h - 124, "Werkstrasse 3, 70173 Stuttgart")

    y = h - 170
    if spec.kind == "invoice":
        cols = [40, 70, 170, 380, 440, 520]
        c.setFont("Helvetica-Bold", 9)
        for x, head in zip(cols, ["Pos", "Part No.", "Description", "Qty", "Unit Price", "Amount"], strict=True):
            c.drawString(x, y, head)
        c.setFont("Helvetica", 9)
        for i, ((pn, desc, qty, price), lt) in enumerate(zip(spec.lines, line_totals, strict=True)):
            y -= 16
            for x, text in zip(cols, [str(i + 1), pn, desc, str(qty), n(price), n(lt)], strict=True):
                c.drawString(x, y, text)
            printed.update({f"{i}.part": pn, f"{i}.desc": desc, f"{i}.qty": str(qty),
                            f"{i}.price": n(price), f"{i}.total": n(lt)})  # fmt: skip
    else:
        c.setFont("Helvetica", 10)
        c.drawString(40, y, "We are pleased to quote as follows:")
        for i, ((pn, desc, qty, price), lt) in enumerate(zip(spec.lines, line_totals, strict=True)):
            y -= 18
            c.drawString(50, y, f"{i + 1}) Article {pn} - {desc}")
            c.drawRightString(w - 40, y, f"{qty} pcs x {n(price)} = {n(lt)}")
            printed.update({f"{i}.part": f"Article {pn}", f"{i}.desc": desc, f"{i}.qty": f"{qty} pcs",
                            f"{i}.price": f"x {n(price)}", f"{i}.total": f"= {n(lt)}"})  # fmt: skip

    y -= 40
    tax_label = f"VAT {spec.tax_rate * 100:g}%" if spec.tax_rate else "Sales tax (exempt)"
    for label, value, key in [("Subtotal", subtotal, "subtotal"), (tax_label, total - subtotal, None),
                              ("Total", total, "total")]:  # fmt: skip
        text = f"{label}: {n(value)}"
        c.drawRightString(w - 40, y, text)
        if key:
            printed[key] = text
        y -= 14
    c.drawString(40, y - 20, f"All amounts in {spec.currency}. Payment terms: 30 days net.")
    printed["currency"] = f"All amounts in {spec.currency}"
    c.showPage()
    c.save()
    return printed


# ---------- labels and simulated model output ----------


def label(spec: Spec) -> dict[str, object]:
    line_totals, subtotal, total = totals(spec)
    return {
        "document_type": spec.kind,
        "supplier_name": SUPPLIERS[spec.supplier][0],
        "document_number": spec.number,
        "document_date": spec.issued.isoformat(),
        "currency": spec.currency,
        "subtotal": subtotal,
        "total": total,
        "line_items": [
            {"part_number": pn, "quantity": q, "unit_price": p, "line_total": lt}
            for (pn, _, q, p), lt in zip(spec.lines, line_totals, strict=True)
        ],
    }


def f(value: object, evidence: str | None, confidence: float = 0.96) -> dict[str, object]:
    return {"value": value, "confidence": confidence, "evidence": evidence}


def fixture(spec: Spec, printed: dict[str, str]) -> dict[str, object]:
    lab = label(spec)
    lines = [
        {
            "part_number": f(li["part_number"], printed[f"{i}.part"]),
            "description": f(spec.lines[i][1], printed[f"{i}.desc"], 0.9),
            "quantity": f(li["quantity"], printed[f"{i}.qty"]),
            "unit_price": f(li["unit_price"], printed[f"{i}.price"]),
            "line_total": f(li["line_total"], printed[f"{i}.total"]),
        }
        for i, li in enumerate(lab["line_items"])  # type: ignore[arg-type]
    ]
    out: dict[str, object] = {
        "document_type": spec.kind,
        "supplier_name": f(lab["supplier_name"], printed["supplier_name"], 0.97),
        "document_number": f(spec.number, printed["document_number"]),
        "document_date": f(lab["document_date"], printed["document_date"], 0.93),
        "currency": f(spec.currency, printed["currency"], 0.97),
        "line_items": lines,
        "subtotal": f(lab["subtotal"], printed["subtotal"]),
        "total": f(lab["total"], printed["total"]),
    }

    # Injected mistakes, each typical of LLM extraction on real documents
    if spec.doc_id == "doc02":  # misread quantity: 10x off -> arithmetic + evidence checks fire
        q = lines[0]["quantity"]
        q["value"] = q["value"] / 10 if q["value"] >= 10 else q["value"] * 10  # type: ignore[index, operator]
        q["confidence"] = 0.9  # type: ignore[index]
    if spec.doc_id == "doc04":  # took the buyer as supplier, grounded and confident -> NOT caught
        out["supplier_name"] = f(BUYER, printed["buyer"], 0.9)
    if spec.doc_id == "doc06":  # ambiguous US date read as day-first, model unsure -> caught by confidence
        d = spec.issued
        out["document_date"] = f(date(d.year, d.day, d.month).isoformat(), printed["document_date"], 0.65)
    if spec.doc_id == "doc08":  # took the PO number as invoice number, grounded -> NOT caught
        out["document_number"] = f(spec.po_number, printed["po_number"], 0.92)
    if spec.doc_id == "doc10":  # hallucinated unit price that is not printed -> evidence check fires
        p = lines[1]["unit_price"]
        p["value"] = round(p["value"] + 1.0, 2)  # type: ignore[index, operator]
        p["evidence"] = fmt_number(p["value"], spec.number_format)  # type: ignore[index, arg-type]
    if spec.doc_id == "doc11":  # skipped the last line item -> subtotal check fires
        lines.pop()
    if spec.doc_id == "doc12":  # correct but unsure -> reviewed unnecessarily
        lines[0]["part_number"]["confidence"] = 0.6  # type: ignore[index]
    return out


def main() -> None:
    (ROOT / "dataset").mkdir(exist_ok=True)
    (ROOT / "fixtures").mkdir(exist_ok=True)
    for spec in make_specs():
        printed = render(spec, ROOT / "dataset" / f"{spec.doc_id}.pdf")
        (ROOT / "dataset" / f"{spec.doc_id}.json").write_text(json.dumps(label(spec), indent=2) + "\n")
        (ROOT / "fixtures" / f"{spec.doc_id}.json").write_text(json.dumps(fixture(spec, printed), indent=2) + "\n")
    print(f"wrote {len(make_specs())} documents to {ROOT / 'dataset'}")


if __name__ == "__main__":
    main()
