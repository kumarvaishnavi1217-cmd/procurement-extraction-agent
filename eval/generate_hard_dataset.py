"""Generate the hard labelled set: documents built around the layouts that break extraction in practice.

    python eval/generate_hard_dataset.py

Writes eval/dataset-hard/<id>.pdf + <id>.json (labels). Unlike eval/dataset there are no simulated model
outputs: this set is only meaningful against a live model (`procex-eval --dataset eval/dataset-hard ...`).

Each case targets one failure pattern, in two variants:

    german      DIN 5008 German invoice: supplier only in the small sender line and footer, seven
                reference numbers and three dates next to the invoice number and date
    billto      buyer printed large at the top, supplier only in a "remit to" footer
    multipage   16 line items over two pages, with "carried forward" / "brought forward" sums
    bulk        thousand-separated quantities (10.000 pcs) and four-decimal unit prices
    scanned     image-only PDF, slightly rotated, with noise: no text layer to check evidence against
    quote       ambiguous US dates, an RFQ number and date, validity and lead-time lines
    deposit     a deposit deducted after the total, so the largest bold number is the balance due
    lookalike   near-identical part numbers, numbers inside descriptions, SAP-style position numbers
"""

import json
import random
import time
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import reportlab
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

ROOT = Path(__file__).parent
OUT = ROOT / "dataset-hard"
BUYER = "Meridian Manufacturing GmbH"
FONTS = Path(reportlab.__file__).parent / "fonts"
SCALE = 150 / 72  # scan resolution: 150 dpi

SUPPLIERS = {
    "nordwerk": ("Nordwerk Industrietechnik GmbH", "Hafenstrasse 12, 20457 Hamburg", "de"),
    "rhine": ("Rhine Precision Parts GmbH", "Rheinallee 40, 55118 Mainz", "de"),
    "apex": ("Apex Fasteners Ltd", "Unit 4, Riverside Park, Leeds LS9 0RQ", "uk"),
    "sierra": ("Sierra Components Inc.", "2201 Mission Blvd, Fremont, CA 94539", "us"),
    "keystone": ("Keystone Electrical Supply Co.", "88 Commerce Dr, Pittsburgh, PA 15222", "us"),
    "baltic": ("Baltic Hydraulics AB", "Industrivägen 8, 211 24 Malmö", "se"),
}
CATALOG = [
    ("HX-M8-25-ZN", "Hex bolt M8x25, zinc plated"),
    ("HX-M8-30-ZN", "Hex bolt M8x30, zinc plated"),
    ("HX-M10-40-A2", "Hex bolt M10x40, stainless A2"),
    ("NUT-M8-ZN", "Hex nut M8, zinc plated"),
    ("WSH-8.4-ZN", "Washer 8.4 mm, zinc plated"),
    ("BRG-6204-2RS", "Deep groove ball bearing 6204-2RS"),
    ("BRG-6205-2RS", "Deep groove ball bearing 6205-2RS"),
    ("BRG-6204-ZZ", "Deep groove ball bearing 6204-ZZ"),
    ("VLV-BALL-DN25", "Ball valve DN25 stainless"),
    ("VLV-BALL-DN32", "Ball valve DN32 stainless"),
    ("SNS-PT100-A", "PT100 temperature sensor class A"),
    ("SNS-PT1000-B", "PT1000 temperature sensor class B"),
    ("CBL-NYY-3G2.5", "Power cable NYY-J 3G2.5 (per m)"),
    ("CBL-NYY-5G1.5", "Power cable NYY-J 5G1.5 (per m)"),
    ("FLT-HYD-10U", "Hydraulic filter element 10 micron"),
    ("FLT-HYD-25U", "Hydraulic filter element 25 micron"),
    ("GSK-NBR-50", "NBR gasket 50 mm"),
    ("GSK-EPDM-50", "EPDM gasket 50 mm"),
    ("CON-M12-4P", "M12 connector 4-pin female"),
    ("CON-M12-8P", "M12 connector 8-pin female"),
    ("MTR-0.75KW-4P", "Induction motor 0.75 kW 4-pole"),
    ("MTR-1.1KW-4P", "Induction motor 1.1 kW 4-pole"),
    ("RLY-24VDC-2CO", "Relay 24 V DC 2 CO"),
    ("RLY-230VAC-2CO", "Relay 230 V AC 2 CO"),
]

Line = tuple[str, str, int, float]  # part number, description, quantity, unit price


# ---------- formatting ----------


def num(value: float, style: str, decimals: int = 2) -> str:
    s = f"{value:,.{decimals}f}"
    return s if style == "us" else s.replace(",", "_").replace(".", ",").replace("_", ".")


def qty(value: int, style: str) -> str:
    return f"{value:,}" if style == "us" else f"{value:,}".replace(",", ".")


def us_date(d: date) -> str:
    return d.strftime("%m/%d/%Y")


def eu_date(d: date) -> str:
    return d.strftime("%d.%m.%Y")


def long_date(d: date) -> str:
    return f"{d.day} {d.strftime('%B %Y')}"


def line_total(line: Line) -> float:
    return round(line[2] * line[3], 2)


def label(
    kind: str, supplier: str, number: str, issued: date, currency: str, lines: list[Line], total: float
) -> dict[str, Any]:
    subtotal = round(sum(line_total(li) for li in lines), 2)
    return {
        "document_type": kind,
        "supplier_name": SUPPLIERS[supplier][0],
        "document_number": number,
        "document_date": issued.isoformat(),
        "currency": currency,
        "subtotal": subtotal,
        "total": total,
        "line_items": [
            {"part_number": pn, "quantity": q, "unit_price": p, "line_total": line_total((pn, d, q, p))}
            for pn, d, q, p in lines
        ],
    }


def pick_lines(rng: random.Random, n: int, qtys: list[int], lo: float, hi: float, decimals: int = 2) -> list[Line]:
    return [(pn, desc, rng.choice(qtys), round(rng.uniform(lo, hi), decimals)) for pn, desc in rng.sample(CATALOG, n)]


# ---------- drawing: a real PDF, or an image-only "scan" ----------


class Doc:
    """A4 pages with y measured from the top."""

    W, H = A4

    def __init__(self, path: Path, rng: random.Random, scanned: bool = False) -> None:
        self.path, self.rng, self.scanned = path, rng, scanned
        self.images: list[Image.Image] = []
        if scanned:
            self._new_image()
        else:
            self.c = canvas.Canvas(str(path), pagesize=A4, invariant=True)  # no timestamps: reproducible bytes

    def _new_image(self) -> None:
        img = Image.new("L", (int(self.W * SCALE), int(self.H * SCALE)), 248)
        self.images.append(img)
        self.draw = ImageDraw.Draw(img)

    def text(self, x: float, y: float, s: str, size: float = 9, bold: bool = False, right: bool = False) -> None:
        if self.scanned:
            font = ImageFont.truetype(str(FONTS / ("VeraBd.ttf" if bold else "Vera.ttf")), round(size * SCALE))
            px = x * SCALE - (self.draw.textlength(s, font=font) if right else 0)
            self.draw.text((px, (y - size * 0.8) * SCALE), s, font=font, fill=self.rng.randint(15, 55))
            return
        self.c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
        (self.c.drawRightString if right else self.c.drawString)(x, self.H - y, s)

    def rule(self, y: float, x1: float = 40, x2: float = 555) -> None:
        if self.scanned:
            self.draw.line([(x1 * SCALE, y * SCALE), (x2 * SCALE, y * SCALE)], fill=90, width=2)
        else:
            self.c.setLineWidth(0.5)
            self.c.line(x1, self.H - y, x2, self.H - y)

    def new_page(self) -> None:
        if self.scanned:
            self._new_image()
        else:
            self.c.showPage()

    def save(self) -> None:
        if not self.scanned:
            self.c.showPage()
            self.c.save()
            return
        pages = []
        for img in self.images:
            for _ in range(2500):  # dust and toner specks
                x, y = self.rng.randrange(img.width), self.rng.randrange(img.height)
                img.putpixel((x, y), self.rng.randint(120, 200))
            pages.append(img.rotate(self.rng.uniform(-0.8, 0.8), fillcolor=248, resample=Image.Resampling.BICUBIC))
        fixed = time.gmtime(datetime(2026, 1, 1, tzinfo=UTC).timestamp())  # reproducible bytes
        pages[0].save(
            self.path, "PDF", resolution=150, save_all=True, append_images=pages[1:], creationDate=fixed, modDate=fixed
        )


def table(doc: Doc, y: float, cols: list[tuple[float, str, bool]], rows: list[list[str]], size: float = 8.5) -> float:
    """cols: (x, header, right-aligned). Returns the y below the last row."""
    for x, head, right in cols:
        doc.text(x, y, head, size, bold=True, right=right)
    doc.rule(y + 4)
    for row in rows:
        y += 15
        for (x, _, right), cell in zip(cols, row, strict=True):
            doc.text(x, y, cell, size, right=right)
    doc.rule(y + 6)
    return y + 6


def totals(doc: Doc, y: float, rows: list[tuple[str, str, bool]], x: float = 555) -> float:
    for text, value, bold in rows:
        y += 15
        doc.text(x - 110, y, text, 9, bold=bold, right=True)
        doc.text(x, y, value, 9, bold=bold, right=True)
    return y


# ---------- cases ----------


def german(doc: Doc, rng: random.Random, sup: str) -> dict[str, Any]:
    name, address, _ = SUPPLIERS[sup]
    issued = date(2026, rng.randint(1, 9), rng.randint(13, 28))
    number = f"RE-26-{rng.randint(10000, 99999)}"
    lines = pick_lines(rng, rng.randint(3, 6), [1, 2, 4, 5, 10, 20, 50], 2, 600)
    sub = round(sum(line_total(li) for li in lines), 2)
    total = round(sub * 1.19, 2)

    # no printed letterhead: in real life it is a logo image, so the name appears only in small print
    doc.text(40, 118, f"{name} · {address}", 6.5)
    doc.rule(121, 40, 300)
    for i, (text, bold) in enumerate([("Firma", False), (BUYER, True), ("Zentraleinkauf", False),
                                      ("Werkstrasse 3", False), ("70173 Stuttgart", False)]):  # fmt: skip
        doc.text(40, 136 + i * 13, text, 10.5 if bold else 10, bold=bold)
    info = [
        ("Kundennummer:", f"KD-{rng.randint(10000, 99999)}"),
        ("Rechnungsnummer:", number),
        ("Rechnungsdatum:", eu_date(issued)),
        ("Lieferschein-Nr.:", f"LS-26-{rng.randint(10000, 99999)}"),
        ("Lieferdatum:", eu_date(issued - timedelta(days=rng.randint(2, 6)))),
        ("Ihre Bestellung:", f"4500{rng.randint(100000, 999999)}"),
        ("Bestelldatum:", eu_date(issued - timedelta(days=rng.randint(10, 30)))),
        ("Ihre USt-IdNr.:", "DE811234567"),
    ]
    for i, (k, v) in enumerate(info):
        doc.text(360, 130 + i * 12, k, 8.5)
        doc.text(555, 130 + i * 12, v, 8.5, right=True)
    doc.text(40, 250, "Rechnung", 15, bold=True)
    doc.text(40, 272, "Sehr geehrte Damen und Herren,", 9.5)
    doc.text(40, 285, "für unsere Lieferung erlauben wir uns, Ihnen wie folgt zu berechnen:", 9.5)
    cols = [(40, "Pos.", False), (70, "Artikel-Nr.", False), (160, "Bezeichnung", False), (405, "Menge", True),
            (480, "Einzelpreis €", True), (555, "Gesamtpreis €", True)]  # fmt: skip
    rows = [[str(i + 1), pn, d, f"{q} Stk", num(p, "eu"), num(line_total((pn, d, q, p)), "eu")]
            for i, (pn, d, q, p) in enumerate(lines)]  # fmt: skip
    y = table(doc, 312, cols, rows)
    y = totals(doc, y, [("Nettobetrag", f"{num(sub, 'eu')} €", False),
                        ("zzgl. 19 % MwSt.", f"{num(total - sub, 'eu')} €", False),
                        ("Rechnungsbetrag", f"{num(total, 'eu')} €", True)])  # fmt: skip
    doc.text(40, y + 30, "Zahlbar innerhalb von 14 Tagen ohne Abzug. Es gelten unsere AGB.", 9)
    doc.rule(790)
    doc.text(40, 802, f"{name} · {address} · Geschäftsführer: Jan Petersen · Amtsgericht HRB 112233", 6.5)
    doc.text(40, 811, "USt-IdNr. DE298765432 · Commerzbank · IBAN DE12 2004 0000 0123 4567 00 · BIC COBADEFFXXX", 6.5)
    return label("invoice", sup, number, issued, "EUR", lines, total)


def billto(doc: Doc, rng: random.Random, sup: str) -> dict[str, Any]:
    name, address, region = SUPPLIERS[sup]
    us = region == "us"
    style, currency = ("us", "USD") if us else ("us", "GBP")
    issued = date(2026, rng.randint(1, 9), rng.randint(13, 28))
    number = f"{rng.randint(200000, 299999)}"
    lines = pick_lines(rng, rng.randint(3, 5), [2, 5, 10, 25, 100], 1, 400)
    sub = round(sum(line_total(li) for li in lines), 2)
    tax_label, rate = ("Sales Tax (7%)", 0.07) if us else ("VAT 20%", 0.20)
    total = round(sub * (1 + rate), 2)

    doc.text(40, 60, "BILL TO", 9, bold=True)
    doc.text(40, 82, BUYER, 18, bold=True)
    doc.text(40, 98, "Accounts Payable · Werkstrasse 3 · 70173 Stuttgart · Germany", 9)
    doc.text(40, 122, "SHIP TO", 9, bold=True)
    doc.text(40, 136, f"{BUYER} - Plant 2, Goods Receipt Gate 4, 71229 Leonberg", 9)
    doc.text(555, 60, "INVOICE", 20, bold=True, right=True)
    for i, (k, v) in enumerate([("Invoice #", number), ("Invoice Date", us_date(issued) if us else long_date(issued)),
                                ("Customer ID", "MER-0042"), ("Customer PO", f"PO-{rng.randint(10000, 99999)}"),
                                ("Terms", "Net 30")]):  # fmt: skip
        doc.text(450, 84 + i * 13, k, 9, bold=True, right=True)
        doc.text(555, 84 + i * 13, v, 9, right=True)
    cols = [(40, "Item", False), (150, "Description", False), (390, "Qty", True), (470, "Unit Price", True),
            (555, "Amount", True)]  # fmt: skip
    rows = [[pn, d, str(q), num(p, style), num(line_total((pn, d, q, p)), style)] for pn, d, q, p in lines]
    y = table(doc, 180, cols, rows)
    y = totals(doc, y, [("Subtotal", num(sub, style), False), (tax_label, num(total - sub, style), False),
                        (f"Total {currency}", num(total, style), True)])  # fmt: skip
    doc.text(40, y + 40, "Thank you for your business!", 10, bold=True)
    doc.rule(780)
    doc.text(40, 793, f"Please remit to: {name} · {address}", 8)
    doc.text(40, 804, "Questions? accounts@supplier-portal.example · Tax ID 41-2873345", 8)
    return label("invoice", sup, number, issued, currency, lines, total)


def multipage(doc: Doc, rng: random.Random, sup: str) -> dict[str, Any]:
    name, address, region = SUPPLIERS[sup]
    style, currency, rate = ("eu", "SEK", 0.25) if region == "se" else ("us", "USD", 0.0)
    issued = date(2026, rng.randint(1, 9), rng.randint(13, 28))
    number = f"INV-{rng.randint(100000, 999999)}"
    lines = pick_lines(rng, 16, [1, 2, 4, 8, 12, 20, 40], 5, 900)
    sub = round(sum(line_total(li) for li in lines), 2)
    total = round(sub * (1 + rate), 2)
    fdate = eu_date(issued) if style == "eu" else us_date(issued)
    cols = [(40, "Pos", False), (70, "Part No.", False), (170, "Description", False), (400, "Qty", True),
            (470, "Unit Price", True), (555, "Amount", True)]  # fmt: skip

    def rows(chunk: list[Line], start: int) -> list[list[str]]:
        return [[str(start + i), pn, d, str(q), num(p, style), num(line_total((pn, d, q, p)), style)]
                for i, (pn, d, q, p) in enumerate(chunk)]  # fmt: skip

    doc.text(40, 50, name, 14, bold=True)
    doc.text(40, 64, address, 9)
    doc.text(555, 50, "INVOICE", 16, bold=True, right=True)
    doc.text(555, 70, f"Invoice No: {number}", 10, right=True)
    doc.text(555, 84, f"Date: {fdate}", 10, right=True)
    doc.text(555, 98, "Page 1 of 2", 9, right=True)
    doc.text(40, 110, f"Bill To: {BUYER}, Werkstrasse 3, 70173 Stuttgart", 10)
    first = lines[:10]
    carried = round(sum(line_total(li) for li in first), 2)
    y = table(doc, 150, cols, rows(first, 1))
    doc.text(555, y + 18, f"Carried forward: {num(carried, style)}", 9, bold=True, right=True)
    doc.text(40, 812, f"{name} · continued on page 2", 7)

    doc.new_page()
    doc.text(40, 50, f"{name} - Invoice {number} (continued)", 10, bold=True)
    doc.text(555, 50, "Page 2 of 2", 9, right=True)
    doc.text(555, 80, f"Brought forward: {num(carried, style)}", 9, bold=True, right=True)
    y = table(doc, 110, cols, rows(lines[10:], 11))
    tax = (f"VAT {rate * 100:g}%", num(total - sub, style), False) if rate else ("Sales tax", "exempt", False)
    y = totals(doc, y, [("Subtotal", num(sub, style), False), tax, ("Total", num(total, style), True)])
    doc.text(40, y + 30, f"All amounts in {currency}. Payment terms: 30 days net.", 9)
    return label("invoice", sup, number, issued, currency, lines, total)


def bulk(doc: Doc, rng: random.Random, sup: str) -> dict[str, Any]:
    name, address, region = SUPPLIERS[sup]
    currency = "SEK" if region == "se" else "EUR"
    issued = date(2026, rng.randint(1, 9), rng.randint(13, 28))
    number = f"F{rng.randint(2600000, 2699999)}"
    lines = pick_lines(rng, rng.randint(4, 6), [1000, 2500, 5000, 10000, 25000, 1500], 0.004, 0.9, decimals=4)
    sub = round(sum(line_total(li) for li in lines), 2)
    rate = 0.25 if currency == "SEK" else 0.19
    total = round(sub * (1 + rate), 2)

    doc.text(40, 50, name, 14, bold=True)
    doc.text(40, 64, address, 9)
    doc.text(555, 50, "INVOICE", 16, bold=True, right=True)
    doc.text(555, 70, f"Invoice no. {number}", 10, right=True)
    doc.text(555, 84, f"Invoice date {eu_date(issued)}", 10, right=True)
    doc.text(40, 104, f"Customer: {BUYER}, Werkstrasse 3, 70173 Stuttgart", 10)
    cols = [(40, "Article", False), (140, "Description", False), (390, "Quantity", True),
            (470, f"Unit price {currency}", True), (555, f"Net {currency}", True)]  # fmt: skip
    rows = [[pn, d, f"{qty(q, 'eu')} pcs", num(p, "eu", 4), num(line_total((pn, d, q, p)), "eu")]
            for pn, d, q, p in lines]  # fmt: skip
    y = table(doc, 140, cols, rows)
    y = totals(doc, y, [("Net amount", num(sub, "eu"), False), (f"VAT {rate * 100:g}%", num(total - sub, "eu"), False),
                        (f"Total {currency}", num(total, "eu"), True)])  # fmt: skip
    doc.text(40, y + 30, "Prices per piece. Packaging units of 500 pcs. Delivery DAP Stuttgart.", 9)
    return label("invoice", sup, number, issued, currency, lines, total)


def scanned(doc: Doc, rng: random.Random, sup: str) -> dict[str, Any]:
    name, address, region = SUPPLIERS[sup]
    style, currency, rate = ("us", "USD", 0.0) if region == "us" else ("eu", "EUR", 0.19)
    issued = date(2026, rng.randint(1, 9), rng.randint(13, 28))
    number = f"INV-2026-{rng.randint(1000, 9999)}"
    lines = pick_lines(rng, rng.randint(3, 5), [1, 2, 5, 10, 25], 3, 700)
    sub = round(sum(line_total(li) for li in lines), 2)
    total = round(sub * (1 + rate), 2)

    doc.text(40, 50, name, 14, bold=True)
    doc.text(40, 64, address, 9)
    doc.text(555, 50, "INVOICE", 16, bold=True, right=True)
    doc.text(555, 70, f"Invoice No: {number}", 10, right=True)
    doc.text(555, 84, f"Date: {us_date(issued) if style == 'us' else eu_date(issued)}", 10, right=True)
    doc.text(555, 98, f"Your PO: PO-{rng.randint(10000, 99999)}", 10, right=True)
    doc.text(40, 120, f"Bill To: {BUYER}", 10)
    cols = [(40, "Part No.", False), (140, "Description", False), (390, "Qty", True), (470, "Unit Price", True),
            (555, "Amount", True)]  # fmt: skip
    rows = [[pn, d, str(q), num(p, style), num(line_total((pn, d, q, p)), style)] for pn, d, q, p in lines]
    y = table(doc, 160, cols, rows)
    tax = (f"VAT {rate * 100:g}%", num(total - sub, style), False) if rate else ("Sales tax", "exempt", False)
    y = totals(doc, y, [("Subtotal", num(sub, style), False), tax, ("Total", num(total, style), True)])
    doc.text(40, y + 30, f"All amounts in {currency}. Payment terms: 30 days net.", 9)
    return label("invoice", sup, number, issued, currency, lines, total)


def quote(doc: Doc, rng: random.Random, sup: str) -> dict[str, Any]:
    name, address, _ = SUPPLIERS[sup]
    issued = date(2026, rng.randint(2, 9), rng.randint(1, 12))  # day <= 12: month/day order is ambiguous
    number = f"QT-{rng.randint(40000, 49999)}"
    lines = pick_lines(rng, rng.randint(3, 5), [10, 25, 50, 100, 200], 0.5, 300)
    sub = round(sum(line_total(li) for li in lines), 2)

    doc.text(40, 50, name, 14, bold=True)
    doc.text(40, 64, address, 9)
    doc.text(40, 76, "Phone (412) 555-0187 · sales@keystone-supply.example", 8)
    doc.text(555, 50, "QUOTATION", 16, bold=True, right=True)
    rfq_date = issued - timedelta(days=rng.randint(3, 9))
    for i, (k, v) in enumerate([("Quote No.:", number), ("Quote Date:", us_date(issued)),
                                ("Valid Until:", us_date(issued + timedelta(days=30))),
                                ("Your RFQ:", f"RFQ-2026-{rng.randint(100, 999)} of {us_date(rfq_date)}"),
                                ("Sales Rep:", "D. Morales")]):  # fmt: skip
        doc.text(430, 72 + i * 13, k, 9, bold=True, right=True)
        doc.text(555, 72 + i * 13, v, 9, right=True)
    doc.text(40, 110, f"Prepared for: {BUYER}, Werkstrasse 3, 70173 Stuttgart, Germany", 9.5)
    cols = [(40, "Line", False), (70, "Part Number", False), (170, "Description", False), (400, "Qty", True),
            (470, "Unit Price", True), (555, "Ext. Price", True)]  # fmt: skip
    rows = [[str(i + 1), pn, d, str(q), num(p, "us"), num(line_total((pn, d, q, p)), "us")]
            for i, (pn, d, q, p) in enumerate(lines)]  # fmt: skip
    y = table(doc, 150, cols, rows)
    y = totals(doc, y, [("Merchandise", num(sub, "us"), False), ("Sales tax", "not included", False),
                        ("Quote Total USD", num(sub, "us"), True)])  # fmt: skip
    for i, text in enumerate(["Lead time: 4-6 weeks ARO. Freight: FOB Origin, prepay & add.",
                              "Prices firm for 30 days from quote date. Minimum order value USD 250.00."]):  # fmt: skip
        doc.text(40, y + 30 + i * 13, text, 9)
    return label("quote", sup, number, issued, "USD", lines, sub)


def deposit(doc: Doc, rng: random.Random, sup: str) -> dict[str, Any]:
    name, address, region = SUPPLIERS[sup]
    style, currency = ("us", "GBP") if region == "uk" else ("eu", "EUR")
    rate = 0.20 if currency == "GBP" else 0.19
    share = 0.30 if currency == "GBP" else 0.10  # 30%: balance below subtotal; 10%: balance above it
    issued = date(2026, rng.randint(1, 9), rng.randint(13, 28))
    number = f"SI-{rng.randint(30000, 39999)}"
    lines = pick_lines(rng, rng.randint(2, 4), [1, 2, 3], 800, 4000)
    sub = round(sum(line_total(li) for li in lines), 2)
    total = round(sub * (1 + rate), 2)
    paid = round(total * share, 2)

    doc.text(40, 50, name, 14, bold=True)
    doc.text(40, 64, address, 9)
    doc.text(555, 50, "SALES INVOICE", 16, bold=True, right=True)
    doc.text(555, 70, f"Invoice: {number}", 10, right=True)
    doc.text(555, 84, f"Tax point: {long_date(issued) if style == 'us' else eu_date(issued)}", 10, right=True)
    doc.text(40, 104, f"Invoice to: {BUYER}, Werkstrasse 3, 70173 Stuttgart", 10)
    cols = [(40, "Code", False), (150, "Description", False), (390, "Qty", True), (470, "Price", True),
            (555, "Net", True)]  # fmt: skip
    rows = [[pn, d, str(q), num(p, style), num(line_total((pn, d, q, p)), style)] for pn, d, q, p in lines]
    y = table(doc, 140, cols, rows)
    proforma = f"PF-{rng.randint(1000, 9999)}"
    y = totals(doc, y, [("Net total", num(sub, style), False), (f"VAT {rate * 100:g}%", num(total - sub, style), False),
                        ("Invoice total", num(total, style), False),
                        (f"Less deposit (proforma {proforma})", f"-{num(paid, style)}", False)])  # fmt: skip
    doc.text(445, y + 24, "BALANCE DUE", 13, bold=True, right=True)
    doc.text(555, y + 24, f"{currency} {num(total - paid, style)}", 13, bold=True, right=True)
    doc.text(40, y + 56, f"All amounts in {currency}. Deposit received with thanks.", 9)
    return label("invoice", sup, number, issued, currency, lines, total)


def lookalike(doc: Doc, rng: random.Random, sup: str) -> dict[str, Any]:
    name, address, region = SUPPLIERS[sup]
    style, currency = ("us", "USD") if region == "us" else ("eu", "EUR")
    rate = 0.0 if region == "us" else 0.19
    issued = date(2026, rng.randint(1, 9), rng.randint(13, 28))
    number = f"90{rng.randint(1000000, 9999999)}"
    similar = [c for c in CATALOG if c[0].startswith(("BRG-620", "HX-M8", "CON-M12", "VLV-BALL"))]
    picked = rng.sample(similar, 5)
    descs = {"BRG": "C3 clearance, 20x47x14 mm", "HX-": "DIN 933, 8.8, box of 100", "CON": "IP67, 5 m PUR cable",
             "VLV": "PN40, 2-piece, ISO 5211 F05"}  # fmt: skip
    lines = [(pn, f"{d}, {descs[pn[:3]]}", rng.choice([2, 4, 6, 12, 24]), round(rng.uniform(4, 250), 2))
             for pn, d in picked]  # fmt: skip
    sub = round(sum(line_total(li) for li in lines), 2)
    total = round(sub * (1 + rate), 2)
    fdate = us_date(issued) if style == "us" else eu_date(issued)

    doc.text(40, 50, name, 14, bold=True)
    doc.text(40, 64, address, 9)
    doc.text(555, 50, "INVOICE", 16, bold=True, right=True)
    for i, (k, v) in enumerate([("Billing document", number), ("Date", fdate),
                                ("Sales order", f"10{rng.randint(100000, 999999)}"),
                                ("Delivery", f"80{rng.randint(1000000, 9999999)}"),
                                ("Customer", "100423")]):  # fmt: skip
        doc.text(450, 70 + i * 12, k, 8.5, right=True)
        doc.text(555, 70 + i * 12, v, 8.5, right=True)
    doc.text(40, 104, f"Sold-to: {BUYER}, Werkstrasse 3, 70173 Stuttgart", 10)
    cols = [(40, "Item", False), (70, "Material", False), (160, "Description", False), (420, "Qty / UoM", True),
            (480, "Price", True), (555, "Value", True)]  # fmt: skip
    rows = [[str((i + 1) * 10), pn, d, f"{q} {'BOX' if pn.startswith('HX') else 'EA'}", num(p, style),
             num(line_total((pn, d, q, p)), style)] for i, (pn, d, q, p) in enumerate(lines)]  # fmt: skip
    y = table(doc, 160, cols, rows, size=8)
    tax = (f"VAT {rate * 100:g}%", num(total - sub, style), False) if rate else ("Sales tax", "exempt", False)
    y = totals(doc, y, [("Net value", num(sub, style), False), tax, (f"Total {currency}", num(total, style), True)])
    doc.text(40, y + 30, "Material numbers are manufacturer part numbers. Item = SAP position number.", 8)
    return label("invoice", sup, number, issued, currency, lines, total)


CASES: list[tuple[str, Callable[[Doc, random.Random, str], dict[str, Any]], tuple[str, str], bool]] = [
    ("german", german, ("nordwerk", "rhine"), False),
    ("billto", billto, ("apex", "keystone"), False),
    ("multipage", multipage, ("baltic", "sierra"), False),
    ("bulk", bulk, ("rhine", "baltic"), False),
    ("scanned", scanned, ("nordwerk", "keystone"), True),
    ("quote", quote, ("sierra", "keystone"), False),
    ("deposit", deposit, ("apex", "nordwerk"), False),
    ("lookalike", lookalike, ("sierra", "rhine"), False),
]


def main() -> None:
    OUT.mkdir(exist_ok=True)
    count = 0
    for case, draw, suppliers, is_scan in CASES:
        for variant, sup in enumerate(suppliers, start=1):
            doc_id = f"{case}-{variant}"
            rng = random.Random(f"{doc_id}-2026")  # noqa: S311 - deterministic test data, not security
            doc = Doc(OUT / f"{doc_id}.pdf", rng, scanned=is_scan)
            lab = draw(doc, rng, sup)
            doc.save()
            (OUT / f"{doc_id}.json").write_text(json.dumps(lab, indent=2, ensure_ascii=False) + "\n")
            count += 1
    print(f"wrote {count} documents to {OUT}")


if __name__ == "__main__":
    main()
