"""Deterministic checks on the model's output.

The model's self-reported confidence is only a starting point. Each rule that fails caps the confidence
of the fields it implicates, and any field that ends below the review threshold goes to a human.
"""

import contextlib
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from procex.pdf import normalise
from procex.schema import (
    REQUIRED_HEADER_FIELDS,
    REQUIRED_LINE_FIELDS,
    ExtractedDocument,
    NumberField,
    TextField,
)

# Confidence caps by severity
ERROR_CAP = 0.3  # value is missing, malformed, or contradicts other values
UNGROUNDED_CAP = 0.5  # value cannot be found in the document text
NO_TEXT_LAYER_CAP = 0.75  # scanned PDF: nothing to ground against, always review

CURRENCIES = {
    "AUD", "CAD", "CHF", "CNY", "CZK", "DKK", "EUR", "GBP", "HUF", "INR", "JPY", "MXN", "NOK", "NZD",
    "PLN", "SEK", "SGD", "USD", "ZAR",
}  # fmt: skip
PO_LABEL = re.compile(r"\b(p\.?o\.?|purchase order|order no\.?|your order|bestell\w*)\b", re.IGNORECASE)
PART_NUMBER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/\-]{1,39}$")
_NUMBER_TOKEN = re.compile(r"\d[\d.,']*")


@dataclass
class FieldResult:
    path: str
    value: Any
    model_confidence: float
    confidence: float
    issues: list[str] = field(default_factory=list)

    def flag(self, issue: str, cap: float) -> None:
        self.issues.append(issue)
        self.confidence = min(self.confidence, cap)


@dataclass
class ValidationResult:
    fields: dict[str, FieldResult]
    document_issues: list[str]

    def needs_review(self, threshold: float) -> list[FieldResult]:
        return [f for f in self.fields.values() if f.confidence < threshold]

    def data(self) -> dict[str, Any]:
        """Nested plain values, the shape stored and returned by the API."""
        out: dict[str, Any] = {}
        for path, f in self.fields.items():
            set_path(out, path, f.value)
        out.setdefault("line_items", [])
        return out


def validate(doc: ExtractedDocument, pdf_text: str, buyer_names: list[str] | None = None) -> ValidationResult:
    fields: dict[str, FieldResult] = {}
    document_issues: list[str] = []
    text = normalise(pdf_text)
    has_text_layer = len(text) > 20

    def add(path: str, f: TextField | NumberField, kind: str) -> FieldResult:
        conf = max(0.0, min(1.0, f.confidence))
        result = FieldResult(path, f.value, conf, conf)
        fields[path] = result
        if f.value is None:
            return result
        if not has_text_layer:
            result.flag("no_text_layer", NO_TEXT_LAYER_CAP)
        elif not f.evidence or normalise(f.evidence) not in text:
            result.flag("evidence_not_in_document", UNGROUNDED_CAP)
        elif (
            kind == "number"
            and not _number_in_evidence(float(f.value), f.evidence)
            or kind == "text"
            and normalise(str(f.value)) not in normalise(f.evidence)
        ):
            result.flag("value_not_in_evidence", UNGROUNDED_CAP)
        return result

    # header
    fields["document_type"] = FieldResult("document_type", doc.document_type, 1.0, 1.0)
    if doc.document_type == "other":
        fields["document_type"].flag("not_a_quote_or_invoice", ERROR_CAP)
    add("supplier_name", doc.supplier_name, "text")
    add("document_number", doc.document_number, "text")
    doc_date = add("document_date", doc.document_date, "date")
    currency = add("currency", doc.currency, "code")
    subtotal = add("subtotal", doc.subtotal, "number")
    total = add("total", doc.total, "number")

    for name in REQUIRED_HEADER_FIELDS:
        f = fields[name]
        if f.value is None or not str(f.value).strip():
            f.flag("missing_required", ERROR_CAP)
    supplier = fields["supplier_name"]
    if supplier.value and normalise(str(supplier.value)) in {normalise(b) for b in buyer_names or []}:
        supplier.flag("supplier_is_buyer", ERROR_CAP)
    if doc.document_number.evidence and PO_LABEL.search(doc.document_number.evidence):
        fields["document_number"].flag("looks_like_po_number", ERROR_CAP)
    if doc_date.value is not None and not _is_iso_date(str(doc_date.value)):
        doc_date.flag("invalid_date", ERROR_CAP)
    if currency.value is not None and str(currency.value).upper() not in CURRENCIES:
        currency.flag("unknown_currency", ERROR_CAP)

    # line items
    if not doc.line_items:
        document_issues.append("no_line_items")
    line_totals: list[float] = []
    for i, item in enumerate(doc.line_items):
        p = f"line_items[{i}]"
        part = add(f"{p}.part_number", item.part_number, "text")
        add(f"{p}.description", item.description, "text")
        qty = add(f"{p}.quantity", item.quantity, "number")
        unit = add(f"{p}.unit_price", item.unit_price, "number")
        line_total = add(f"{p}.line_total", item.line_total, "number")

        for name in REQUIRED_LINE_FIELDS:
            f = fields[f"{p}.{name}"]
            if f.value is None:
                f.flag("missing_required", ERROR_CAP)
        if part.value is not None and not PART_NUMBER.match(str(part.value)):
            part.flag("invalid_part_number", ERROR_CAP)
        if qty.value is not None and qty.value <= 0:
            qty.flag("non_positive_quantity", ERROR_CAP)
        if unit.value is not None and unit.value < 0:
            unit.flag("negative_price", ERROR_CAP)
        values = (qty.value, unit.value, line_total.value)
        if None not in values and not _close(values[0] * values[1], values[2]):
            for f in (qty, unit, line_total):
                f.flag("line_arithmetic_mismatch", ERROR_CAP)
        if line_total.value is not None:
            line_totals.append(line_total.value)

    if subtotal.value is not None and line_totals and not _close(sum(line_totals), subtotal.value):
        subtotal.flag("subtotal_mismatch", ERROR_CAP)
    if None not in (subtotal.value, total.value) and total.value < subtotal.value - 0.01:
        total.flag("total_below_subtotal", ERROR_CAP)

    return ValidationResult(fields, document_issues)


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= max(0.01, abs(b) * 0.001)


def _is_iso_date(value: str) -> bool:
    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return True


def parse_number_candidates(token: str) -> set[float]:
    """All plausible readings of a printed number: 1,234.50 / 1.234,50 / 1'234.50."""
    token = token.rstrip(".,").replace("'", "")
    candidates: set[float] = set()
    for decimal_sep, group_sep in ((".", ","), (",", ".")):
        cleaned = token.replace(group_sep, "").replace(decimal_sep, ".")
        with contextlib.suppress(ValueError):
            candidates.add(float(cleaned))
    return candidates


def _number_in_evidence(value: float, evidence: str) -> bool:
    return any(
        abs(candidate - value) < 0.005
        for token in _NUMBER_TOKEN.findall(evidence)
        for candidate in parse_number_candidates(token)
    )


_PATH_PART = re.compile(r"(\w+)(?:\[(\d+)\])?")


def set_path(target: dict[str, Any], path: str, value: Any) -> None:
    """Set ``line_items[0].quantity``-style paths on nested dicts/lists."""
    parts = path.split(".")
    node: Any = target
    for depth, part in enumerate(parts):
        m = _PATH_PART.fullmatch(part)
        if m is None:
            raise ValueError(f"Invalid field path: {path}")
        key, index = m.group(1), m.group(2)
        last = depth == len(parts) - 1
        if index is None:
            if last:
                node[key] = value
            else:
                node = node.setdefault(key, {})
        else:
            items = node.setdefault(key, [])
            i = int(index)
            while len(items) <= i:
                items.append({})
            if last:
                items[i] = value
            else:
                node = items[i]
