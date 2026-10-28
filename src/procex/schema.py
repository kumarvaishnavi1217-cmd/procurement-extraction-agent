"""Extraction schema.

``ExtractedDocument`` is the schema Claude's structured output is constrained to. Every field carries the
model's self-reported confidence and a verbatim evidence snippet, so the validator can check the value is
actually grounded in the PDF instead of trusting the model's confidence alone.
"""

from typing import Literal

from pydantic import BaseModel, Field


class TextField(BaseModel):
    value: str | None = Field(description="Extracted value, or null if not present in the document")
    confidence: float = Field(description="Your confidence from 0.0 to 1.0 that the value is correct")
    evidence: str | None = Field(description="Exact text copied verbatim from the document supporting the value")


class NumberField(BaseModel):
    value: float | None = Field(description="Extracted number as a plain decimal (1234.5), or null")
    confidence: float = Field(description="Your confidence from 0.0 to 1.0 that the value is correct")
    evidence: str | None = Field(description="Exact text copied verbatim from the document supporting the value")


class LineItem(BaseModel):
    part_number: TextField = Field(description="Supplier or manufacturer part number / SKU")
    description: TextField
    quantity: NumberField
    unit_price: NumberField = Field(description="Price per unit, excluding tax")
    line_total: NumberField = Field(description="Line amount as printed on the document")


class ExtractedDocument(BaseModel):
    document_type: Literal["quote", "invoice", "other"]
    supplier_name: TextField = Field(description="Company issuing the quote or invoice (not the buyer)")
    document_number: TextField = Field(description="Quote number or invoice number")
    document_date: TextField = Field(description="Issue date normalised to ISO 8601, YYYY-MM-DD")
    currency: TextField = Field(description="ISO 4217 currency code, e.g. EUR, USD, GBP")
    line_items: list[LineItem]
    subtotal: NumberField = Field(description="Total before tax")
    total: NumberField = Field(description="Grand total including tax")


# Fields that must be present for a document to be usable downstream.
REQUIRED_HEADER_FIELDS = ("supplier_name", "document_number", "currency")
REQUIRED_LINE_FIELDS = ("part_number", "quantity", "unit_price")
