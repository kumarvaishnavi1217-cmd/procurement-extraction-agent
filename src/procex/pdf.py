import io
import re

from pypdf import PdfReader
from pypdf.errors import PdfReadError

PDF_MAGIC = b"%PDF-"


class InvalidPdfError(ValueError):
    pass


def extract_text(data: bytes) -> str:
    """Text layer of the PDF, used to ground evidence snippets. Empty for scanned PDFs."""
    if not data.startswith(PDF_MAGIC):
        raise InvalidPdfError("File is not a PDF")
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise InvalidPdfError("Encrypted PDFs are not supported")
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except PdfReadError as e:
        raise InvalidPdfError(f"Unreadable PDF: {e}") from e


_WS = re.compile(r"\s+")


def normalise(text: str) -> str:
    return _WS.sub(" ", text).strip().casefold()
