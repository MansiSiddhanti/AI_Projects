import hashlib
import io
import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

# "transfor-\nmer" -> "transformer" (only lowercase-to-lowercase to keep
# genuine hyphenated compounds such as "COVID-19" intact).
_HYPHEN_BREAK = re.compile(r"([a-z])-\s*\n\s*([a-z])")
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class Page:
    number: int
    text: str


def clean_text(text: str) -> str:
    text = text.replace("­", "")  # soft hyphens
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    return _WHITESPACE.sub(" ", text).strip()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_pdf_bytes(data: bytes) -> list[Page]:
    # TODO: OCR fallback, scanned pdfs currently come back empty
    reader = PdfReader(io.BytesIO(data))
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if text:
            pages.append(Page(number=number, text=text))
    return pages


def load_pdf(path: str | Path) -> tuple[list[Page], str]:
    data = Path(path).read_bytes()
    return load_pdf_bytes(data), sha256_bytes(data)
