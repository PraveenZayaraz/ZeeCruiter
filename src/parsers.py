"""Defensive, in-memory PDF and text parsing."""

from dataclasses import dataclass
import io
from typing import BinaryIO, Union

import fitz

from config import CONFIG, AppConfig


class DocumentParsingError(ValueError):
    """Raised when an uploaded document cannot be handled safely."""


@dataclass(frozen=True)
class ParsedDocument:
    text: str
    page_count: int
    processed_pages: int
    warnings: tuple[str, ...] = ()


def _read_bytes(source: Union[bytes, bytearray, BinaryIO]) -> bytes:
    if isinstance(source, (bytes, bytearray)):
        return bytes(source)
    if hasattr(source, "getvalue"):
        return bytes(source.getvalue())
    if not hasattr(source, "read"):
        raise DocumentParsingError("Unsupported document source")
    position = None
    if hasattr(source, "tell"):
        try:
            position = source.tell()
        except (OSError, ValueError):
            position = None
    payload = source.read()
    if position is not None and hasattr(source, "seek"):
        try:
            source.seek(position)
        except (OSError, ValueError):
            pass
    return payload


def parse_pdf(
    source: Union[bytes, bytearray, BinaryIO], config: AppConfig = CONFIG
) -> ParsedDocument:
    payload = _read_bytes(source)
    if not payload:
        raise DocumentParsingError("The PDF is empty")
    if len(payload) > int(config.max_pdf_size_mb * 1024 * 1024):
        raise DocumentParsingError(
            f"The PDF exceeds the {config.max_pdf_size_mb:g} MB limit"
        )
    if not payload.lstrip().startswith(b"%PDF-"):
        raise DocumentParsingError("The uploaded file is not a valid PDF")

    try:
        document = fitz.open(stream=payload, filetype="pdf")
    except Exception as error:
        raise DocumentParsingError("The PDF is corrupt or unreadable") from error

    try:
        if document.needs_pass:
            raise DocumentParsingError("Password-protected PDFs are not supported")
        page_count = len(document)
        processed_pages = min(page_count, config.max_pdf_pages)
        warnings: list[str] = []
        if page_count > processed_pages:
            warnings.append(
                f"Only the first {processed_pages} of {page_count} pages were analyzed."
            )
        chunks = [document.load_page(index).get_text("text") for index in range(processed_pages)]
        text = clean_text("\n".join(chunks))
        if not text:
            warnings.append(
                "No selectable text was found; scanned PDFs require OCR before analysis."
            )
        return ParsedDocument(text, page_count, processed_pages, tuple(warnings))
    except DocumentParsingError:
        raise
    except Exception as error:
        raise DocumentParsingError("Text extraction failed") from error
    finally:
        document.close()


def parse_text(
    source: Union[str, bytes, bytearray, BinaryIO], max_bytes: int = 2_000_000
) -> str:
    if isinstance(source, str):
        return clean_text(source)
    payload = _read_bytes(source)
    if len(payload) > max_bytes:
        raise DocumentParsingError("The text file exceeds the 2 MB limit")
    try:
        return clean_text(payload.decode("utf-8-sig"))
    except UnicodeDecodeError as error:
        raise DocumentParsingError("Text files must use UTF-8 encoding") from error


def clean_text(text: str) -> str:
    lines = [" ".join(line.split()) for line in text.replace("\x00", "").splitlines()]
    return "\n".join(line for line in lines if line).strip()
