import io
from typing import List

from PyPDF2 import PdfReader


SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


class UnsupportedFileTypeError(Exception):
    pass


class UnreadableDocumentError(Exception):
    pass


def validate_file_extension(filename: str) -> str:
    """
    Validate that the file extension is one of the supported types.

    Returns the lowercase extension (e.g., '.pdf') or raises UnsupportedFileTypeError.
    """
    lower_name = filename.lower()
    for ext in SUPPORTED_EXTENSIONS:
        if lower_name.endswith(ext):
            return ext
    raise UnsupportedFileTypeError(
        "Unsupported file format. Please upload .txt, .md, or .pdf"
    )


def extract_text_from_txt_md(file_bytes: bytes) -> str:
    """
    Extract UTF-8 text from a .txt or .md file.
    """
    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise UnreadableDocumentError("Failed to decode text file as UTF-8") from exc


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extract text from a standard, text-based PDF using PyPDF2.
    """
    try:
        pdf_stream = io.BytesIO(file_bytes)
        reader = PdfReader(pdf_stream)
        text_parts: List[str] = []

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)

        full_text = "\n".join(text_parts).strip()
        if not full_text:
            raise UnreadableDocumentError(
                "PDF appears to contain no extractable text. It might be scanned or image-based."
            )
        return full_text
    except Exception as exc:
        raise UnreadableDocumentError(f"Failed to read PDF: {exc}") from exc


def extract_text(file_bytes: bytes, extension: str) -> str:
    """
    Dispatch text extraction based on file extension.
    """
    if extension in {".txt", ".md"}:
        text = extract_text_from_txt_md(file_bytes)
    elif extension == ".pdf":
        text = extract_text_from_pdf(file_bytes)
    else:
        # This should not happen if validate_file_extension is used correctly
        raise UnsupportedFileTypeError(
            "Unsupported file format. Please upload .txt, .md, or .pdf"
        )

    if not text or not text.strip():
        raise UnreadableDocumentError("Document appears unreadable or empty.")
    return text.strip()


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200,
) -> List[str]:
    """
    Split a long string into overlapping chunks.

    Example:
        chunk_size=1000, overlap=200 means:
        - chunk 1: text[0:1000]
        - chunk 2: text[800:1800]
        - etc.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    text = text.strip()
    chunks: List[str] = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == text_length:
            break
        # move start by (chunk_size - overlap)
        start = end - overlap

    return chunks