import pytest

from document_processor import (
    UnsupportedFileTypeError,
    UnreadableDocumentError,
    chunk_text,
    extract_text_from_txt_md,
    validate_file_extension,
)


def test_validate_file_extension_supported():
    assert validate_file_extension("report.txt") == ".txt"
    assert validate_file_extension("notes.MD") == ".md"
    assert validate_file_extension("file.PDF") == ".pdf"


def test_validate_file_extension_unsupported():
    with pytest.raises(UnsupportedFileTypeError):
        validate_file_extension("image.png")
    with pytest.raises(UnsupportedFileTypeError):
        validate_file_extension("data.csv")


def test_extract_text_from_txt_md_utf8():
    content = "Hello, world! This is a test.".encode("utf-8")
    text = extract_text_from_txt_md(content)
    assert "Hello, world!" in text


def test_extract_text_from_txt_md_unreadable():
    # simulate invalid bytes
    bad_bytes = b"\xff\xfe\x00\x00"
    with pytest.raises(UnreadableDocumentError):
        extract_text_from_txt_md(bad_bytes)


def test_chunk_text_basic():
    text = "abcde" * 300  # long text
    chunks = chunk_text(text, chunk_size=1000, overlap=200)
    assert len(chunks) >= 1
    # all chunks should be non-empty
    assert all(chunk.strip() for chunk in chunks)