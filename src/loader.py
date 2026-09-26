from pathlib import Path
from typing import Any, Dict, List

from docx import Document
from pypdf import PdfReader


SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx"}


def load_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def load_pdf(file_path: str) -> List[Dict[str, Any]]:
    reader = PdfReader(file_path)
    pages: List[Dict[str, Any]] = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append({"text": text, "page": page_number})

    return pages


def load_docx(file_path: str) -> str:
    document = Document(file_path)
    paragraphs: List[str] = []

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if text:
            paragraphs.append(text)

    return "\n".join(paragraphs)


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    if not text or not text.strip():
        return []

    normalized = text.strip()
    if len(normalized) <= chunk_size:
        return [normalized]

    chunks: List[str] = []
    start = 0

    while start < len(normalized):
        end = min(len(normalized), start + chunk_size)
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(normalized):
            break
        start += max(1, chunk_size - overlap)

    return chunks


def load_documents(folder_path: str) -> List[Dict[str, Any]]:
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"Document folder not found: {folder}")

    documents: List[Dict[str, Any]] = []

    for file_path in sorted(folder.iterdir()):
        if not file_path.is_file():
            continue

        extension = file_path.suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            continue

        if extension == ".txt":
            text = load_txt(str(file_path))
            if text.strip():
                documents.append({"text": text, "source": file_path.name})

        elif extension == ".pdf":
            pages = load_pdf(str(file_path))
            for page in pages:
                documents.append(
                    {
                        "text": page["text"],
                        "source": file_path.name,
                        "page": page["page"],
                    }
                )

        elif extension == ".docx":
            text = load_docx(str(file_path))
            if text.strip():
                documents.append({"text": text, "source": file_path.name})

    return documents
