import io
from typing import Any, Dict, List
import pymupdf
import docx


def extract_pages_from_file(file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
    """
    Extract text segmented by pages/sections with page metadata.
    Returns: List of dicts [{"page": 1, "text": "content..."}, ...]
    """
    ext = filename.lower().split('.')[-1]
    pages: List[Dict[str, Any]] = []

    if ext == 'txt':
        content = file_bytes.decode('utf-8', errors='ignore').strip()
        if content:
            pages.append({"page": 1, "text": content})
        return pages

    elif ext == 'pdf':
        with pymupdf.open(stream=file_bytes, filetype="pdf") as doc:
            for page_idx, page in enumerate(doc, start=1):
                page_text = page.get_text().strip()
                if page_text:
                    pages.append({"page": page_idx, "text": page_text})
        return pages

    elif ext == 'docx':
        doc = docx.Document(io.BytesIO(file_bytes))
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        full_text = "\n\n".join(paragraphs)
        if full_text:
            pages.append({"page": 1, "text": full_text})
        return pages

    else:
        raise ValueError(f"Unsupported file extension: {ext}")


def extract_text_from_file(file_bytes: bytes, filename: str) -> str:
    """
    Extract unified text from a file based on its extension.
    Supported: .txt, .pdf, .docx
    """
    pages = extract_pages_from_file(file_bytes, filename)
    return "\n\n".join(p["text"] for p in pages)
