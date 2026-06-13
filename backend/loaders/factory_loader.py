from __future__ import annotations

from pathlib import Path

from backend.loaders.base_loader import BaseLoader


# Maps file extension → (module path relative to backend.ingestion.loaders, class name)
_EXTENSION_MAP: dict[str, tuple[str, str]] = {
    "pdf":  ("document.pdf_loader",  "PDFLoader"),
    "docx": ("document.docx_loader", "DocxLoader"),
    "md":   ("document.md_loader",   "MarkdownLoader"),
    "txt":  ("document.txt_loader",  "TxtLoader"),
}


def _import_loader(module_rel: str, class_name: str) -> type[BaseLoader]:
    """Lazily import a loader class to avoid loading heavy deps at startup."""
    import importlib
    full_module = f"backend.loaders.{module_rel}"
    module = importlib.import_module(full_module)
    return getattr(module, class_name)


def get_loader(file_path: str | Path) -> BaseLoader:
    """
    Return the correct LangChain-backed loader for the given file.

    Usage:
        loader = get_loader("paper.pdf")
        result = loader.load()
        # result = {"full_text": ..., "pages": [...], "metadata": {...}}
    """
    path = Path(file_path)
    ext  = path.suffix.lstrip(".").lower()

    if ext not in _EXTENSION_MAP:
        supported = ", ".join(f".{e}" for e in _EXTENSION_MAP)
        raise ValueError(
            f"Unsupported file type '.{ext}'. Supported types: {supported}"
        )

    module_rel, class_name = _EXTENSION_MAP[ext]
    loader_class = _import_loader(module_rel, class_name)
    return loader_class(file_path)