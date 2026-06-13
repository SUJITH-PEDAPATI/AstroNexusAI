from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class BaseLoader(ABC):
    """
    Abstract base for all AstroNexus document loaders.

    Every concrete loader wraps a LangChain loader internally but
    always returns the same contract dict:
        {
            "full_text": str,
            "pages":     list[str],   # per-page text (empty list if N/A)
            "metadata":  dict,        # file-level metadata (title, authors, …)
        }
    """

    def __init__(self, file_path: str | Path) -> None:
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

    @abstractmethod
    def load(self) -> dict:
        """Load the document and return structured content."""
        ...

    def _base_metadata(self) -> dict:
        """Common metadata every loader must include."""
        return {
            "source_file": self.file_path.name,
            "file_type": self.file_path.suffix.lstrip(".").lower(),
        }