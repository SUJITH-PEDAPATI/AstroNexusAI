from __future__ import annotations

from backend.loaders.base_loader import BaseLoader


class TxtLoader(BaseLoader):
    """
    Load a plain text file using LangChain's TextLoader.

    TextLoader is built into langchain-core — no extra install needed.
    """

    def load(self) -> dict:
        try:
            from langchain_community.document_loaders import TextLoader
        except ImportError as e:
            raise ImportError(
                "Install required packages: pip install langchain-community"
            ) from e

        lc_loader = TextLoader(str(self.file_path), encoding="utf-8")
        documents = lc_loader.load()

        full_text: str = "\n\n".join(doc.page_content for doc in documents)

        metadata = self._base_metadata()
        metadata["page_count"] = 1

        return {
            "full_text": full_text,
            "pages":     [full_text],
            "metadata":  metadata,
        }