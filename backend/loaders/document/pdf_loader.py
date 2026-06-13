from __future__ import annotations

from backend.loaders.base_loader import BaseLoader


class PDFLoader(BaseLoader):
    """
    Load a PDF research paper using LangChain's PyMuPDFLoader.

    PyMuPDFLoader returns one LangChain Document per page, so we get
    per-page text for free — ideal for preserving page_num in chunks later.

    Install: pip install pymupdf langchain-community
    """

    def load(self) -> dict:
        try:
            from langchain_community.document_loaders import PyMuPDFLoader
        except ImportError as e:
            raise ImportError(
                "Install required packages: pip install pymupdf langchain-community"
            ) from e

        lc_loader = PyMuPDFLoader(str(self.file_path))
        documents = lc_loader.load()   # one Document per page

        # Each doc.page_content is one page; doc.metadata has page, source, etc.
        pages: list[str] = [doc.page_content for doc in documents]
        full_text: str = "\n\n".join(pages)
        # Pull whatever PyMuPDF surfaces in metadata (title, author, etc.)
        first_meta: dict = documents[0].metadata if documents else {}
        metadata = self._base_metadata()
        metadata.update(
            {
                "page_count": len(pages),
                "pdf_title":  first_meta.get("title", "").strip() or None,
                "pdf_author": first_meta.get("author", "").strip() or None,
                "pdf_subject": first_meta.get("subject", "").strip() or None,
            }
        )

        return {
            "full_text": full_text,
            "pages":     pages,
            "metadata":  metadata,
        }