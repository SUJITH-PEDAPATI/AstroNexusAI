from __future__ import annotations

from backend.loaders.base_loader import BaseLoader
class DocxLoader(BaseLoader):
    """
    Load a .docx file using LangChain's Docx2txtLoader.

    Returns the full document as a single page (no per-page breakdown
    is possible with .docx format).

    Install: pip install docx2txt langchain-community
    """

    def load(self) -> dict:
        try:
            from langchain_community.document_loaders import Docx2txtLoader
        except ImportError as e:
            raise ImportError(
                "Install required packages: pip install docx2txt langchain-community"
            ) from e

        lc_loader = Docx2txtLoader(str(self.file_path))
        documents = lc_loader.load()   

        full_text: str = "\n\n".join(doc.page_content for doc in documents)
        first_meta: dict = documents[0].metadata if documents else {}

        metadata = self._base_metadata()
        metadata.update(
            {
                "page_count":  None,  
                "docx_title":  first_meta.get("title") or None,
                "docx_author": first_meta.get("author") or None,
            }
        )

        return {
            "full_text": full_text,
            "pages":     [full_text],  
            "metadata":  metadata,
        }