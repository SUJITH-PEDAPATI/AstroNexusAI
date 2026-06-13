from __future__ import annotations

from backend.loaders.base_loader import BaseLoader


class MarkdownLoader(BaseLoader):
    """
    Load a Markdown file using LangChain's UnstructuredMarkdownLoader.

    Unstructured parses headings, paragraphs, and code blocks properly,
    which improves chunk quality in the next pipeline stage.

    Install: pip install unstructured langchain-community
    """

    def load(self) -> dict:
        try:
            from langchain_community.document_loaders import UnstructuredMarkdownLoader
        except ImportError as e:
            raise ImportError(
                "Install required packages: pip install unstructured langchain-community"
            ) from e

        lc_loader = UnstructuredMarkdownLoader(
            str(self.file_path),
            mode="single",         # return full doc as one Document
        )
        documents = lc_loader.load()
        full_text: str = "\n\n".join(doc.page_content for doc in documents)
        metadata = self._base_metadata()
        metadata["page_count"] = 1

        # Extract title from first H1 heading
        for line in full_text.splitlines():
            stripped = line.strip()
            if stripped.startswith("# "):
                metadata["md_title"] = stripped.lstrip("# ").strip()
                break

        return {
            "full_text": full_text,
            "pages":     [full_text],
            "metadata":  metadata,
        }