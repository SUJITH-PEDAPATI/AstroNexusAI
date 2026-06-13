from __future__ import annotations

import re
import unicodedata


# ── Patterns to strip ──────────────────────────────────────────────────────────

# Common academic header/footer patterns
_PAGE_NUMBER_RE = re.compile(r"^\s*-?\s*\d+\s*-?\s*$", re.MULTILINE)
_RUNNING_HEADER_RE = re.compile(
    r"^.{0,80}(arXiv|Preprint|Conference|Proceedings|Journal|Volume|Vol\.|No\.|pp\.|©|Copyright).{0,80}$",
    re.MULTILINE | re.IGNORECASE,
)
_URL_IN_ISOLATION_RE = re.compile(
    r"^\s*https?://\S+\s*$", re.MULTILINE
)

# Add to existing patterns


_HYPHEN_BREAK_RE = re.compile(r"(\w)-\n(\w)")
_LIGATURE_MAP = str.maketrans(
    {
        "\ufb00": "ff",
        "\ufb01": "fi",
        "\ufb02": "fl",
        "\ufb03": "ffi",
        "\ufb04": "ffl",
        "\ufb05": "st",
        "\ufb06": "st",
    }
)
# Special tokens leaked from PDF rendering
_SPECIAL_TOKENS_RE = re.compile(r'<[A-Z]+>|<\/[A-Z]+>')

# PDF watermark/permission lines
_WATERMARK_RE = re.compile(
    r'^.*provided proper attribution.*permission to.*$',
    re.IGNORECASE | re.MULTILINE
)
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_MULTI_SPACE_RE = re.compile(r"[^\S\n]+")


def clean_text(raw_text: str) -> str:
    """
    Clean raw text extracted from a research paper.

    Steps:
        1. Fix ligatures and unicode normalization
        2. Rejoin hyphenated line-breaks
        3. Strip page numbers and running headers/footers
        4. Strip isolated URLs
        5. Normalize whitespace
    """
    if not raw_text:
        return ""

    # 1. Unicode normalize (NFC) + ligature fix
    text = unicodedata.normalize("NFC", raw_text)
    text = text.translate(_LIGATURE_MAP)

    # 2. Rejoin hyphenated line-breaks before anything else
    text = _HYPHEN_BREAK_RE.sub(r"\1\2", text)

    # 3. Strip page numbers
    text = _PAGE_NUMBER_RE.sub("", text)

    # 4. Strip running headers/footers
    text = _RUNNING_HEADER_RE.sub("", text)

    # 5. Strip isolated URLs (keep inline URLs)
    text = _URL_IN_ISOLATION_RE.sub("", text)

    # 6. Normalize whitespace
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)

    # 6. Strip special tokens
    text = _SPECIAL_TOKENS_RE.sub("", text)

    # 7. Strip watermark lines
    text = _WATERMARK_RE.sub("", text)
    text = text.strip()

    return text


def clean_pages(pages: list[str]) -> list[str]:
    """Clean each page individually (used for per-page metadata)."""
    return [clean_text(p) for p in pages]
