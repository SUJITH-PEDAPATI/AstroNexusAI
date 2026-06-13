from __future__ import annotations
import re
 
# Common academic paper section headings
_SECTION_PATTERNS = re.compile(
    r"^\s*(?:\d+\.?\s+)?"   # optional numbering: "1. " or "1 "
    r"("
    r"abstract|introduction|background|related\s+work|"
    r"literature\s+review|methodology|methods?|"
    r"proposed\s+method|approach|model|architecture|"
    r"experiments?|experimental\s+setup|results?|"
    r"evaluation|discussion|conclusion|"
    r"future\s+work|references?|appendix|acknowledgements?"
    r")\s*$",
    re.IGNORECASE | re.MULTILINE,
)
 
 
def detect_section(text_chunk: str) -> str:
    """
    Return the most likely section name for a given chunk of text.
 
    Scans the first 3 lines of the chunk for a known section heading.
    Falls back to 'body' if nothing matches.
    """
    first_lines = "\n".join(text_chunk.splitlines()[:3])
    match = _SECTION_PATTERNS.search(first_lines)
    if match:
        # Normalize: title-case, strip numbers
        return match.group(1).strip().title()
    return "Body"
 
 
def split_by_sections(full_text: str) -> list[dict]:
    """
    Split document text into named sections.
 
    Returns a list of dicts:
        [{"section": "Introduction", "text": "..."}, ...]
 
    If no section headers are detected, returns the full text as one 'Body' section.
    """
    boundaries = [
        (m.start(), m.group(1).strip().title())
        for m in _SECTION_PATTERNS.finditer(full_text)
    ]
 
    if not boundaries:
        return [{"section": "Body", "text": full_text}]
    sections: list[dict] = []

    # Text before the first heading (if any) goes into a "Preamble" section
    if boundaries[0][0] > 0:
        pre = full_text[: boundaries[0][0]].strip()
        if pre:
            sections.append({"section": "Preamble", "text": pre})
 
    for i, (start, section_name) in enumerate(boundaries):
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(full_text)
        section_text = full_text[start:end].strip()
        if section_text:
            sections.append({"section": section_name, "text": section_text})
 
    return sections