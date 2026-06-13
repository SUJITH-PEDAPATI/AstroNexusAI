from __future__ import annotations

import json
import logging
import os
import re
import time
from collections import Counter

from backend.ingestion.models import RawDocument
from backend.graph.models import (
    AuthorNode, DatasetNode, ExtractionResult,
    GraphRelation, ModelNode, PaperNode,
    RelationType, TaskNode, VenueNode,
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL PATTERNS  (document-agnostic — valid across all scientific papers)
# ══════════════════════════════════════════════════════════════════════════════

# ── Author patterns ────────────────────────────────────────────────────────────
# Covers: "John Smith", "J. Smith", "John A. Smith", "Smith et al."
_AUTHOR_LINE_RE = re.compile(
    r"""
    (?:^|\n)                            # start of line
    \s*
    (
        [A-Z][a-záéíóúàèìòùâêîôûäëïöü\-]+ # First name (supports accented chars)
        (?:\s+[A-Z]\.)*                 # optional middle initials
        \s+
        [A-Z][a-záéíóúàèìòùâêîôûäëïöü\-]+ # Last name
        (?:\s*[∗†‡§¶\*\d,]+)?          # optional affiliation markers
    )
    (?:\s*,\s*|\s+and\s+|\s*\n)        # separator between authors
    """,
    re.VERBOSE | re.MULTILINE,
)

# Matches affiliation markers and footnote symbols to strip from names
_AFFILIATION_MARKERS_RE = re.compile(r'[∗†‡§¶\*\d,;]+$')

# ── Model name patterns ────────────────────────────────────────────────────────
# Covers: BERT, GPT-4, ConvS2S, Transformer, T5, DINOv2, Florence-2
_MODEL_PATTERN_RE = re.compile(
    r"""
    \b(
        # ALL-CAPS acronyms (2-8 chars), optionally versioned
        [A-Z]{2,8}(?:[-_]?v?\d+(?:\.\d+)?)?
        |
        # CamelCase names (at least 2 humps)
        [A-Z][a-z]+(?:[A-Z][a-z0-9]+)+(?:[-_]?v?\d+(?:\.\d+)?)?
        |
        # Hyphenated model names: Florence-2, GPT-4, Llama-3
        [A-Z][a-zA-Z0-9]+(?:-\d+(?:\.\d+)?)+
    )\b
    """,
    re.VERBOSE,
)

# False positive model names to always exclude
_MODEL_BLACKLIST = {
    # English words that match CamelCase/CAPS but aren't models
    "We", "In", "The", "For", "Our", "This", "These", "Such", "Note",
    "While", "Here", "When", "Then", "Also", "Both", "Each", "Since",
    "Given", "Using", "Thus", "Hence", "Table", "Figure", "Section",
    "Equation", "Algorithm", "Appendix", "However", "Moreover", "Therefore",
    # Common abbreviations
    "GPU", "CPU", "TPU", "RAM", "SSD", "API", "URL", "PDF", "XML", "JSON",
    "BLEU", "ROUGE", "MSE", "MAE", "AUC", "ROC", "SGD", "Adam", "ReLU",
    "NLP", "CV", "ML", "AI", "DL", "RL", "NER", "POS", "MT",
    # Units and math
    "GHz", "MHz", "GB", "MB", "KB", "FPS", "Hz",
}

# Well-known model names — always include regardless of pattern
_KNOWN_MODELS = {
    "Transformer", "BERT", "GPT", "GPT-2", "GPT-3", "GPT-4",
    "RoBERTa", "T5", "BART", "XLNet", "ALBERT", "ELECTRA",
    "DistilBERT", "LLaMA", "Mistral", "Falcon", "BLOOM",
    "CLIP", "DALL-E", "Whisper", "DINOv2", "Florence-2",
    "SAM", "SAM2", "ResNet", "VGG", "EfficientNet",
    "LSTM", "GRU", "Seq2Seq", "Word2Vec", "GloVe", "FastText",
    "ConvS2S", "ByteNet", "WaveNet", "DETR", "ViT",
}

# ── Dataset patterns ───────────────────────────────────────────────────────────
# Covers: WMT 2014, ImageNet, SQuAD 1.1, CIFAR-10, MS COCO
_DATASET_PATTERN_RE = re.compile(
    r"""
    \b(
        # CAPS name optionally followed by year or version
        [A-Z]{2,}(?:[-\s]\d{4})?(?:[-\s](?:v\d+|\d+\.\d+))?
        |
        # Mixed case with year: Penn Treebank, News Crawl 2014
        [A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+\d{4}
        |
        # Name + number version: CIFAR-10, SQuAD 2.0
        [A-Z][a-zA-Z]+[-]\d+(?:\.\d+)?
    )\b
    """,
    re.VERBOSE,
)

_DATASET_BLACKLIST = {
    "We", "In", "The", "For", "GPU", "CPU", "TPU",
    "BLEU", "ROUGE", "Adam", "SGD",
}

_KNOWN_DATASETS = {
    "ImageNet", "COCO", "CIFAR-10", "CIFAR-100", "MNIST", "FMNIST",
    "SQuAD", "SQuAD 2.0", "GLUE", "SuperGLUE", "WMT", "WMT 2014",
    "MS MARCO", "Natural Questions", "TriviaQA",
    "CC-News", "BookCorpus", "Penn Treebank",
    "newstest2013", "newstest2014",
}

# ── Task patterns ──────────────────────────────────────────────────────────────
# Detects tasks by suffix — works across any domain
_TASK_SUFFIX_RE = re.compile(
    r"""
    \b(
        [\w\s\-]{3,40}?
        (?:
            \s+(?:translation|classification|recognition|detection|
                  segmentation|generation|summarization|parsing|
                  modeling|transduction|understanding|prediction|
                  retrieval|ranking|reasoning|completion|tagging|
                  identification|estimation|synthesis|captioning)
        )
    )\b
    """,
    re.VERBOSE | re.IGNORECASE,
)

# ── Venue patterns ─────────────────────────────────────────────────────────────
_VENUE_RE = re.compile(
    r"""
    \b(
        (?:
            (?:Proceedings\s+of\s+(?:the\s+)?)?
            (?:Annual\s+)?
            (?:International\s+)?
            (?:Conference|Workshop|Symposium|Journal|Transactions|
               ACL|EMNLP|NAACL|ICLR|NeurIPS|ICML|CVPR|ICCV|ECCV|
               AAAI|IJCAI|ACM|IEEE|arXiv)
            [\w\s,\-]*
        )
    )\b
    """,
    re.VERBOSE,
)

# ── Non-person terms (structural/mathematical — universal across all papers) ───
_NON_PERSON_TERMS = {
    # Document structure
    "abstract", "introduction", "background", "conclusion", "appendix",
    "related work", "methodology", "experiments", "results", "discussion",
    "references", "acknowledgements", "figure", "table", "section",
    "equation", "algorithm", "theorem", "lemma", "proof",
    # Math/code notation
    "sublayer", "layernorm", "softmax", "sigmoid", "relu", "dropout",
    "encoder", "decoder", "attention", "embedding",
    # Hardware
    "gpu", "cpu", "tpu",
}

_NON_PERSON_CHARS = set("(){}[]+=*/<>\\^_|~`@#$%")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — STRUCTURAL AUTHOR EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def _extract_authors_structural(text: str) -> list[str]:
    """
    Extract authors from the paper header (first 2 pages).

    Strategy:
        1. Isolate the header block — text before the Abstract section
        2. Apply author regex pattern
        3. Validate and clean each candidate

    This is far more reliable than NER for author extraction because
    author names always appear in a predictable location and format.
    """
    # Isolate header: text before "Abstract" heading
    abstract_match = re.search(r'\bAbstract\b', text, re.IGNORECASE)
    header = text[:abstract_match.start()] if abstract_match else text[:2000]

    raw_names = _AUTHOR_LINE_RE.findall(header)
    authors: list[str] = []

    for name in raw_names:
        name = _AFFILIATION_MARKERS_RE.sub("", name).strip()
        name = re.sub(r'\s+', ' ', name)

        if _is_valid_author(name):
            authors.append(name)

    # Deduplicate preserving order
    return _dedup_ordered(authors)


def _is_valid_author(name: str) -> bool:
    """
    Universal author name validator.
    Works across all scientific papers regardless of domain.
    """
    if not name or len(name) < 4 or len(name) > 60:
        return False

    # Reject if contains math/code characters
    if any(c in name for c in _NON_PERSON_CHARS):
        return False

    # Reject known non-person structural terms
    if name.lower().strip() in _NON_PERSON_TERMS:
        return False

    # Must start with uppercase
    if not name[0].isupper():
        return False

    # Must contain at least one space (first + last name minimum)
    if ' ' not in name:
        return False

    # Reject if more than 5 words (likely a sentence fragment)
    if len(name.split()) > 5:
        return False

    # Reject if contains digits (e.g. "Layer 3", "Table 1")
    if re.search(r'\d', name):
        return False

    return True


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — SPACY NER + UNIVERSAL PATTERN MATCHING
# ══════════════════════════════════════════════════════════════════════════════

def _load_spacy():
    """Load spaCy model, downloading if needed."""
    try:
        import spacy
    except ImportError as e:
        raise ImportError("Install spaCy: pip install spacy") from e

    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        logger.info("[Extractor] Downloading spaCy model...")
        import subprocess, sys
        subprocess.run(
            [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
            check=True
        )
        return spacy.load("en_core_web_sm")


def _extract_models(text: str) -> list[str]:
    """
    Extract model names using:
        1. Pattern matching (CamelCase, ALL-CAPS, hyphenated versions)
        2. Known model vocabulary
        3. Frequency filter — must appear at least twice to reduce noise
    """
    candidates: Counter = Counter()

    # Pattern matching
    for match in _MODEL_PATTERN_RE.finditer(text):
        name = match.group(1).strip()
        if name not in _MODEL_BLACKLIST and len(name) >= 2:
            candidates[name] += 1

    # Known models — always include if present
    for model in _KNOWN_MODELS:
        pattern = re.compile(r'\b' + re.escape(model) + r'\b', re.IGNORECASE)
        count = len(pattern.findall(text))
        if count > 0:
            candidates[model] += count

    # Keep only names that appear at least twice (frequency filter)
    # Exception: known models always included
    results = [
        name for name, count in candidates.items()
        if count >= 2 or name in _KNOWN_MODELS
    ]

    return _dedup_canonical(results)


def _extract_datasets(text: str) -> list[str]:
    """
    Extract dataset names using:
        1. Pattern matching (CAPS + year, versioned names)
        2. Known dataset vocabulary
        3. Context validation — must appear near dataset-related keywords
    """
    dataset_context_re = re.compile(
        r'(?:dataset|corpus|benchmark|trained on|evaluated on|'
        r'training set|test set|data from|split)\s{0,30}',
        re.IGNORECASE
    )

    candidates: set[str] = set()

    # Known datasets — direct match
    for dataset in _KNOWN_DATASETS:
        pattern = re.compile(r'\b' + re.escape(dataset) + r'\b', re.IGNORECASE)
        if pattern.search(text):
            candidates.add(dataset)

    # Pattern matching with context validation
    for match in _DATASET_PATTERN_RE.finditer(text):
        name = match.group(1).strip()
        if name in _DATASET_BLACKLIST or len(name) < 3:
            continue
        # Check if near dataset-related context
        start = max(0, match.start() - 100)
        context = text[start : match.end() + 100]
        if dataset_context_re.search(context):
            candidates.add(name)

    return _dedup_canonical(list(candidates))


def _extract_tasks(text: str) -> list[str]:
    """
    Extract tasks using suffix pattern matching.
    Requires at least 2 occurrences to filter false positives.
    """
    candidates: Counter = Counter()

    for match in _TASK_SUFFIX_RE.finditer(text):
        task = match.group(1).strip().lower()
        # Filter out very short or very long matches
        if 5 < len(task) < 60:
            candidates[task.title()] += 1

    # Only keep tasks that appear at least twice
    return [task for task, count in candidates.items() if count >= 2]


def _extract_venues(text: str) -> list[str]:
    """Extract venue names from paper text."""
    candidates: set[str] = set()

    for match in _VENUE_RE.finditer(text[:5000]):  # venues usually in first pages
        venue = match.group(1).strip()
        if len(venue) > 3:
            candidates.add(venue)

    return list(candidates)


def _extract_with_spacy_and_patterns(text: str) -> dict:
    """
    Stage 2: Combined spaCy NER + universal pattern matching.
    """
    nlp = _load_spacy()

    # Process first 10000 chars — covers abstract + intro + methods
    sample = text[:10000]
    doc    = nlp(sample)

    authors:  set[str] = set()
    models:   set[str] = set()
    datasets: set[str] = set()
    venues:   set[str] = set()

    # spaCy NER pass
    for ent in doc.ents:
        name = ent.text.strip()
        if not name:
            continue

        if ent.label_ == "PERSON" and _is_valid_author(name):
            authors.add(name)
        elif ent.label_ in ("ORG", "PRODUCT", "WORK_OF_ART"):
            name_lower = name.lower()
            if any(m.lower() in name_lower for m in _KNOWN_MODELS):
                models.add(name)
            elif any(d.lower() in name_lower for d in _KNOWN_DATASETS):
                datasets.add(name)
        elif ent.label_ == "EVENT":
            venues.add(name)

    # Universal pattern matching on full text
    models   |= set(_extract_models(text))
    datasets |= set(_extract_datasets(text))
    tasks     = _extract_tasks(text)
    venues   |= set(_extract_venues(text))

    return {
        "authors":  _dedup_canonical(list(authors)),
        "models":   _dedup_canonical(list(models)),
        "datasets": _dedup_canonical(list(datasets)),
        "tasks":    tasks,
        "venues":   _dedup_canonical(list(venues)),
    }


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — LLM FALLBACK (HF Inference API)
# ══════════════════════════════════════════════════════════════════════════════

def _extract_with_llm(text: str, existing: dict) -> dict:
    """
    Stage 3: LLM-based extraction via HF Inference API.

    Only triggered when Stage 1+2 combined result is thin:
        - fewer than 2 real authors, OR
        - fewer than 1 model

    Uses structured JSON prompt for reliable parsing.
    """
    import requests

    token = os.environ.get("HF_API_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY")
    if not token:
        logger.warning("[Extractor] No HF token — skipping LLM fallback")
        return existing

    model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    url      = f"https://api-inference.huggingface.co/models/{model_id}"

    prompt = f"""<s>[INST]
You are an information extraction system for scientific papers.
Extract named entities from the paper excerpt below.
Return ONLY a valid JSON object. No explanation. No markdown.

JSON format:
{{
  "authors": ["Full Name", ...],
  "models": ["ModelName", ...],
  "datasets": ["DatasetName", ...],
  "tasks": ["task name", ...],
  "venues": ["Venue Name", ...]
}}

Rules:
- authors: full person names only (Firstname Lastname format)
- models: ML model names (BERT, Transformer, ResNet, etc.)
- datasets: dataset/benchmark names (ImageNet, WMT 2014, etc.)
- tasks: ML tasks (machine translation, image classification, etc.)
- venues: conference/journal names

Paper excerpt:
{text[:3000]}
[/INST]"""

    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 500,
            "temperature": 0.05,
            "return_full_text": False,
        },
        "options": {"wait_for_model": True},
    }

    for attempt in range(3):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)

            if resp.status_code == 200:
                raw       = resp.json()
                generated = raw[0].get("generated_text", "") if isinstance(raw, list) else ""

                json_match = re.search(r'\{.*\}', generated, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group())

                    # Merge: union of Stage 1+2 and LLM results
                    merged = {}
                    for key in ("authors", "models", "datasets", "tasks", "venues"):
                        existing_vals = set(existing.get(key, []))
                        llm_vals      = set(
                            v for v in parsed.get(key, [])
                            if isinstance(v, str) and v.strip()
                        )
                        # Validate authors from LLM too
                        if key == "authors":
                            llm_vals = {v for v in llm_vals if _is_valid_author(v)}

                        merged[key] = _dedup_canonical(list(existing_vals | llm_vals))

                    logger.info("[Extractor] LLM fallback merged successfully")
                    return merged

            elif resp.status_code == 503:
                logger.warning(f"[Extractor] Model loading — retry {attempt+1}/3")
                time.sleep(20)

        except json.JSONDecodeError:
            logger.warning("[Extractor] LLM returned invalid JSON — using Stage 2 results")
            break
        except Exception as e:
            logger.warning(f"[Extractor] LLM fallback error: {e}")
            break

    return existing


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

# Canonical name map — universal spellings used across all papers
_CANONICAL_NAMES = {
    "convs2s":     "ConvS2S",
    "bytenet":     "ByteNet",
    "wmt":         "WMT",
    "wmt2014":     "WMT 2014",
    "wmt 2014":    "WMT 2014",
    "imagenet":    "ImageNet",
    "squad":       "SQuAD",
    "squad 2.0":   "SQuAD 2.0",
    "bert":        "BERT",
    "gpt":         "GPT",
    "gpt-2":       "GPT-2",
    "gpt-3":       "GPT-3",
    "gpt-4":       "GPT-4",
    "roberta":     "RoBERTa",
    "xlnet":       "XLNet",
    "distilbert":  "DistilBERT",
    "albert":      "ALBERT",
    "electra":     "ELECTRA",
    "lstm":        "LSTM",
    "gru":         "GRU",
    "glue":        "GLUE",
    "superglue":   "SuperGLUE",
    "ms coco":     "MS COCO",
    "ms marco":    "MS MARCO",
}


def _canonical(name: str) -> str:
    """Return canonical spelling of a name if known, else return as-is."""
    return _CANONICAL_NAMES.get(name.lower().strip(), name)


def _dedup_canonical(names: list[str]) -> list[str]:
    """
    Deduplicate a list of names by canonical lowercase key.
    Prefers the canonical form when duplicates exist.
    """
    seen:   set[str]  = set()
    result: list[str] = []

    for name in names:
        canonical = _canonical(name)
        key       = canonical.lower().strip()
        if key not in seen:
            seen.add(key)
            result.append(canonical)

    return result


def _dedup_ordered(names: list[str]) -> list[str]:
    """Deduplicate preserving original order."""
    seen:   set[str]  = set()
    result: list[str] = []
    for name in names:
        key = name.lower().strip()
        if key not in seen:
            seen.add(key)
            result.append(name)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def extract_entities(doc: RawDocument) -> ExtractionResult:
    """
    Extract all entities and relations from a RawDocument.

    Three-stage pipeline:
        Stage 1 — Structural author extraction from paper header
        Stage 2 — spaCy NER + universal pattern matching
        Stage 3 — LLM fallback via HF API (only if Stage 1+2 is thin)

    Designed to work well across all scientific paper domains:
        NLP, CV, ML, satellite imagery, physics, biology, etc.

    Args:
        doc: Output from ingest_paper()

    Returns:
        ExtractionResult ready for graph_builder.py
    """
    logger.info(f"[Extractor] Extracting entities: '{doc.metadata.title}'")

    # ── Stage 1: Structural author extraction ─────────────────────────────────
    structural_authors = _extract_authors_structural(doc.full_text)
    logger.info(f"[Extractor] Stage 1 — {len(structural_authors)} authors from header")

    # ── Stage 2: spaCy + pattern matching ────────────────────────────────────
    stage2 = _extract_with_spacy_and_patterns(doc.full_text)

    # Merge Stage 1 authors with Stage 2 authors (Stage 1 takes priority)
    all_authors = _dedup_canonical(structural_authors + stage2.get("authors", []))
    stage2["authors"] = all_authors

    logger.info(
        f"[Extractor] Stage 2 — "
        f"{len(stage2['models'])} models, "
        f"{len(stage2['datasets'])} datasets, "
        f"{len(stage2['tasks'])} tasks"
    )

    # ── Stage 3: LLM fallback if result is thin ───────────────────────────────
    needs_llm = (
        len(stage2.get("authors",  [])) < 2 or
        len(stage2.get("models",   [])) < 1
    )

    if needs_llm:
        logger.info("[Extractor] Stage 3 — LLM fallback triggered")
        final = _extract_with_llm(doc.full_text, stage2)
    else:
        final = stage2
        logger.info("[Extractor] Stage 3 — skipped (Stage 1+2 sufficient)")

    # ── Build PaperNode ───────────────────────────────────────────────────────
    paper_node = PaperNode(
        title=doc.metadata.title or "Unknown",
        year=doc.metadata.year,
        doi=doc.metadata.doi,
        abstract=doc.metadata.abstract,
        source_file=doc.metadata.source_file,
        paper_id=doc.paper_id,
    )

    # ── Build entity nodes ────────────────────────────────────────────────────
    author_nodes  = [AuthorNode(name=n)  for n in final.get("authors",  []) if n.strip()]
    model_nodes   = [ModelNode(name=n)   for n in final.get("models",   []) if n.strip()]
    dataset_nodes = [DatasetNode(name=n) for n in final.get("datasets", []) if n.strip()]
    task_nodes    = [TaskNode(name=n)    for n in final.get("tasks",    []) if n.strip()]
    venue_nodes   = [VenueNode(name=n)   for n in final.get("venues",   []) if n.strip()]

    # ── Build relations ───────────────────────────────────────────────────────
    relations: list[GraphRelation] = []

    for author  in author_nodes:
        relations.append(GraphRelation(
            from_id=paper_node.node_id,
            to_id=author.node_id,
            relation_type=RelationType.AUTHORED_BY,
        ))
    for model in model_nodes:
        relations.append(GraphRelation(
            from_id=paper_node.node_id,
            to_id=model.node_id,
            relation_type=RelationType.USES,
        ))
    for dataset in dataset_nodes:
        relations.append(GraphRelation(
            from_id=paper_node.node_id,
            to_id=dataset.node_id,
            relation_type=RelationType.USES,
        ))
    for task in task_nodes:
        relations.append(GraphRelation(
            from_id=paper_node.node_id,
            to_id=task.node_id,
            relation_type=RelationType.SOLVES,
        ))
    for venue in venue_nodes:
        relations.append(GraphRelation(
            from_id=paper_node.node_id,
            to_id=venue.node_id,
            relation_type=RelationType.PUBLISHED_AT,
        ))

    result = ExtractionResult(
        paper=paper_node,
        authors=author_nodes,
        models=model_nodes,
        datasets=dataset_nodes,
        tasks=task_nodes,
        venues=venue_nodes,
        relations=relations,
    )

    logger.info(
        f"[Extractor] Complete — "
        f"{len(author_nodes)} authors, "
        f"{len(model_nodes)} models, "
        f"{len(dataset_nodes)} datasets, "
        f"{len(task_nodes)} tasks, "
        f"{len(venue_nodes)} venues, "
        f"{len(relations)} relations"
    )

    return result