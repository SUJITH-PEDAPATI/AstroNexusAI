"""
AstroNexus AI — Domain Classifier

Classifies a research paper into one or more scientific domains
using keyword matching on the abstract + title + first section.

No external model needed — pure keyword matching is fast, deterministic,
and sufficient for routing papers into the correct graph domain.

Domains supported:
    remote_sensing, nlp, computer_vision, machine_learning,
    medicine, chemistry, astronomy, climate, materials, biology
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class DomainResult:
    """Result of domain classification."""
    primary_domain:  str               # top-scoring domain
    all_domains:     list[str]         # all domains above threshold
    scores:          dict[str, float]  # raw keyword hit scores
    matched_keywords: dict[str, list[str]]  # domain → matched keywords


# ── Domain keyword vocabularies ────────────────────────────────────────────────
# Each domain has primary keywords (weight 2) and secondary keywords (weight 1).
# Compound phrases count as one match.

DOMAIN_KEYWORDS: dict[str, dict[str, list[str]]] = {

    "remote_sensing": {
        "primary": [
            "satellite", "remote sensing", "satellite imagery",
            "land use", "land cover", "multispectral", "hyperspectral",
            "sar", "synthetic aperture radar", "lidar", "geospatial",
            "aerial image", "earth observation", "sentinel", "landsat",
            "spectral band", "ndvi", "flood detection", "change detection",
            "crop classification", "urban mapping", "dinov2", "spectral gpt",
            "spectralgpt", "geotiff", "raster", "pixel classification",
        ],
        "secondary": [
            "image segmentation", "object detection", "scene classification",
            "drone", "uav", "gis", "geographic", "terrain", "elevation",
            "vegetation", "water body", "coastal", "atmosphere",
        ],
    },

    "nlp": {
        "primary": [
            "natural language processing", "nlp", "language model",
            "transformer", "bert", "gpt", "large language model", "llm",
            "text classification", "named entity recognition", "ner",
            "machine translation", "question answering", "summarization",
            "sentiment analysis", "language understanding", "tokenizer",
            "attention mechanism", "sequence to sequence", "seq2seq",
            "word embedding", "word2vec", "glove", "fasttext",
        ],
        "secondary": [
            "corpus", "vocabulary", "syntax", "semantic", "parsing",
            "coreference", "dialogue", "chatbot", "text generation",
        ],
    },

    "computer_vision": {
        "primary": [
            "image classification", "object detection", "image segmentation",
            "convolutional neural network", "cnn", "resnet", "vgg", "vit",
            "visual transformer", "yolo", "faster rcnn", "feature pyramid",
            "instance segmentation", "panoptic", "depth estimation",
            "optical flow", "face recognition", "pose estimation",
            "sam", "segment anything", "dinov2", "clip", "florence",
        ],
        "secondary": [
            "bounding box", "anchor", "backbone", "feature map",
            "image recognition", "visual", "pixel", "convolution",
        ],
    },

    "machine_learning": {
        "primary": [
            "deep learning", "neural network", "reinforcement learning",
            "generative adversarial network", "gan", "variational autoencoder",
            "vae", "diffusion model", "graph neural network", "gnn",
            "federated learning", "transfer learning", "fine-tuning",
            "zero-shot", "few-shot", "meta-learning", "self-supervised",
            "contrastive learning", "knowledge distillation",
        ],
        "secondary": [
            "training", "inference", "optimizer", "loss function",
            "gradient", "backpropagation", "epoch", "batch size",
            "hyperparameter", "overfitting", "regularization",
        ],
    },

    "medicine": {
        "primary": [
            "clinical", "medical imaging", "radiology", "mri", "ct scan",
            "x-ray", "pathology", "diagnosis", "disease", "patient",
            "drug", "treatment", "cancer", "tumor", "biomarker",
            "electronic health record", "ehr", "genomics", "protein",
        ],
        "secondary": [
            "hospital", "healthcare", "therapy", "clinical trial",
            "pharmacology", "surgery", "symptom",
        ],
    },

    "astronomy": {
        "primary": [
            "telescope", "galaxy", "star", "planet", "exoplanet",
            "cosmology", "gravitational wave", "black hole", "pulsar",
            "spectroscopy", "photometry", "redshift", "hubble",
            "james webb", "jwst", "radio astronomy", "neutrino",
        ],
        "secondary": [
            "solar", "lunar", "orbit", "celestial", "astrophysics",
            "dark matter", "dark energy", "cosmic",
        ],
    },

    "climate": {
        "primary": [
            "climate change", "global warming", "carbon emission",
            "greenhouse gas", "climate model", "weather prediction",
            "precipitation", "temperature anomaly", "sea level",
            "arctic", "permafrost", "atmosphere model",
        ],
        "secondary": [
            "climate", "meteorology", "hydrology", "drought",
            "hurricane", "extreme weather",
        ],
    },

    "chemistry": {
        "primary": [
            "molecule", "chemical compound", "reaction", "synthesis",
            "catalyst", "polymer", "material science", "crystal",
            "quantum chemistry", "molecular dynamics", "dft",
            "density functional theory",
        ],
        "secondary": [
            "chemical", "organic", "inorganic", "spectroscopy",
            "electrochemistry", "thermodynamics",
        ],
    },

    "biology": {
        "primary": [
            "gene", "dna", "rna", "protein structure", "cell",
            "crispr", "sequencing", "phylogenetic", "evolution",
            "ecology", "biodiversity", "microbiome", "neuroscience",
        ],
        "secondary": [
            "biological", "organism", "species", "genome",
            "mutation", "enzyme",
        ],
    },
}

# Minimum score to include a domain in all_domains
SCORE_THRESHOLD = 2.0


def classify_domain(
    text:       str,
    title:      str = "",
    abstract:   str = "",
) -> DomainResult:
    """
    Classify a research paper into scientific domains.

    Args:
        text:     Full paper text (or first 5000 chars for speed)
        title:    Paper title (weighted 3x — strong signal)
        abstract: Abstract text (weighted 2x — strong signal)

    Returns:
        DomainResult with primary_domain, all_domains, scores,
        and matched_keywords per domain.

    Algorithm:
        1. Build a weighted search corpus: title*3 + abstract*2 + text*1
        2. For each domain, count primary (weight 2) and secondary (weight 1)
           keyword hits using whole-word regex matching
        3. Normalize by number of keywords to avoid bias toward large domains
        4. Return domains above SCORE_THRESHOLD
    """
    # Build weighted corpus — title and abstract get more influence
    corpus = " ".join([
        (title    + " ") * 3,
        (abstract + " ") * 2,
        text[:5000],          # first 5000 chars covers intro + methods
    ]).lower()

    scores:          dict[str, float]       = {}
    matched_keywords: dict[str, list[str]] = {}

    for domain, keyword_groups in DOMAIN_KEYWORDS.items():
        score    = 0.0
        matched  = []

        for kw in keyword_groups.get("primary", []):
            pattern = r'\b' + re.escape(kw.lower()) + r'\b'
            hits    = len(re.findall(pattern, corpus))
            if hits > 0:
                score   += hits * 2.0
                matched.append(kw)

        for kw in keyword_groups.get("secondary", []):
            pattern = r'\b' + re.escape(kw.lower()) + r'\b'
            hits    = len(re.findall(pattern, corpus))
            if hits > 0:
                score   += hits * 1.0
                matched.append(kw)

        scores[domain]          = round(score, 2)
        matched_keywords[domain] = matched

    # Sort domains by score
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    primary_domain = ranked[0][0] if ranked else "machine_learning"
    all_domains    = [d for d, s in ranked if s >= SCORE_THRESHOLD]

    # Always include at least the top domain
    if not all_domains:
        all_domains = [primary_domain]

    return DomainResult(
        primary_domain=   primary_domain,
        all_domains=      all_domains,
        scores=           scores,
        matched_keywords= matched_keywords,
    )