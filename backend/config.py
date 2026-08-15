"""
AstroNexus AI — Central Configuration

All constants, paths, and environment variables in one place.
Every module imports from here — no hardcoded strings elsewhere.

Usage:
    from backend.config import cfg

    model = cfg.WHISPER_MODEL
    key   = cfg.GEMINI_API_KEY
"""
from __future__ import annotations

import os
from pathlib import Path

# Load .env if python-dotenv is installed
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class _Config:
    """
    Single configuration object.
    All values read from environment variables with sensible defaults.
    """

    # ── Project paths ──────────────────────────────────────────────────────────
    BASE_DIR:    Path = Path(__file__).resolve().parent.parent  # project root
    OUTPUT_DIR:  Path = BASE_DIR / "output"
    DATA_DIR:    Path = BASE_DIR / "data"
    MODELS_DIR:  Path = BASE_DIR / "models"
    AUDIO_DIR:   Path = BASE_DIR / "output" / "audio"

    # ── API Keys ───────────────────────────────────────────────────────────────
    GEMINI_API_KEY: str = os.environ.get("GEMINI_API_KEY", "")
    HF_API_TOKEN:   str = os.environ.get("HF_API_TOKEN", "")

    # ── Qdrant ─────────────────────────────────────────────────────────────────
    QDRANT_HOST:            str = os.environ.get("QDRANT_HOST", "localhost")
    QDRANT_PORT:            int = int(os.environ.get("QDRANT_PORT", "6333"))
    QDRANT_COLLECTION:      str = os.environ.get("QDRANT_COLLECTION", "papers")
    QDRANT_SATELLITE_COLL:  str = os.environ.get("QDRANT_SATELLITE_COLL", "satellite_images")

    # ── Neo4j ──────────────────────────────────────────────────────────────────
    NEO4J_URI:      str = os.environ.get("NEO4J_URI",      "bolt://localhost:7687")
    NEO4J_USER:     str = os.environ.get("NEO4J_USER",     "neo4j")
    NEO4J_PASSWORD: str = os.environ.get("NEO4J_PASSWORD", "astronexus123")

    # ── Ollama ─────────────────────────────────────────────────────────────────
    OLLAMA_BASE_URL: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL:    str = os.environ.get("OLLAMA_MODEL",    "qwen3:4b")

    # ── Embedding model ────────────────────────────────────────────────────────
    EMBEDDING_MODEL:    str = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3")
    EMBEDDING_DIM:      int = int(os.environ.get("EMBEDDING_DIM", "1024"))

    # ── RAG ────────────────────────────────────────────────────────────────────
    RAG_TOP_K:           int   = int(os.environ.get("RAG_TOP_K",           "5"))
    RAG_SCORE_THRESHOLD: float = float(os.environ.get("RAG_SCORE_THRESHOLD", "0.40"))
    CHUNK_SIZE:          int   = int(os.environ.get("CHUNK_SIZE",           "1024"))
    CHUNK_OVERLAP:       int   = int(os.environ.get("CHUNK_OVERLAP",        "128"))

    # ── Vision ─────────────────────────────────────────────────────────────────
    DINOV2_MODEL:       str = os.environ.get("DINOV2_MODEL", "facebook/dinov2-base")
    DINOV2_VECTOR_DIM:  int = int(os.environ.get("DINOV2_VECTOR_DIM", "768"))
    GEMINI_MODEL:       str = os.environ.get("GEMINI_MODEL",       "gemini-3.1-flash-lite")
    GEMINI_VISION_MODEL:str = os.environ.get("GEMINI_VISION_MODEL", "gemini-3.1-flash-lite")
    SAM2_MODEL:         str = os.environ.get("SAM2_MODEL", "facebook/sam2-hiera-base-plus")
    VISION_MAX_IMG_SIZE:int = int(os.environ.get("VISION_MAX_IMG_SIZE", "1024"))

    # ── Whisper ────────────────────────────────────────────────────────────────
    WHISPER_MODEL: str = os.environ.get("WHISPER_MODEL", "base")

    # ── TTS ────────────────────────────────────────────────────────────────────
    TTS_VOICE:    str  = os.environ.get("TTS_VOICE",    "en-US-AriaNeural")
    TTS_RATE:     str  = os.environ.get("TTS_RATE",     "+0%")
    TTS_OUT_PATH: Path = OUTPUT_DIR / "audio" / "response.mp3"

    # ── FastAPI ────────────────────────────────────────────────────────────────
    API_HOST:     str = os.environ.get("API_HOST", "0.0.0.0")
    API_PORT:     int = int(os.environ.get("API_PORT", "8000"))
    API_RELOAD:   bool = os.environ.get("API_RELOAD", "false").lower() == "true"
    CORS_ORIGINS: list = os.environ.get(
        "CORS_ORIGINS", "http://localhost:3000,http://localhost:8080"
    ).split(",")

    # ── Logging ────────────────────────────────────────────────────────────────
    LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO")

    def ensure_dirs(self) -> None:
        """Create all output directories if they don't exist."""
        for d in [self.OUTPUT_DIR, self.DATA_DIR, self.AUDIO_DIR]:
            d.mkdir(parents=True, exist_ok=True)

    def validate(self) -> list[str]:
        """
        Check required config values.
        Returns list of warning strings (empty = all good).
        """
        warnings = []
        if not self.GEMINI_API_KEY:
            warnings.append("GEMINI_API_KEY not set — vision captioning will fail")
        if not self.HF_API_TOKEN:
            warnings.append("HF_API_TOKEN not set — embedding via HF API will fail")
        return warnings

    def __repr__(self) -> str:
        return (
            f"AstroNexusConfig("
            f"qdrant={self.QDRANT_HOST}:{self.QDRANT_PORT}, "
            f"neo4j={self.NEO4J_URI}, "
            f"ollama={self.OLLAMA_BASE_URL}, "
            f"gemini={'set' if self.GEMINI_API_KEY else 'NOT SET'}"
            f")"
        )


# ── Singleton instance ─────────────────────────────────────────────────────────
cfg = _Config()
cfg.ensure_dirs()