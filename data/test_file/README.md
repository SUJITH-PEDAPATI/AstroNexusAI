# Team ALPHA
## Team Members
- Sujith Pedapati
- Hanuma Sathvik Mandalapu
- Yadidhya Gandham

## College Name - IIITDM Kurnool
# 🚀 HR Automation Agent - Complete HR System

A sophisticated, enterprise-grade HR automation platform with 6 integrated phases for candidate screening, interview scheduling, leave management, and escalation detection.

**Status:** ✅ Production Ready | **Performance:** 5-8x faster with multithreading | **Documentation:** 55+ pages

---

## 📁 Project Organization

This project uses a clean, organized folder structure for easy navigation:

```
HR Automation Agent/
├── README.md                     ← You are here
├── FOLDER_STRUCTURE.txt          ← Detailed structure guide
│
├── 📁 docs/                      14 comprehensive guides
│   ├── ARCHITECTURE_OVERVIEW.md  (System architecture)
│   ├── QUICK_START.md            (Get started in 5 min)
│   ├── MULTITHREADING_GUIDE.md   (Performance optimization)
│   ├── INTEGRATION_EXAMPLES.md   (Code examples per phase)
│   └── ... 10 more guides
│
├── 📁 src/                       All source code modules
│   ├── interview_engine/         (Phase 1: Candidate screening)
│   ├── phase3_scheduling/        (Phase 3: Interview scheduling)
│   ├── phase4_leave/             (Phase 4: Leave management)
│   ├── phase6_escalation/        (Phase 6: Escalation detection)
│   ├── ui/                       (Streamlit dashboard)
│   ├── utils/                    (Shared utilities & threading)
│   └── resume_extractor/         (Resume processing)
│
├── 📁 config/                    Configuration & dependencies
│   ├── config.py
│   ├── config_dashboard.py
│   ├── requirements.txt
│   └── requirements_ui.txt
│
├── 📁 scripts/                   Entry points
│   ├── main.py                   (Main orchestrator)
│   ├── pipeline.py               (Pipeline controller)
│   └── run_dashboard.py          (Start Streamlit dashboard)
│
├── 📁 output/                    Generated results
│   └── results.json
│
└── 📁 logs/                      Application logs
```

## 🎯 Quick Navigation

| What You Want | Where to Go |
|---------------|-------------|
| **Get Started** | Read: [docs/QUICK_START.md](docs/QUICK_START.md) or run: `python scripts/main.py` |
| **System Architecture** | Read: [docs/ARCHITECTURE_OVERVIEW.md](docs/ARCHITECTURE_OVERVIEW.md) |
| **Start Dashboard** | Run: `python scripts/run_dashboard.py` |
| **Performance Optimization** | Read: [docs/MULTITHREADING_GUIDE.md](docs/MULTITHREADING_GUIDE.md) |
| **Code Examples** | Read: [docs/INTEGRATION_EXAMPLES.md](docs/INTEGRATION_EXAMPLES.md) |
| **Winning Features** | Read: [docs/WINNING_SUBMISSION.md](docs/WINNING_SUBMISSION.md) |

---

## 🏗️ System Architecture Overview

```
Resume PDF/DOCX
      │
      ▼
├─ Text Extraction (pdfplumber / python-docx)  [pdf_extractor.py]
├─ OCR (Tesseract for scanned PDFs)            [ocr_engine.py]
      │
      ▼
├─ LLM: Phi-3-mini-4k-instruct                 [docs_extractor.py]
│   └─ Structured JSON extraction
│       {skills[], experience[], education[], total_years_exp}
│
├─ NER: GLiNER (Named Entity Recognition)       [docs_extractor.py]
│   └─ Extract: skills, titles, companies, degrees
│
      │
      ▼
├─ Embeddings: BGE-Large (1024-dim)             [pipeline.py]
│   └─ Semantic representation of resumes
│
├─ ANN Index: FAISS L2                          [pipeline.py]
│   └─ Fast top-K retrieval
│
├─ Reranking: BGE-Reranker-Large               [pipeline.py]
│   └─ Cross-encoder pairwise scoring
│
      │
      ▼
├─ Final Score (Weighted Combination):          [pipeline.py]
│   ├─ 0.35 × Reranker Score (semantic match)
│   ├─ 0.30 × Skill Overlap (exact keywords)
│   ├─ 0.20 × Experience Match (years)
│   └─ 0.15 × Title Similarity (job title)
│
      ▼
Ranked Resume List (with explainable scores)
```

## File Distribution

### 1. **pdf_extractor.py** - Text Extraction
- `PDFExtractor.extract_from_pdf()` - Uses pdfplumber for clean text extraction
- `PDFExtractor.extract_from_docx()` - Uses python-docx for Word documents
- `PDFExtractor.extract_text()` - Unified interface

**Dependencies:** pdfplumber, python-docx

### 2. **ocr_engine.py** - OCR for Scanned PDFs
- `OCREngine.extract_from_image()` - Tesseract OCR on images
- `OCREngine.extract_from_pdf_page()` - OCR for scanned PDF pages
- Optional preprocessing for better accuracy

**Dependencies:** pytesseract, opencv-python, Pillow

### 3. **docs_extractor.py** - Structured Data Extraction
**LLM Extraction:**
- `DocsExtractor.extract_with_llm()` - Phi-3 JSON extraction
- Instruction-following LLM for structured output
- Cost-effective: 3.8B parameters, runs on 8GB RAM

**NER Extraction:**
- `DocsExtractor.extract_with_ner()` - GLiNER entity recognition
- Extracts: skills, job titles, companies, degrees
- Complements LLM with entity-level precision

**Combined Extraction:**
- `DocsExtractor.extract()` - Merges LLM + NER results
- Deduplicates and combines both approaches

**Dependencies:** transformers, torch, gliner

### 4. **parser.py** - Pipeline Orchestration
- `ResumeParser.parse_file()` - Single resume → structured data
- `ResumeParser.parse_folder()` - Batch process all resumes
- Handles errors gracefully with logging

**Dependencies:** pdf_extractor, ocr_engine, docs_extractor

### 5. **pipeline.py** - Matching & Ranking Engine
**Embeddings:**
- `ResumeMatcher.encode_text()` - BGE-Large semantic embeddings
- 1024-dimensional vectors for similarity matching

**ANN Indexing:**
- `ResumeMatcher.index_resumes()` - Build FAISS L2 index
- O(log n) retrieval time for large datasets

**Reranking:**
- `ResumeMatcher.rerank_candidates()` - Cross-encoder scoring
- Pairwise comparisons for precise matching

**Scoring Functions:**
- `calculate_skill_overlap()` - Keyword-based skill matching
- `calculate_experience_match()` - Years of experience comparison
- `calculate_title_similarity()` - Job title embedding similarity

**Final Matching:**
- `match_job_description()` - End-to-end pipeline
- Returns ranked list with explainable component scores

**Dependencies:** sentence-transformers, faiss-cpu, scikit-learn, numpy

## Recommended Models

| Component | Model | Size | Notes |
|-----------|-------|------|-------|
| Text-to-JSON | microsoft/Phi-3-mini-4k-instruct | 3.8B | Excellent instruction following, 8GB RAM |
| NER | urchade/gliner_base | ~300M | Fast entity recognition |
| Embeddings | BAAI/bge-large-en-v1.5 | 335M | 1024-dim, semantic text understanding |
| Reranking | BAAI/bge-reranker-large | 355M | Cross-encoder for precise matching |

## GPU/CPU Requirements

**Minimum (CPU-only, ~8GB RAM):**
- Phi-3-mini + BGE-Large + GLiNER

**Recommended (GPU, ~14GB VRAM):**
- All models + batch processing

**Optional GPU Acceleration:**
```python
import torch
# Automatically uses GPU if available
device = 0 if torch.cuda.is_available() else -1
```

## Installation

```bash
pip install -r requirements.txt

# For Tesseract OCR (Windows)
# Download and install from: https://github.com/UB-Mannheim/tesseract/wiki

# For GPU support (optional)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install faiss-gpu
```

## Usage

### Basic Usage
```python
from resume_extractor import ResumeParser
from pipeline import ResumeMatcher

# Parse resumes
parser = ResumeParser()
resumes = parser.parse_folder("path/to/resumes")

# Initialize matcher
matcher = ResumeMatcher()
matcher.index_resumes(resumes)

# Match job description
job_description = "We're hiring a Python engineer with AWS experience..."
results = matcher.match_job_description(job_description)

# Results include:
# - Ranked candidates
# - Component scores (reranker, skill_overlap, experience, title_similarity)
# - Final weighted score
```

### Advanced: Custom Weights
```python
# Modify scoring weights in pipeline.py:
# final_score = (
#     0.40 * reranker_score +      # ← Adjust
#     0.25 * skill_overlap +       # ← Adjust
#     0.20 * experience_match +    # ← Adjust
#     0.15 * title_similarity      # ← Adjust
# )
```

## Performance

- **Text Extraction:** ~1-2s per resume (PDF/DOCX)
- **LLM Extraction:** ~2-5s per resume (Phi-3-mini on CPU)
- **NER Extraction:** ~0.5-1s per resume (GLiNER)
- **Indexing:** ~0.1s per resume (FAISS)
- **Top-K Retrieval:** ~0.01-0.1s (O(log n))
- **Reranking:** ~0.1-0.5s per candidate

**Total for 100 resumes:** ~5-10 minutes (CPU) or ~2-3 minutes (GPU)

## Error Handling

- Missing LLM model → fallback to NER only
- Missing Reranker → fallback to embedding similarity
- Missing FAISS → fallback to brute-force cosine similarity
- Unsupported file formats → warning + skip

All errors are logged with context for debugging.

## Future Enhancements

- [ ] Multi-language support (Gemma-2, Qwen multilingual)
- [ ] Custom fine-tuned models per industry
- [ ] Real-time incremental indexing
- [ ] Resume-to-Resume similarity (diversity)
- [ ] Explainability dashboard (component breakdown)
- [ ] Skill progression analysis (career path)
- [ ] Integration with ATS (Applicant Tracking System)

=======
# NETRIK_HACKATHON
>>>>>>> ae43fb0d918666fbc24943503397df51ce5c8e47
