"""
Smoke test for the Day 2 ingestion pipeline.

Run from project root:
    python -m backend.ingestion.test_ingestion <path_to_file>

Supports: .pdf  .docx  .md  .txt

Output is saved to:  output/ingestion_result.json
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")


def run_test(file_path: str) -> None:
    from backend.ingestion.paper_ingestion import ingest_paper

    path = Path(file_path)
    if not path.exists():
        print(f"[Error] File not found: {path}")
        sys.exit(1)

    print(f"\n[Test] Ingesting: {path.name}\n")
    doc = ingest_paper(path)

    # ── Terminal summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("INGESTION RESULT")
    print("=" * 55)
    print(f"  paper_id  : {doc.paper_id}")
    print(f"  title     : {doc.metadata.title}")
    print(f"  doi       : {doc.metadata.doi}")
    print(f"  year      : {doc.metadata.year}")
    print(f"  pages     : {doc.metadata.page_count}")
    print(f"  file_type : {doc.metadata.file_type}")
    print(f"  abstract  : {(doc.metadata.abstract or '')[:120]}...")
    print(f"  text_len  : {len(doc.full_text):,} chars")
    print("=" * 55)

    # ── Save full result to JSON ───────────────────────────────────────────────
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "ingestion_result.json"

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(doc.model_dump(), f, indent=2, default=str)

    print(f"\n[OK] Full result saved to: {out_file}")
    print(" -> Open it to verify cleaned text, metadata, and per-page content.\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m backend.ingestion.test_ingestion <path_to_file>")
        print("Example: python -m backend.ingestion.test_ingestion data/paper.pdf")
        sys.exit(1)

    run_test(sys.argv[1])