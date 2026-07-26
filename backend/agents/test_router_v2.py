"""
Router v2 validation — run this before the full benchmark.

Tests all the specific failure modes identified in the benchmark results.

Run: python -m backend.agents.test_router_v2
"""
from __future__ import annotations

import sys
import logging
logging.basicConfig(level=logging.WARNING, format="%(message)s")

from backend.agents.router import classify_query

TESTS = [

    # ── Research Queries ───────────────────────────────────────────────
    ("What assumptions does the paper make?",                    {"metadata":{"paper_loaded":True}}, "research"),
    ("Summarize the methodology section.",                       {"metadata":{"paper_loaded":True}}, "research"),
    ("What optimizer was used during training?",                 {"metadata":{"paper_loaded":True}}, "research"),
    ("How many experiments were conducted?",                     {"metadata":{"paper_loaded":True}}, "research"),
    ("Did the authors release their source code?",               {"metadata":{"paper_loaded":True}}, "research"),
    ("Which benchmark datasets were evaluated?",                 {"metadata":{"paper_loaded":True}}, "research"),
    ("What preprocessing steps were applied?",                   {"metadata":{"paper_loaded":True}}, "research"),
    ("Explain the architecture proposed in the paper.",          {"metadata":{"paper_loaded":True}}, "research"),
    ("What are the future work suggestions?",                    {"metadata":{"paper_loaded":True}}, "research"),
    ("Does the paper discuss computational complexity?",         {"metadata":{"paper_loaded":True}}, "research"),
    ("What evaluation metrics were reported?",                   {"metadata":{"paper_loaded":True}}, "research"),
    ("Which baseline performed the best?",                       {"metadata":{"paper_loaded":True}}, "research"),
    ("How does the proposed approach differ from previous work?",{"metadata":{"paper_loaded":True}}, "research"),
    ("What are the major findings?",                             {"metadata":{"paper_loaded":True}}, "research"),
    ("Give me a one paragraph summary.",                         {"metadata":{"paper_loaded":True}}, "research"),

    # ── Graph Queries ─────────────────────────────────────────────────
    ("Which papers use Vision Transformers?",                    {}, "graph"),
    ("List all satellites with optical sensors.",               {}, "graph"),
    ("Which journals publish remote sensing papers?",           {}, "graph"),
    ("Find datasets related to wildfire detection.",            {}, "graph"),
    ("Which institutions collaborate with ESA?",                {}, "graph"),
    ("Show algorithms related to image classification.",        {}, "graph"),
    ("Which conferences published segmentation papers?",        {}, "graph"),
    ("List all remote sensing keywords.",                       {}, "graph"),
    ("Find papers related to NDVI.",                            {}, "graph"),
    ("Which satellites monitor agriculture?",                   {}, "graph"),
    ("Who developed the U-Net architecture?",                   {}, "graph"),
    ("Show all missions operated by ESA.",                      {}, "graph"),
    ("Which papers mention DINOv2?",                            {}, "graph"),
    ("Find datasets used for flood detection.",                 {}, "graph"),
    ("Which authors published multiple papers?",                {}, "graph"),

    # ── Satellite Image Queries ───────────────────────────────────────
    ("Estimate vegetation density.",                            {"metadata":{"image_path":"sample.png"}}, "satellite"),
    ("Detect agricultural fields.",                             {"metadata":{"image_path":"sample.png"}}, "satellite"),
    ("Identify possible wildfire regions.",                     {"metadata":{"image_path":"sample.png"}}, "satellite"),
    ("Locate roads and highways.",                              {"metadata":{"image_path":"sample.png"}}, "satellite"),
    ("Estimate urban building density.",                        {"metadata":{"image_path":"sample.png"}}, "satellite"),
    ("Identify coastal regions.",                               {"metadata":{"image_path":"sample.png"}}, "satellite"),
    ("Detect cloud cover percentage.",                          {"metadata":{"image_path":"sample.png"}}, "satellite"),
    ("Highlight water bodies.",                                 {"metadata":{"image_path":"sample.png"}}, "satellite"),
    ("Describe this satellite image.",                          {"metadata":{"image_path":"sample.png"}}, "satellite"),
    ("Is there evidence of erosion?",                           {"metadata":{"image_path":"sample.png"}}, "satellite"),

    # ── General Knowledge ─────────────────────────────────────────────
    ("What is machine learning?",                               {}, "general"),
    ("Explain Newton's laws of motion.",                        {}, "general"),
    ("Who invented the telephone?",                             {}, "general"),
    ("Why is the sky blue?",                                    {}, "general"),
    ("What is the capital of Australia?",                       {}, "general"),
    ("Explain blockchain technology.",                          {}, "general"),
    ("How does Wi-Fi work?",                                    {}, "general"),
    ("What is the tallest mountain in the world?",              {}, "general"),
    ("What causes earthquakes?",                                {}, "general"),
    ("Who discovered penicillin?",                              {}, "general"),
]

def run():
    passed = 0
    failed = []

    print(f"\n{'='*65}")
    print("ASTRONEXUS ROUTER v2 — VALIDATION")
    print(f"{'='*65}")

    for query, state, expected in TESTS:
        got = classify_query(query, state)
        ok  = got == expected
        if ok:
            passed += 1
        else:
            failed.append((query, expected, got))
        icon = "[OK]" if ok else "[FAIL]"
        print(f"  {icon}  {query[:48]:<50}  {got:<12}  (expected {expected})")

    total = len(TESTS)
    print(f"\n{'='*65}")
    print(f"  {passed}/{total} passed  ({100*passed//total}%)")

    if failed:
        print(f"\n  FAILURES:")
        for q, exp, got in failed:
            print(f"    '{q[:55]}' -> got={got}, expected={exp}")

    print(f"{'='*65}\n")
    return passed == total


if __name__ == "__main__":
    ok = run()
    sys.exit(0 if ok else 1)