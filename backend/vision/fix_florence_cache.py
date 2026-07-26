"""
AstroNexus AI — Florence-2 Surgical Cache Patch

Fixes Florence2LanguageConfig.__init__ in the cached remote code file
by adding missing attributes BEFORE they are first accessed.

This matches Microsoft's latest fix:
    Before: self.forced_bos_token_id  (raises AttributeError)
    After:  getattr(self, "forced_bos_token_id", None)

Run once from project root:
    python -m backend.vision.patch_florence2
"""
from __future__ import annotations

import re
import sys
from pathlib import Path


MISSING_ATTRS = [
    "forced_bos_token_id",
    "forced_eos_token_id",
    "suppress_tokens",
    "begin_suppress_tokens",
]


def find_cached_file() -> Path:
    """Locate configuration_florence2.py in HuggingFace module cache."""
    import os

    search_roots = [
        Path.home() / ".cache" / "huggingface" / "modules",
        Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "modules",
        Path(os.environ.get("TRANSFORMERS_CACHE", "")) / "modules",
    ]

    for root in search_roots:
        if not root.exists():
            continue
        for found in root.rglob("configuration_florence2.py"):
            content = found.read_text(encoding="utf-8")
            if "Florence2LanguageConfig" in content:
                return found

    raise FileNotFoundError(
        "Could not find configuration_florence2.py in HuggingFace cache.\n"
        "Run this first to populate the cache:\n"
        "  python -c \"from transformers import AutoProcessor; "
        "AutoProcessor.from_pretrained('microsoft/Florence-2-base', "
        "trust_remote_code=True)\"\n"
        "Then re-run this script."
    )


def show_class_init(lines: list[str], class_name: str, n: int = 30) -> None:
    """Print the __init__ of a class for inspection."""
    in_class = False
    in_init  = False
    printed  = 0
    for i, line in enumerate(lines, 1):
        if f"class {class_name}" in line:
            in_class = True
        if in_class and "def __init__" in line:
            in_init = True
        if in_init:
            print(f"  {i:5d}: {line}")
            printed += 1
            if printed >= n:
                break


def patch_file(path: Path) -> bool:
    """
    Apply surgical patch to Florence2LanguageConfig.__init__.

    Strategy: find every occurrence of bare `self.<attr>` where <attr>
    is one of the missing attributes and replace with
    `getattr(self, "<attr>", None)`.

    Also insert the attributes as defaults at the TOP of __init__
    before any super().__init__() call, matching Microsoft's latest fix.
    """
    original = path.read_text(encoding="utf-8")

    if "# ASTRONEXUS_PATCH_V2" in original:
        print(f"  Already patched (V2): {path}")
        return True

    lines   = original.split("\n")
    patched = list(lines)

    # ── Find Florence2LanguageConfig.__init__ ─────────────────────────────────
    class_start = None
    init_start  = None
    init_indent = None

    for i, line in enumerate(lines):
        if "class Florence2LanguageConfig" in line:
            class_start = i
        if class_start is not None and "def __init__" in line and init_start is None:
            init_start  = i
            init_indent = len(line) - len(line.lstrip())

    if init_start is None:
        print("  ERROR: Could not find Florence2LanguageConfig.__init__")
        return False

    print(f"\n  Found Florence2LanguageConfig at line {class_start + 1}")
    print(f"  Found __init__ at line {init_start + 1}")

    # ── Find the first line of the __init__ body ───────────────────────────────
    # Body starts after the def line (may span multiple lines for long signatures)
    body_start = init_start + 1
    for i in range(init_start + 1, len(lines)):
        stripped = lines[i].strip()
        if stripped == "":
            continue
        # Skip continuation lines of the def signature
        if stripped.startswith("self,") or stripped.startswith("**") or \
           stripped.endswith(",") or stripped.endswith("\\"):
            continue
        # Check if we're past the def signature (indented more than def)
        line_indent = len(lines[i]) - len(lines[i].lstrip())
        if line_indent > init_indent and stripped != "":
            body_start = i
            break

    print(f"  __init__ body starts at line {body_start + 1}: {lines[body_start].strip()[:60]}")

    # ── Build the attribute initialisation block ───────────────────────────────
    body_indent = " " * (init_indent + 4)
    insert_lines = [
        f"{body_indent}# ASTRONEXUS_PATCH_V2: initialise attrs before super().__init__()",
    ]
    for attr in MISSING_ATTRS:
        insert_lines.append(
            f'{body_indent}if not hasattr(self, "{attr}"):'
            f' self.{attr} = kwargs.pop("{attr}", None)'
        )
    insert_lines.append("")   # blank line separator

    # ── Step 2: replace bare `self.<attr>` accesses with getattr ──────────────
    # Only within the __init__ body of Florence2LanguageConfig
    # Find where __init__ ends (next def at same or lower indent)
    init_end = len(lines)
    for i in range(init_start + 1, len(lines)):
        if lines[i].strip() == "":
            continue
        line_indent = len(lines[i]) - len(lines[i].lstrip())
        if line_indent <= init_indent and lines[i].strip().startswith("def "):
            init_end = i
            break

    print(f"  __init__ ends at line {init_end}")

    # Replace `self.forced_bos_token_id` → `getattr(self, "forced_bos_token_id", None)`
    # but NOT in assignment context (left-hand side is fine to keep)
    replacements = 0
    for i in range(body_start, init_end):
        line = patched[i]
        for attr in MISSING_ATTRS:
            pattern = rf'(?<!=)\bself\.{attr}\b(?!\s*=)'
            replacement = f'getattr(self, "{attr}", None)'
            new_line, n = re.subn(pattern, replacement, line)
            if n > 0:
                patched[i] = new_line
                replacements += n
                print(f"  Line {i+1}: replaced self.{attr} → getattr(...)")

    print(f"  Total getattr replacements: {replacements}")

    # ── Insert the initialisation block at top of body ─────────────────────────
    for j, ins_line in enumerate(insert_lines):
        patched.insert(body_start + j, ins_line)

    # ── Write patched file ─────────────────────────────────────────────────────
    path.write_text("\n".join(patched), encoding="utf-8")

    # Clear compiled bytecode
    pycache = path.parent / "__pycache__"
    if pycache.exists():
        for pyc in pycache.glob("configuration_florence2*.pyc"):
            pyc.unlink()
            print(f"  Deleted pyc: {pyc.name}")

    print(f"\n  ✓ Patched file written: {path}")
    return True


def verify(path: Path) -> bool:
    """Re-read the patched file and confirm the patch markers are present."""
    content = path.read_text(encoding="utf-8")
    ok = "ASTRONEXUS_PATCH_V2" in content
    print(f"  Patch marker present: {ok}")
    return ok


def clear_module_cache() -> None:
    """Remove Florence-2 modules from sys.modules so Python reimports them."""
    to_remove = [
        k for k in sys.modules
        if "florence" in k.lower() or "transformers_modules" in k.lower()
    ]
    for k in to_remove:
        del sys.modules[k]
    if to_remove:
        print(f"  Cleared {len(to_remove)} cached module(s)")


def test_import() -> bool:
    """Verify that AutoProcessor loads without AttributeError."""
    clear_module_cache()
    try:
        from transformers import AutoProcessor
        AutoProcessor.from_pretrained(
            "microsoft/Florence-2-base",
            trust_remote_code=True,
        )
        print("  ✓ AutoProcessor loaded successfully")
        return True
    except AttributeError as e:
        print(f"  ✗ AttributeError still occurs: {e}")
        return False
    except Exception as e:
        print(f"  ✗ Other error (not our fix): {type(e).__name__}: {e}")
        return False


def run() -> None:
    import transformers
    print("\n" + "="*60)
    print("FLORENCE-2 SURGICAL PATCH  (V2)")
    print("="*60)
    print(f"  transformers : {transformers.__version__}")

    # ── 1. Find file ───────────────────────────────────────────────────────────
    print("\n[1] Locating cached configuration_florence2.py...")
    try:
        path = find_cached_file()
        print(f"  Found: {path}")
    except FileNotFoundError as e:
        print(f"  {e}")
        sys.exit(1)

    # ── 2. Show current state ──────────────────────────────────────────────────
    print("\n[2] Current __init__ (first 25 lines of body):")
    lines = path.read_text(encoding="utf-8").split("\n")
    show_class_init(lines, "Florence2LanguageConfig", n=25)

    # ── 3. Apply patch ─────────────────────────────────────────────────────────
    print("\n[3] Applying patch...")
    ok = patch_file(path)
    if not ok:
        print("  Patch failed.")
        sys.exit(1)

    # ── 4. Verify patch on disk ────────────────────────────────────────────────
    print("\n[4] Verifying patch on disk...")
    ok = verify(path)
    if not ok:
        print("  Patch marker not found — something went wrong.")
        sys.exit(1)

    # ── 5. Test import ─────────────────────────────────────────────────────────
    print("\n[5] Testing AutoProcessor import...")
    ok = test_import()

    print("\n" + "="*60)
    if ok:
        print("✓ PATCH SUCCESSFUL")
        print("  Run: python -m backend.vision.satellite_pipeline <img> --no-sam2")
    else:
        print("✗ PATCH DID NOT RESOLVE THE ISSUE")
        print("  Show the output above to diagnose the remaining problem.")
    print("="*60 + "\n")


if __name__ == "__main__":
    run()