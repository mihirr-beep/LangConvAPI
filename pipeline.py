"""
High-level orchestrator: text + graphics + reference rewrite, all in one call.

This is the API-shaped counterpart to translate_xliff_openai_2.translate_file()
from the standalone CLI: same logic, no argparse, takes bytes in and returns
the path of the assembled deliverable folder on disk.

What it does, per call (one input XLF, one target language)
───────────────────────────────────────────────────────────
1. Run the text-only translator (translator.translate_xliff_bytes) on the
   uploaded XLF bytes — produces a translated .xlf with <target> populated
   and FrameMaker-18-safe <source>-clone targets.

2. Build the strict deliverable layout under `work_dir`:

       <work_dir>/translated_<lang>/                ← root that gets zipped
           graphics/graphics/<subfolder>/<file>     ← translated assets
           translated_<lang>/<stem>_<lang>.xlf      ← translated XLF (nested)

   The XLF lives in `<root>/translated_<lang>/`, the graphics under
   `<root>/graphics/graphics/`, so the relative MIF reference from XLF to
   any graphic is `../graphics/graphics/<subfolder>/<file>`.

3. If a graphics source folder was supplied, call
   image_ocr_translator.process_xlf_references — that reads the embedded
   MIF blob, locates each <ImportObFileDI> file inside the uploaded
   Graphics folder, OCR-translates images and PDFs, writes them under
   `graphics/graphics/<subfolder>/`, and returns the mapping ready for
   the rewriter.

4. Run update_xlf_references on the translated XLF so every <ImportObFile>
   in its embedded MIF points at the new relative path. <ImportObFileDI>
   is left intact (it's the device-independent ORIGINAL — never rewritten).

If no graphics folder was supplied, steps 3 and 4 are skipped — the
caller receives a deliverable folder containing only the translated XLF
under `<root>/translated_<lang>/`, with the original MIF refs untouched.
"""

# pipeline.py
"""
High-level orchestrator: text + graphics + reference rewrite, all in one call.
"""

from __future__ import annotations

import base64
import gzip
import logging
import re
from pathlib import Path
from typing import Callable, Dict, Optional

from lxml import etree

from translator import translate_xliff_bytes
from image_ocr_translator import (
    MEDIA_EXTENSIONS,
    process_xlf_references,
)

log = logging.getLogger(__name__)

_OB_RE = re.compile(
    r"(<(?:[A-Za-z_][\w\-]*:)?ImportObFile\b[^>]*>)"
    r"([^<]+)"
    r"(</?:[A-Za-z_][\w\-]*:)?ImportObFile>",
    re.IGNORECASE,
)

def update_xlf_references(
    xlf_path: Path,
    path_mapping: Dict[str, str],
    on_log: Optional[Callable[[str, str], None]] = None,
) -> int:
    log_fn = on_log or (lambda msg, level="info": getattr(log, level, log.info)(msg))

    if not path_mapping:
        log_fn("update_xlf_references: empty mapping; nothing to do", "warning")
        return 0

    filename_to_new: Dict[str, str] = {}
    for new_path in set(path_mapping.values()):
        bn = Path(new_path.replace("\\", "/")).name
        filename_to_new[bn] = new_path

    log_fn(f"Rewrite plan: {len(filename_to_new)} filename(s)")

    parser = etree.XMLParser(remove_blank_text=False, recover=True)
    tree   = etree.parse(str(xlf_path), parser)
    root   = tree.getroot()

    internal_el = None
    for elem in root.iter():
        if elem.tag.split("}")[-1] == "internal-file":
            internal_el = elem
            break

    if internal_el is None or not (internal_el.text and internal_el.text.strip()):
        log_fn("No <internal-file> element in XLF — nothing to rewrite", "warning")
        return 0

    try:
        raw_b64    = internal_el.text.strip()
        compressed = base64.b64decode(raw_b64)
        was_gzip   = compressed[:2] == b"\x1f\x8b"
        mif_str = (
            gzip.decompress(compressed).decode("utf-8", errors="replace")
            if was_gzip
            else compressed.decode("utf-8", errors="replace")
        )
    except Exception as e:
        log_fn(f"Failed to decode <internal-file>: {e}", "error")
        return 0

    rewrite_count = 0
    miss_samples: list = []

    def _replace(match: re.Match) -> str:
        nonlocal rewrite_count
        head, current, tail = match.group(1), match.group(2), match.group(3)
        
        # Normalize target file references safely
        norm = current.replace("\\", "/").replace(":", "/")
        norm_bn = norm.split("/")[-1] if "/" in norm else norm

        for bn, new_path in filename_to_new.items():
            if norm_bn == bn or norm.endswith("/" + bn):
                log_fn(f"  ✓ {current!r} → {new_path!r}")
                rewrite_count += 1
                return f"{head}{new_path}{tail}"
        if len(miss_samples) < 10:
            miss_samples.append(current)
        return match.group(0)

    new_mif = _OB_RE.sub(_replace, mif_str)
    log_fn(f"Rewrote {rewrite_count} <ImportObFile> reference(s) in MIF blob")

    raw = new_mif.encode("utf-8", errors="replace")
    if was_gzip:
        raw = gzip.compress(raw)
    internal_el.text = base64.b64encode(raw).decode("ascii")

    tree.write(str(xlf_path), encoding="utf-8", xml_declaration=True)
    return rewrite_count

def translate_project(
    xlf_bytes: bytes,
    xlf_filename: str,
    target_lang: str,
    work_dir: Path,
    graphics_source_dir: Optional[Path] = None,
    on_log: Optional[Callable[[str, str], None]] = None,
) -> Path:
    log_fn = on_log or (lambda msg, level="info": getattr(log, level, log.info)(msg))

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    deliverable_root = work_dir / f"translated_{target_lang}"
    xlf_out_dir      = deliverable_root / f"translated_{target_lang}"
    graphics_out_dir = deliverable_root / "graphics" / "graphics"
    
    xlf_out_dir.mkdir(parents=True, exist_ok=True)
    if graphics_source_dir is not None:
        graphics_out_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(xlf_filename).stem.rstrip(".") or "translated"
    ext  = Path(xlf_filename).suffix or ".xlf"
    if ext.lower() not in {".xlf", ".xliff"}:
        ext = ".xlf"

    log_fn(f"[{target_lang}] translating text segments…")
    translated_bytes = translate_xliff_bytes(xlf_bytes, target_lang=target_lang)

    out_xlf_path = xlf_out_dir / f"{stem}_{target_lang}{ext}"
    out_xlf_path.write_bytes(translated_bytes)
    log_fn(f"[{target_lang}] text done")

    if graphics_source_dir is None:
        log_fn(f"[{target_lang}] no graphics folder supplied — skipping image OCR")
        return deliverable_root

    graphics_source_dir = Path(graphics_source_dir)
    log_fn(f"[{target_lang}] processing graphics from {graphics_source_dir}…")
    
    mapping = process_xlf_references(
        xlf_path=out_xlf_path,
        target_lang=target_lang,
        out_folder=graphics_out_dir,
        rename_with_lang=False,
        out_xlf_path=out_xlf_path,
        src_graphics_folder=graphics_source_dir,
    )

    if not mapping:
        log_fn(f"[{target_lang}] no graphics translated", "warning")
        return deliverable_root

    log_fn(f"[{target_lang}] rewriting MIF references in translated XLF…")
    n = update_xlf_references(out_xlf_path, mapping, on_log=on_log)
    log_fn(f"[{target_lang}] graphics done — {n} reference(s) rewritten")

    return deliverable_root
