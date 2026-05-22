"""
High-level orchestrator: text + graphics + reference rewrite, all in one call.
"""

from __future__ import annotations

import base64
import gzip
import html
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
    r'(<(?:[A-Za-z_][\w\-]*:)?ImportObFile\b[^>]*>)'
    r'([^<]+)'
    r'(</(?:[A-Za-z_][\w\-]*:)?ImportObFile>)',
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
    for bn, np in filename_to_new.items():
        log_fn(f"  • {bn}  →  {np}")

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
        
        # THE FIX: Decode FrameMaker's internal <u> and <c> tags to correctly extract the basename
        decoded = html.unescape(current.strip())
        decoded = decoded.replace("<u>", "/").replace("<c>", "/")
        decoded = decoded.replace("\\", "/").replace(":", "/")
        parts = [p for p in decoded.split("/") if p]
        bn_current = parts[-1] if parts else decoded
        
        for bn, new_path in filename_to_new.items():
            if bn_current == bn:
                log_fn(f"  ✓ {current!r} → {new_path!r}  (matched {bn!r})")
                rewrite_count += 1
                
                # Replace the old <c> encoded path with our strictly formatted relative path
                return f"{head}{new_path}{tail}"
        
        if len(miss_samples) < 10:
            miss_samples.append(current)
        return match.group(0)

    new_mif = _OB_RE.sub(_replace, mif_str)
    log_fn(
        f"Rewrote {rewrite_count} <ImportObFile> reference(s) in MIF blob",
        "info" if rewrite_count else "warning",
    )

    if rewrite_count == 0:
        log_fn("  No rewrites fired — dumping <ImportObFile> samples:", "warning")
        for i, m in enumerate(_OB_RE.finditer(mif_str)):
            if i >= 10:
                break
            log_fn(f"    [{i}] {m.group(2)!r}", "warning")
        log_fn("  Available basenames:", "warning")
        for bn in filename_to_new:
            log_fn(f"    {bn!r}", "warning")
        return 0

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
    log_fn(f"[{target_lang}] text done → {out_xlf_path.relative_to(work_dir)}")

    if graphics_source_dir is None:
        log_fn(f"[{target_lang}] no graphics folder supplied — skipping image OCR")
        return deliverable_root

    graphics_source_dir = Path(graphics_source_dir)
    if not graphics_source_dir.is_dir():
        log_fn(f"[{target_lang}] graphics_source_dir not a directory: {graphics_source_dir} — skipping image OCR", "warning")
        return deliverable_root

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
        log_fn(f"[{target_lang}] no graphics translated (either no <ImportObFileDI> entries or no matching files found)", "warning")
        return deliverable_root

    log_fn(f"[{target_lang}] rewriting MIF references in translated XLF…")
    n = update_xlf_references(out_xlf_path, mapping, on_log=on_log)
    log_fn(f"[{target_lang}] graphics done — {n} reference(s) rewritten")

    return deliverable_root
