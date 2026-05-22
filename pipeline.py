"""
High-level orchestrator: text + graphics + reference rewrite, all in one call.
"""

from __future__ import annotations

import base64
import gzip
import html
import logging
import re
import urllib.parse
from pathlib import Path
from typing import Callable, Dict, Optional

from lxml import etree

from translator import translate_xliff_bytes
from image_ocr_translator import (
    MEDIA_EXTENSIONS,
    process_xlf_references,
)

log = logging.getLogger(__name__)

def _log_fn(msg: str, level: str = "info", on_log: Optional[Callable] = None):
    if on_log:
        on_log(msg, level)
    else:
        getattr(log, level, log.info)(msg)

# THE FIX: Match both regular and 'DI' (Device Independent) FrameMaker reference tags
_OB_RE = re.compile(
    r'(<(?:[A-Za-z_][\w\-]*:)?ImportObFile(?:DI)?\b[^>]*>)'
    r'([^<]+)'
    r'(</(?:[A-Za-z_][\w\-]*:)?ImportObFile(?:DI)?>)',
    re.IGNORECASE,
)

def update_xlf_references(
    xlf_path: Path,
    path_mapping: Dict[str, str],
    on_log: Optional[Callable[[str, str], None]] = None,
) -> int:
    def local_log(msg, level="info"): return _log_fn(msg, level, on_log)

    if not path_mapping:
        local_log("update_xlf_references: empty mapping; nothing to do", "warning")
        return 0

    filename_to_new: Dict[str, str] = {}
    for old_key, new_path in path_mapping.items():
        # THE FIX: Double unescape to handle FrameMaker's &amp;auml; -> &auml; -> ä translations
        decoded_key = html.unescape(old_key.strip())
        decoded_key = html.unescape(decoded_key)
        decoded_key = urllib.parse.unquote(decoded_key)
        decoded_key = decoded_key.replace("<u>", "/").replace("<c>", "/")
        bn = Path(decoded_key.replace("\\", "/")).name
        filename_to_new[bn] = new_path

    local_log(f"Rewrite plan: {len(filename_to_new)} filename(s)")

    parser = etree.XMLParser(remove_blank_text=False, recover=True)
    tree   = etree.parse(str(xlf_path), parser)
    
    internal_el = None
    for elem in tree.getroot().iter():
        if elem.tag.split("}")[-1] == "internal-file":
            internal_el = elem
            break

    if internal_el is None or not (internal_el.text and internal_el.text.strip()):
        local_log("No <internal-file> element in XLF — nothing to rewrite", "warning")
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
        local_log(f"Failed to decode <internal-file>: {e}", "error")
        return 0

    rewrite_count = 0
    miss_samples: list = []

    def _replace(match: re.Match) -> str:
        nonlocal rewrite_count
        head, current, tail = match.group(1), match.group(2), match.group(3)
        
        # Unescape current string to find the true baseline file name
        decoded = html.unescape(current.strip())
        decoded = html.unescape(decoded)
        decoded = urllib.parse.unquote(decoded)
        decoded = decoded.replace("<u>", "/").replace("<c>", "/")
        decoded = decoded.replace("\\", "/").replace(":", "/")
        parts = [p for p in decoded.split("/") if p]
        bn_current = parts[-1] if parts else decoded
        
        for bn, new_path in filename_to_new.items():
            if bn_current.lower() == bn.lower():
                rewrite_count += 1
                return f"{head}{new_path}{tail}"
        
        if len(miss_samples) < 10:
            miss_samples.append(current)
        return match.group(0)

    new_mif = _OB_RE.sub(_replace, mif_str)
    
    local_log(
        f"Rewrote {rewrite_count} <ImportObFile> reference(s) in MIF blob",
        "info" if rewrite_count else "warning",
    )

    if rewrite_count == 0:
        local_log("  No rewrites fired — dumping <ImportObFile> samples:", "warning")
        for i, m in enumerate(_OB_RE.finditer(mif_str)):
            if i >= 10:
                break
            local_log(f"    [{i}] {m.group(2)!r}", "warning")
        local_log("  Available basenames:", "warning")
        for bn in filename_to_new:
            local_log(f"    {bn!r}", "warning")
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
    def local_log(msg, level="info"): return _log_fn(msg, level, on_log)

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    deliverable_root = work_dir / f"translated_{target_lang}"
    xlf_out_dir      = deliverable_root / f"translated_{target_lang}"
    graphics_out_dir = deliverable_root / "graphics"
    
    xlf_out_dir.mkdir(parents=True, exist_ok=True)
    if graphics_source_dir is not None:
        graphics_out_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(xlf_filename).stem.rstrip(".") or "translated"
    ext  = Path(xlf_filename).suffix or ".xlf"
    if ext.lower() not in {".xlf", ".xliff"}:
        ext = ".xlf"

    local_log(f"[{target_lang}] translating text segments…")
    translated_bytes = translate_xliff_bytes(xlf_bytes, target_lang=target_lang)

    out_xlf_path = xlf_out_dir / f"{stem}_{target_lang}{ext}"
    out_xlf_path.write_bytes(translated_bytes)

    if graphics_source_dir is None:
        return deliverable_root

    mapping = process_xlf_references(
        xlf_path=out_xlf_path,
        target_lang=target_lang,
        out_folder=graphics_out_dir,
        rename_with_lang=False,
        out_xlf_path=out_xlf_path,
        src_graphics_folder=Path(graphics_source_dir),
    )

    if mapping:
        update_xlf_references(out_xlf_path, mapping, on_log=on_log)

    return deliverable_root
