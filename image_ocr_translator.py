# image_ocr_translator.py — self-contained port for the FastAPI service.
#
# Logic mirrors the standalone CLI version verbatim, EXCEPT for one
# deliberate change: the hardcoded API-key fallback that lived in the
# CLI version has been removed. The key MUST come from the environment
# (loaded via python-dotenv in main.py before this module is imported).
# Leaving a key here would have re-leaked the same secret into deployed
# containers.

import os
import re
import io
import html
import gzip
import json
import base64
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fitz
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from openai import OpenAI
from lxml import etree

try:
    from langdetect import detect as _langdetect
    from langdetect import DetectorFactory
    DetectorFactory.seed = 0
    _LANGDETECT_AVAILABLE = True
except ImportError:
    _LANGDETECT_AVAILABLE = False
    print("[WARN] langdetect not installed — pip install langdetect")

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
client         = OpenAI(api_key=OPENAI_API_KEY)
MODEL          = os.environ.get("OPENAI_MODEL", "gpt-4o").strip() or "gpt-4o"
API_TIMEOUT    = 120
MAX_IMG_DIM    = 3000
BOX_PADDING    = 2
MIN_FONT       = 7
MAX_FONT       = 96
_MIN_CHARS_FOR_LANGDETECT = 20

# ─────────────────────────────────────────────────────────────────────────────
# Language maps
# ─────────────────────────────────────────────────────────────────────────────

LANG_NAMES: Dict[str, str] = {
    "zh-CN": "Simplified Chinese",   "zh-TW": "Traditional Chinese",
    "zh":    "Chinese",
    "ja":    "Japanese",             "ko":    "Korean",
    "de":    "German",               "fr":    "French",
    "es":    "Spanish",              "ar":    "Arabic",
    "pt":    "Portuguese",           "it":    "Italian",
    "vi":    "Vietnamese",           "nl":    "Dutch",
    "pl":    "Polish",               "ru":    "Russian",
    "tr":    "Turkish",              "sv":    "Swedish",
    "da":    "Danish",               "fi":    "Finnish",
    "nb":    "Norwegian",            "cs":    "Czech",
    "en":    "English",
}

_LANG_ROOT: Dict[str, str] = {
    "zh-CN": "zh-cn", "zh-TW": "zh-tw",
    "nb":    "no",    "pt-BR": "pt",
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff"}
PDF_EXTENSIONS   = {".pdf"}
MEDIA_EXTENSIONS = IMAGE_EXTENSIONS | PDF_EXTENSIONS


def _lang_root(lang_code: str) -> str:
    return _LANG_ROOT.get(lang_code, lang_code.split("-")[0].lower())

def _detect_language(text: str) -> Optional[str]:
    if not _LANGDETECT_AVAILABLE:
        return None
    clean = text.strip()
    if len(clean) < _MIN_CHARS_FOR_LANGDETECT:
        return None
    try:
        return _langdetect(clean)
    except Exception:
        return None

def _already_in_target_language(text: str, target_lang: str) -> bool:
    detected = _detect_language(text)
    if detected is None:
        return False
    return detected.lower().split("-")[0] == _lang_root(target_lang).split("-")[0]

_FONT_PATHS_REGULAR = [
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/calibri.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial.ttf",
    "arial.ttf",
]

_FONT_PATHS_BOLD = [
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/calibrib.ttf",
    "C:/Windows/Fonts/segoeuib.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial Bold.ttf",
    "arialbd.ttf",
]

def _get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    size = max(MIN_FONT, size)
    candidates = _FONT_PATHS_BOLD if bold else _FONT_PATHS_REGULAR
    for p in candidates:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    if bold:
        for p in _FONT_PATHS_REGULAR:
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()

def _encode_pil(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def _cap_image(img: Image.Image) -> Image.Image:
    w, h = img.size
    longest = max(w, h)
    if longest <= MAX_IMG_DIM:
        return img
    scale = MAX_IMG_DIM / longest
    nw, nh = int(w * scale), int(h * scale)
    print(f"      (downscaling {w}×{h} → {nw}×{nh})")
    return img.resize((nw, nh), Image.LANCZOS)

def _sample_bg_color(img: Image.Image, x: int, y: int, w: int, h: int) -> Tuple[int, int, int]:
    img_w, img_h = img.size
    pixels: List[Tuple[int, int, int]] = []

    def _push(px: int, py: int) -> None:
        if 0 <= px < img_w and 0 <= py < img_h:
            p = img.getpixel((px, py))
            if isinstance(p, int):
                p = (p, p, p)
            pixels.append(p[:3])

    for dx in range(w):
        _push(x + dx, y - 1)
        _push(x + dx, y + h)
    for dy in range(h):
        _push(x - 1, y + dy)
        _push(x + w, y + dy)

    if not pixels:
        return (255, 255, 255)
    arr = np.array(pixels, dtype=np.int32)
    return tuple(int(c) for c in np.median(arr, axis=0))

def _sample_text_color(
    img: Image.Image, x: int, y: int, w: int, h: int,
    bg: Tuple[int, int, int],
) -> Tuple[int, int, int]:
    img_w, img_h = img.size
    step_x = max(1, w // 12)
    step_y = max(1, h // 6)

    pixels: List[Tuple[int, int, int]] = []
    for dx in range(0, w, step_x):
        for dy in range(0, h, step_y):
            px, py = x + dx, y + dy
            if 0 <= px < img_w and 0 <= py < img_h:
                p = img.getpixel((px, py))
                if isinstance(p, int):
                    p = (p, p, p)
                pixels.append(p[:3])

    if not pixels:
        return (255, 255, 255) if sum(bg) < 384 else (0, 0, 0)

    arr = np.array(pixels, dtype=np.int32)
    bg_arr = np.array(bg, dtype=np.int32)
    dists = np.linalg.norm(arr - bg_arr, axis=1)

    if float(np.max(dists)) < 25:
        return (255, 255, 255) if sum(bg) < 384 else (0, 0, 0)

    top_k = max(1, len(pixels) // 4)
    top_idx = np.argsort(dists)[-top_k:]
    return tuple(int(c) for c in np.median(arr[top_idx], axis=0))

def _wrap_text(draw: ImageDraw.ImageDraw, text: str, max_w: int, font) -> List[str]:
    if not text:
        return []
    words = text.split()
    if not words:
        return [text]

    lines, cur = [], words[0]
    for word in words[1:]:
        test = cur + " " + word
        if draw.textbbox((0, 0), test, font=font)[2] <= max_w:
            cur = test
        else:
            lines.append(cur)
            cur = word
    lines.append(cur)

    wrapped: List[str] = []
    for line in lines:
        if draw.textbbox((0, 0), line, font=font)[2] <= max_w:
            wrapped.append(line)
            continue
        buf = ""
        for ch in line:
            if draw.textbbox((0, 0), buf + ch, font=font)[2] <= max_w:
                buf += ch
            else:
                if buf:
                    wrapped.append(buf)
                buf = ch
        if buf:
            wrapped.append(buf)
    return wrapped

def _fits(
    draw: ImageDraw.ImageDraw, text: str, w: int, h: int, font,
) -> Tuple[bool, List[str]]:
    lines = _wrap_text(draw, text, w, font)
    if not lines:
        return True, []
    line_h = draw.textbbox((0, 0), "Ag", font=font)[3] + 1
    if line_h * len(lines) > h:
        return False, lines
    max_w = max(draw.textbbox((0, 0), ln, font=font)[2] for ln in lines)
    return max_w <= w, lines

def _best_font(
    draw: ImageDraw.ImageDraw, text: str, w: int, h: int, initial_size: int,
    bold: bool = False,
) -> Tuple[ImageFont.FreeTypeFont, List[str]]:
    seed = max(MIN_FONT, min(MAX_FONT, initial_size))
    lo, hi = MIN_FONT, min(MAX_FONT, max(seed * 2, MIN_FONT + 1))

    best_font = _get_font(MIN_FONT, bold=bold)
    best_lines = _wrap_text(draw, text, w, best_font)

    while lo <= hi:
        mid = (lo + hi) // 2
        font = _get_font(mid, bold=bold)
        ok, lines = _fits(draw, text, w, h, font)
        if ok:
            best_font, best_lines = font, lines
            lo = mid + 1
        else:
            hi = mid - 1

    return best_font, best_lines

def _ocr_translate(b64_image: str, target_lang: str,
                   img_w: int, img_h: int) -> list:
    lang_name = LANG_NAMES.get(target_lang, target_lang)
    prompt = (
        "You are a precise OCR and translation engine.\n"
        f"The image is {img_w}×{img_h} pixels.\n\n"
        "TASK\n"
        "1. Detect EVERY piece of visible text in the image.\n"
        f"2. Translate EVERY piece into {lang_name}. "
        "Translate ALL text regardless of what language it appears to be in.\n"
        "3. Return the tight bounding box of each text block in pixels AND a\n"
        "   `bold` flag indicating whether the glyphs appear bold/heavy.\n\n"
        "OUTPUT — return ONLY a valid JSON array, no markdown fences:\n"
        '[{"original":"...","translated":"...","x":0,"y":0,"width":0,"height":0,"bold":false}]\n\n'
        "RULES\n"
        f"* x,y = top-left corner in pixels relative to the {img_w}×{img_h} image.\n"
        "* Include ALL text: headers, titles, table labels, cell text, captions.\n"
        "* Preserve numbers, symbols, and product/model codes exactly.\n"
        "* Brand names and proper nouns that do not translate: set translated=original.\n"
        "* `bold` is true ONLY if the glyph strokes are visibly thick/heavy;\n"
        "  otherwise false. Default to false when uncertain.\n"
        "* Do NOT wrap reply in markdown code blocks.\n"
    )
    print(f"      → GPT-4o OCR (timeout={API_TIMEOUT}s) …", flush=True)
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{b64_image}",
                               "detail": "high"}},
            ]}],
            max_tokens=4096,
            timeout=API_TIMEOUT,
        )
    except Exception as e:
        print(f"      ✗ API failed: {e}")
        return []

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        result = json.loads(raw)
        print(f"      ✓ {len(result)} block(s) detected")
        return result
    except Exception:
        print(f"      ✗ JSON parse failed:\n{raw[:300]}")
        return []

def _translate_texts_batch(texts: List[str], target_lang: str) -> List[str]:
    if not texts:
        return []
    lang_name = LANG_NAMES.get(target_lang, target_lang)
    numbered  = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
    prompt = (
        f"Translate each numbered item into {lang_name}.\n"
        "Rules:\n"
        "- Translate ALL items — every label, header, and field name.\n"
        "- Preserve numbers, symbols, and product/model codes exactly.\n"
        "- Words identical in both languages may stay as-is.\n"
        "- Return ONLY the numbered translations, same numbering, no extra text.\n\n"
        + numbered
    )
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4000,
            timeout=API_TIMEOUT,
        )
        raw   = resp.choices[0].message.content.strip()
        lines = raw.splitlines()
        out   = []
        for i, original in enumerate(texts):
            prefix  = f"{i+1}. "
            matched = next(
                (ln[len(prefix):].strip() for ln in lines if ln.startswith(prefix)),
                None,
            )
            out.append(matched if matched else original)
        return out
    except Exception as e:
        print(f"      ✗ batch translate error: {e}")
        return texts

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

def _detect_alignment(
    img: Image.Image, x: int, y: int, w: int, h: int,
    bg: Tuple[int, int, int],
) -> str:
    img_w, img_h = img.size
    bg_arr = np.array(bg, dtype=np.int32)

    sample_rows = [y + h // 4, y + h // 2, y + 3 * h // 4]
    sample_rows = [r for r in sample_rows if 0 <= r < img_h]
    if not sample_rows:
        return "left"

    step = max(1, w // 40)
    column_has_text: List[bool] = []
    for dx in range(0, w, step):
        px = x + dx
        if px < 0 or px >= img_w:
            column_has_text.append(False)
            continue
        is_text = False
        for py in sample_rows:
            pix = img.getpixel((px, py))
            if isinstance(pix, int):
                pix = (pix, pix, pix)
            dist = float(np.linalg.norm(
                np.array(pix[:3], dtype=np.int32) - bg_arr
            ))
            if dist > 35:
                is_text = True
                break
        column_has_text.append(is_text)

    text_columns = [i for i, t in enumerate(column_has_text) if t]
    if not text_columns:
        return "left"

    total = len(column_has_text)
    left_margin = text_columns[0] / total
    right_margin = (total - 1 - text_columns[-1]) / total

    if left_margin < 0.10 and right_margin < 0.10:
        return "left"
    if abs(left_margin - right_margin) < 0.10 and left_margin > 0.12:
        return "center"
    if left_margin > right_margin + 0.15:
        return "right"
    return "left"

def _find_text_baseline(
    img: Image.Image, x: int, y: int, w: int, h: int,
    bg: Tuple[int, int, int],
) -> int:
    img_w, img_h = img.size
    bg_arr = np.array(bg, dtype=np.int32)
    step_x = max(1, w // 20)

    for py in range(min(y + h, img_h) - 1, max(y - 1, -1), -1):
        if py < 0:
            continue
        count = 0
        for dx in range(0, w, step_x):
            px = x + dx
            if 0 <= px < img_w:
                pix = img.getpixel((px, py))
                if isinstance(pix, int):
                    pix = (pix, pix, pix)
                dist = float(np.linalg.norm(
                    np.array(pix[:3], dtype=np.int32) - bg_arr
                ))
                if dist > 35:
                    count += 1
        if count >= 2:
            return py
    return min(y + int(h * 0.85), img_h - 1)

def _erase_text_regions(pil_img: Image.Image, block_info: list) -> Image.Image:
    if not block_info:
        return pil_img.copy()

    if _CV2_AVAILABLE:
        try:
            return _erase_with_inpaint(pil_img, block_info)
        except Exception as e:
            print(f"      [WARN] cv2.inpaint failed ({e}); falling back to solid fill.")
            return _erase_with_solid_fill(pil_img, block_info)

    print("      [WARN] opencv-python not installed — using solid-fill fallback.")
    return _erase_with_solid_fill(pil_img, block_info)

def _erase_with_inpaint(pil_img: Image.Image, block_info: list) -> Image.Image:
    arr = np.array(pil_img.convert("RGB"))
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    img_h, img_w = bgr.shape[:2]

    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    text_threshold = 35

    for info in block_info:
        x, y, w, h = info["x"], info["y"], info["w"], info["h"]
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(img_w, x + w)
        y1 = min(img_h, y + h)
        if x1 <= x0 or y1 <= y0:
            continue

        bg_arr = np.array(info["bg"], dtype=np.int32)
        box = arr[y0:y1, x0:x1, :3].astype(np.int32)

        diff = box - bg_arr
        dist = np.linalg.norm(diff, axis=2)
        text_mask = (dist > text_threshold).astype(np.uint8) * 255

        kernel = np.ones((3, 3), np.uint8)
        text_mask = cv2.dilate(text_mask, kernel, iterations=1)

        existing = mask[y0:y1, x0:x1]
        mask[y0:y1, x0:x1] = np.maximum(existing, text_mask)

    cleaned_bgr = cv2.inpaint(bgr, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    cleaned_rgb = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(cleaned_rgb)

def _erase_with_solid_fill(pil_img: Image.Image, block_info: list) -> Image.Image:
    img = pil_img.copy()
    draw = ImageDraw.Draw(img)
    for info in block_info:
        x, y, w, h = info["x"], info["y"], info["w"], info["h"]
        draw.rectangle([(x, y), (x + w, y + h)], fill=info["bg"])
    return img

def _draw_blocks(pil_img: Image.Image, blocks: list) -> Image.Image:
    img_w, img_h = pil_img.size
    block_info: List[dict] = []
    for item in blocks:
        try:
            original   = item.get("original", "") or ""
            translated = item.get("translated") or item.get("text", "") or ""
            if not translated.strip() or translated.strip() == original.strip():
                continue

            x = max(0, min(int(item.get("x", 0)),     img_w - 1))
            y = max(0, min(int(item.get("y", 0)),     img_h - 1))
            w = min(int(item.get("width",  100)),      img_w - x)
            h = min(int(item.get("height",  20)),      img_h - y)
            if w <= 4 or h <= 4:
                continue

            bg = _sample_bg_color(pil_img, x, y, w, h)
            fg = _sample_text_color(pil_img, x, y, w, h, bg)
            alignment = _detect_alignment(pil_img, x, y, w, h, bg)
            baseline_y = _find_text_baseline(pil_img, x, y, w, h, bg)
            is_bold = bool(item.get("bold", False))

            block_info.append({
                "translated": translated,
                "x": x, "y": y, "w": w, "h": h,
                "bg": bg, "fg": fg,
                "alignment": alignment,
                "baseline_y": baseline_y,
                "bold": is_bold,
            })
        except Exception as e:
            print(f"      [analyse] skipping block: {e}")

    if not block_info:
        return pil_img.copy()

    cleaned_img = _erase_text_regions(pil_img, block_info)
    draw = ImageDraw.Draw(cleaned_img)

    for info in block_info:
        try:
            x, y, w, h = info["x"], info["y"], info["w"], info["h"]
            translated = info["translated"]
            fg         = info["fg"]
            alignment  = info["alignment"]
            baseline_y = info["baseline_y"]
            is_bold    = info["bold"]

            inner_w = max(1, w - 2 * BOX_PADDING)
            inner_h = max(1, h - 2 * BOX_PADDING)
            initial = max(MIN_FONT, int(h * 0.75))
            font, lines = _best_font(
                draw, translated, inner_w, inner_h, initial, bold=is_bold,
            )
            if not lines:
                continue

            line_h = draw.textbbox((0, 0), "Ag", font=font)[3] + 1
            total_text_h = line_h * len(lines)

            ty = baseline_y - total_text_h + 1
            ty = max(y + BOX_PADDING, min(ty, y + h - total_text_h))

            for line in lines:
                if ty + line_h > y + h:
                    break
                line_w = draw.textbbox((0, 0), line, font=font)[2]

                if alignment == "right":
                    tx = x + w - line_w - BOX_PADDING
                elif alignment == "center":
                    tx = x + (w - line_w) // 2
                else:
                    tx = x + BOX_PADDING

                draw.text((tx, ty), line, fill=fg, font=font)
                ty += line_h

        except Exception as e:
            print(f"      [draw] skipping block: {e}")

    return cleaned_img

def _extract_pdf_text(doc: fitz.Document, max_chars: int = 2000) -> str:
    parts, total = [], 0
    for page in doc:
        t = page.get_text("text").strip()
        parts.append(t)
        total += len(t)
        if total >= max_chars:
            break
    return " ".join(parts)

def _has_real_text(page: fitz.Page) -> bool:
    return bool(page.get_text("text").strip())

def _get_fitz_fontname(font_name: str) -> str:
    fn = font_name.lower()
    if "bold" in fn and ("italic" in fn or "oblique" in fn):
        return "hebobi" if ("helv" in fn or "arial" in fn) else "tibi"
    if "bold" in fn:
        return "hebo"   if ("helv" in fn or "arial" in fn) else "tibo"
    if "italic" in fn or "oblique" in fn:
        return "hebi"   if ("helv" in fn or "arial" in fn) else "tiit"
    return "helv"

def _process_text_layer_page(page: fitz.Page, target_lang: str) -> bool:
    spans: List[dict] = []
    seen:  set        = set()

    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span["text"].strip()
                if not text:
                    continue
                key = (text, tuple(round(v, 1) for v in span["bbox"]))
                if key in seen:
                    continue
                seen.add(key)
                spans.append({
                    "text": text,
                    "bbox": span["bbox"],
                    "size": span["size"],
                    "font": span.get("font", ""),
                })

    if not spans:
        return False

    all_text = " ".join(s["text"] for s in spans)
    if _already_in_target_language(all_text, target_lang):
        print("      - page already in target language — skipping")
        return False

    originals    = [s["text"] for s in spans]
    translations = _translate_texts_batch(originals, target_lang)

    to_process = [
        (span, tr)
        for span, tr in zip(spans, translations)
        if tr.strip() != span["text"].strip()
    ]

    if not to_process:
        print("      - no text changed after translation")
        return False

    for span, _tr in to_process:
        x0, y0, x1, y1 = span["bbox"]
        page.add_redact_annot(fitz.Rect(x0, y0 - 1, x1, y1 + 1), fill=(1, 1, 1))
    page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)

    any_changed = False
    for span, translated in to_process:
        x0, y0, x1, y1 = span["bbox"]
        rc = page.insert_text(
            (x0, y1 - 1), translated,
            fontsize=span["size"],
            fontname=_get_fitz_fontname(span["font"]),
            color=(0, 0, 0),
        )
        if rc >= 0:
            any_changed = True
            print(f"      ✓ {repr(span['text'][:28])} → {repr(translated[:28])}")
        else:
            print(f"      ✗ insert_text rc={rc} for {repr(span['text'][:28])}")

    return any_changed

def _process_image_layer_page(doc: fitz.Document,
                               page: fitz.Page,
                               target_lang: str) -> bool:
    pix     = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
    pil_img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    pil_img      = _cap_image(pil_img)
    img_w, img_h = pil_img.size

    blocks = _ocr_translate(_encode_pil(pil_img), target_lang, img_w, img_h)
    if not blocks:
        return False

    if all((b.get("translated") or "").strip() == (b.get("original") or "").strip()
           for b in blocks):
        print("      - all OCR blocks unchanged")
        return False

    pil_img = _draw_blocks(pil_img, blocks)
    buf     = io.BytesIO()
    pil_img.save(buf, format="PNG")
    page.clean_contents()
    for item in page.get_images(full=True):
        try:
            page.delete_image(item[0])
        except Exception:
            pass
    page.insert_image(page.rect, stream=buf.getvalue(), keep_proportion=False)
    return True

def process_image(
    source_path: Path,
    target_lang: str,
    out_folder: Path,
    rename_with_lang: bool = True,
) -> str:
    source_path = Path(source_path)
    print(f"\n  [IMG] {source_path.name}", flush=True)

    try:
        pil_img = Image.open(str(source_path)).convert("RGB")
    except Exception as e:
        print(f"  ✗ Cannot open image: {e}")
        return ""

    pil_img      = _cap_image(pil_img)
    img_w, img_h = pil_img.size
    blocks       = _ocr_translate(_encode_pil(pil_img), target_lang, img_w, img_h)

    new_name = (
        source_path.name if not rename_with_lang
        else f"{source_path.stem}_{target_lang}{source_path.suffix}"
    )
    out_path = out_folder / new_name

    if not blocks:
        print("  - No text detected — copying unchanged.")
        shutil.copy2(str(source_path), str(out_path))
        return new_name

    if all((b.get("translated") or "").strip() == (b.get("original") or "").strip()
           for b in blocks):
        print("  - No text required translation — copying unchanged.")
        shutil.copy2(str(source_path), str(out_path))
        return new_name

    pil_img  = _draw_blocks(pil_img, blocks)
    ext      = source_path.suffix.lower()
    fmt_map  = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG",
                ".bmp": "BMP", ".gif": "GIF", ".tif": "TIFF", ".tiff": "TIFF"}
    save_fmt = fmt_map.get(ext, "PNG")
    kw       = {"format": save_fmt}
    if save_fmt == "JPEG":
        kw["quality"] = 95
    pil_img.save(str(out_path), **kw)
    print(f"  ✓ Saved → {new_name}")
    return new_name

def process_pdf(
    source_path: Path,
    target_lang: str,
    out_folder: Path,
    rename_with_lang: bool = True,
) -> str:
    source_path = Path(source_path)
    print(f"\n  [PDF] {source_path.name}", flush=True)

    try:
        doc = fitz.open(str(source_path))
    except Exception as e:
        print(f"  ✗ Cannot open PDF: {e}")
        return ""

    new_name = (
        source_path.name if not rename_with_lang
        else f"{source_path.stem}_{target_lang}.pdf"
    )
    out_path = out_folder / new_name

    full_text = _extract_pdf_text(doc)
    if full_text.strip() and _already_in_target_language(full_text, target_lang):
        print("  - Document already in target language — copying unchanged.")
        doc.close()
        shutil.copy2(str(source_path), str(out_path))
        print(f"  ✓ Copied → {new_name}")
        return new_name

    if not full_text.strip():
        print("  - No extractable text — will attempt image-layer OCR.")

    any_changed = False
    for idx in range(len(doc)):
        page = doc[idx]
        print(f"      page {idx+1}/{len(doc)}", flush=True)
        if _has_real_text(page):
            changed = _process_text_layer_page(page, target_lang)
            print(f"      {'✓' if changed else '-'} "
                  f"{'text-layer translated' if changed else 'text-layer: no changes'}")
        else:
            changed = _process_image_layer_page(doc, page, target_lang)
            print(f"      {'✓' if changed else '-'} "
                  f"{'image-OCR translated' if changed else 'image-OCR: no text'}")
        if changed:
            any_changed = True

    if not any_changed:
        print("  - Nothing translated — copying unchanged.")
        doc.close()
        shutil.copy2(str(source_path), str(out_path))
        print(f"  ✓ Copied → {new_name}")
        return new_name

    tmp = str(out_path) + ".tmp.pdf"
    doc.save(tmp, garbage=4, deflate=True)
    doc.close()
    os.replace(tmp, str(out_path))
    print(f"  ✓ Saved → {new_name}")
    return new_name

_DI_RE = re.compile(
    r'<ImportObFileDI[^>]*>([^<]+)</ImportObFileDI>',
    re.IGNORECASE,
)

_OB_RE = re.compile(
    r'(<ImportObFile[^>]*>)([^<]+)(</ImportObFile>)',
    re.IGNORECASE,
)

def _parse_mif_path(raw_di: str) -> str:
    decoded   = html.unescape(raw_di.strip())
    converted = decoded.replace("<u>", "../").replace("<c>", "/")
    converted = converted.replace("..//" , "../")
    return converted

def _decode_internal_file_blob(xlf_path: Path) -> str:
    parser = etree.XMLParser(remove_blank_text=False, recover=True)
    try:
        tree = etree.parse(str(xlf_path), parser)
    except Exception as e:
        print(f"  ✗ Cannot parse XLF: {e}")
        return ""

    internal_el = None
    for elem in tree.getroot().iter():
        if elem.tag.split("}")[-1] == "internal-file":
            internal_el = elem
            break

    if internal_el is None:
        print("  [WARN] No <internal-file> element in XLF.")
        return ""

    raw_b64 = (internal_el.text or "").strip()
    if not raw_b64:
        print("  [WARN] <internal-file> element is empty.")
        return ""

    try:
        compressed = base64.b64decode(raw_b64)
    except Exception as e:
        print(f"  ✗ base64 decode failed: {e}")
        return ""

    if compressed[:2] == b'\x1f\x8b':
        try:
            return gzip.decompress(compressed).decode("utf-8", errors="replace")
        except Exception as e:
            print(f"  ✗ gzip decompress failed: {e}")
            return ""

    return compressed.decode("utf-8", errors="replace")

def extract_reference_paths(xlf_path: Path) -> List[Tuple[str, str]]:
    print(f"  Decoding internal-file blob in: {xlf_path.name}", flush=True)
    mif = _decode_internal_file_blob(xlf_path)
    if not mif:
        print("  ✗ Could not decode internal-file blob — no refs extracted.")
        return []

    base_dir = xlf_path.parent
    di_raws  = _DI_RE.findall(mif)

    print(f"  ImportObFileDI entries found: {len(di_raws)}")
    if not di_raws:
        print("  [WARN] No <ImportObFileDI> entries in MIF.")
        return []

    seen:   set                    = set()
    result: List[Tuple[str, str]] = []

    for raw in di_raws:
        fs_path_str = _parse_mif_path(raw)
        ext         = Path(fs_path_str).suffix.lower()

        if ext not in MEDIA_EXTENSIONS:
            print(f"    skip (unsupported ext '{ext}'): {fs_path_str!r}")
            continue

        abs_path = (base_dir / fs_path_str).resolve()
        key      = str(abs_path)

        if key in seen:
            continue
        seen.add(key)

        print(f"    DI raw : {raw!r}")
        print(f"    FS path: {fs_path_str!r}")
        print(f"    Abs    : {abs_path}")
        result.append((raw, str(abs_path)))

    return result

def _subfolder_from_di(di_fs_path: str) -> Path:
    p = Path(di_fs_path)
    skip = {'..', '.', '', '/', '\\'}
    real_parts = [
        part for part in p.parent.parts
        if part not in skip
        and not (len(part) == 3 and part[1] == ':')
    ]
    if not real_parts:
        return Path('.')
    return Path(*real_parts)

def process_xlf_references(
    xlf_path,
    target_lang: str,
    out_folder: Optional[Path] = None,
    rel_prefix: Optional[str] = None,
    rename_with_lang: bool = True,
    out_xlf_path: Optional[Path] = None,
    src_graphics_folder: Optional[Path] = None,
) -> Dict[str, str]:
    """
    Translate every graphic referenced inside `xlf_path`'s embedded MIF blob
    and return a {key → new-relative-path} mapping ready for the rewriter.

    Lookup hierarchy (per reference):
      1. Source basename joined to the DI-derived subfolder under `src_graphics_folder`
      2. Source basename joined to the root of `src_graphics_folder`
      3. Recursive `rglob(basename)` anywhere inside `src_graphics_folder`

    Output:
      Translated assets are written to `<out_folder>/<DI-derived-subfolder>/`.
      The mapping value (what becomes the new <ImportObFile> value) is the
      relative path from `out_xlf_path.parent` to the saved file. From the
      strict layout `<root>/translated_<lang>/<file>.xlf` to
      `<root>/graphics/graphics/<file>` that's `../graphics/graphics/<file>`.
    """
    xlf_path = Path(xlf_path)
    base_dir = xlf_path.parent

    if out_xlf_path is None:
        xlf_out_dir = base_dir
    else:
        xlf_out_dir = Path(out_xlf_path).parent

    print(f"\n{'='*60}")
    print(f"  process_xlf_references")
    print(f"  XLF      : {xlf_path}")
    print(f"  Target   : {target_lang}")
    print(f"  XLF out dir: {xlf_out_dir}")
    print(f"{'='*60}")

    refs = extract_reference_paths(xlf_path)
    print(f"\n  Total unique graphic refs: {len(refs)}")

    if not refs:
        print("  Nothing to process.")
        return {}

    if out_folder is None:
        out_folder = base_dir / f"Graphics_{target_lang}"
    out_folder = Path(out_folder)

    print(f"  Output root  : {out_folder}")
    print(f"  Lang suffix  : {rename_with_lang}\n")

    mapping: Dict[str, str] = {}

    for di_raw, abs_path_str in refs:
        abs_path = Path(abs_path_str)
        di_fs    = _parse_mif_path(di_raw)
        print(f"\n  File     : {abs_path.name}")
        print(f"  DI path  : {di_fs!r}")

        # Lookup strategy: prefer subfolder-anchored, then root, then rglob.
        found_src_path = None
        if src_graphics_folder:
            src_g_root = Path(src_graphics_folder)
            sub = _subfolder_from_di(di_fs)

            c1 = src_g_root / sub / abs_path.name
            if c1.is_file():
                found_src_path = c1
            else:
                c2 = src_g_root / abs_path.name
                if c2.is_file():
                    found_src_path = c2
                else:
                    for match in src_g_root.rglob(abs_path.name):
                        if match.is_file():
                            found_src_path = match
                            break

        if not found_src_path:
            print(f"  ✗ Image file not found inside uploaded folder hierarchy: {abs_path.name}")
            continue

        print(f"  ✓ Located image in uploaded Graphics folder -> {found_src_path}")
        abs_path = found_src_path

        sub = _subfolder_from_di(di_fs)
        dest_folder = out_folder / sub
        dest_folder.mkdir(parents=True, exist_ok=True)

        ext = abs_path.suffix.lower()
        try:
            if ext in IMAGE_EXTENSIONS:
                new_name = process_image(
                    abs_path, target_lang, dest_folder,
                    rename_with_lang=rename_with_lang,
                )
            elif ext in PDF_EXTENSIONS:
                new_name = process_pdf(
                    abs_path, target_lang, dest_folder,
                    rename_with_lang=rename_with_lang,
                )
            else:
                print(f"  - Unsupported extension {ext} — skipping.")
                continue

            if not new_name:
                continue

            saved_abs = dest_folder / new_name

            # Relative path from the (nested) XLF directory out to the saved
            # asset — this is what FrameMaker walks on re-import.
            mif_ref = os.path.relpath(str(saved_abs), str(xlf_out_dir)).replace(os.sep, "/")

            print(f"  Saved  → {saved_abs}")
            print(f"  MIF ref: {mif_ref!r}")

            mapping[abs_path.name]   = mif_ref
            mapping[di_fs]           = mif_ref
            mapping[di_raw]          = mif_ref

        except Exception as e:
            print(f"  ✗ Error on {abs_path.name}: {e}")

    print(f"\n{'='*60}")
    print(f"  Done. {len(set(mapping.values()))}/{len(refs)} file(s) translated.")
    print(f"{'='*60}")

    print()
    return mapping
