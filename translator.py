"""
Text-only FrameMaker XLIFF translator.

This is a self-contained port of the text-translation half of
translate_xliff_openai_2.py — no image OCR, no PDF processing, no CLI, no
dependency on the Streamlit app. Drop this file plus main.py into a folder
and the whole API is portable.

THE LOAD-BEARING RULE (do NOT remove)
─────────────────────────────────────
FrameMaker 18 exports two parallel elements per <trans-unit>:
  <source>      — plain text + <g> inline elements. FM reads THIS on re-import.
  <seg-source>  — same content but split into <mrk mtype="seg"> segments for
                  CAT tools. FM IGNORES this on re-import.

If <target> is built by cloning <seg-source>, FM 18 crashes with
"Internal Error 18004" because it can't handle <mrk> inside <target>.

write_back() clones <source> (NOT <seg-source>), then injects the translated
text reconstructed from the per-mrk translations, preserving whitespace/tab
tails. strip_seg_source() then removes every <seg-source> from the output.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import re
import time
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional

from lxml import etree

# Langfuse-wrapped OpenAI when available; falls back to plain OpenAI.
#
# We catch `Exception`, not just `ImportError`, because langfuse pulls in
# opentelemetry → requests → charset_normalizer, and that chain has been
# known to raise `SystemError: error return without exception set` on
# corrupted .pyc caches. A flaky tracing dependency must never take the
# whole API down — tracing is optional, translation is not.
try:
    from langfuse.openai import OpenAI  # type: ignore
except Exception as _e:  # pragma: no cover
    import logging as _logging
    _logging.getLogger(__name__).warning(
        "langfuse.openai import failed (%s: %s) — falling back to plain OpenAI client; "
        "tracing will be disabled for this run.",
        type(_e).__name__, _e,
    )
    from openai import OpenAI  # type: ignore

log = logging.getLogger(__name__)


# ── Defaults (overridable via env) ───────────────────────────────────────────
MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o").strip() or "gpt-4o"
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "8096"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "40"))
BATCH_DELAY = float(os.environ.get("BATCH_DELAY", "0.5"))


# ── Supported languages ──────────────────────────────────────────────────────
LANGUAGES: Dict[str, str] = {
    "zh-CN": "Simplified Chinese (简体中文)",
    "zh-TW": "Traditional Chinese (繁體中文)",
    "ja":    "Japanese (日本語)",
    "ko":    "Korean (한국어)",
    "de":    "German (Deutsch)",
    "fr":    "French (Français)",
    "es":    "Spanish (Español)",
    "ar":    "Arabic (العربية)",
    "pt":    "Portuguese (Português)",
    "it":    "Italian (Italiano)",
    "vi":    "Vietnamese (Tiếng Việt)",
    "nl":    "Dutch (Nederlands)",
    "pl":    "Polish (Polski)",
    "ru":    "Russian (Русский)",
    "tr":    "Turkish (Türkçe)",
    "sv":    "Swedish (Svenska)",
    "da":    "Danish (Dansk)",
    "fi":    "Finnish (Suomi)",
    "nb":    "Norwegian (Norsk Bokmål)",
    "cs":    "Czech (Čeština)",
}

FM_LANG = {k: k for k in LANGUAGES}


# ── DO-NOT-TRANSLATE list, safety styles, glossary ───────────────────────────
DO_NOT_TRANSLATE = {
    "SYSTEM OK", "CLASS 100", "CO2 AUTO CAL", "SYS IN OTEMP", "TSNSR1 ERR",
    "TSNSR2 ERR", "CO2 SNSR ERR", "O2 SNSR ERR", "REPL O2 SNSR", "REPL IR SNSR",
    "REPLACE HEPA", "ADD WATER", "DOOR IS OPEN", "CO2 IS HIGH", "CO2 IS LOW",
    "TEMP IS HIGH", "TEMP IS LOW", "O2 IS HIGH", "O2 IS LOW", "RH IS LOW",
    "IR AUTOZ ERR", "TANK1 LOW", "TANK2 LOW", "TANK 1 and 2 LOW",
    "RJ-11", "RS485", "RS-485", "RS232", "USB", "ESD",
}

SAFETY_STYLES = {
    "Warning", "Caution", "Note", "Important", "Danger", "WarningTitle",
    "CautionTitle", "NoteTitle", "ImportantTitle", "WarningBody", "CautionBody",
    "NoteBody", "ImportantBody", "Admonition", "AdmonitionTitle", "Hazard",
    "SafetyNote",
}

SAFETY_RE = re.compile(
    r"^\s*(Warning|Caution|Important|Note|Danger|WARNING|CAUTION|IMPORTANT)\b",
    re.IGNORECASE,
)

GLOSSARY: Dict[str, Dict[str, str]] = {
    "zh-CN": {
        "Operating Instructions": "操作说明",
        "Biological Safety Cabinet": "生物安全柜",
        "Water Jacket": "水套", "Incubator": "培养箱",
        "HEPA Filter": "HEPA过滤器", "Control Panel": "控制面板",
        "Setpoint": "设定点", "Calibration": "校准",
        "Warning": "警告", "Caution": "注意",
        "Important": "重要", "Note": "备注",
    },
    "zh-TW": {
        "Operating Instructions": "操作說明",
        "Biological Safety Cabinet": "生物安全櫃",
        "Warning": "警告", "Caution": "注意",
        "Important": "重要", "Note": "備註",
    },
    "ja": {
        "Operating Instructions": "取扱説明書",
        "Biological Safety Cabinet": "バイオセーフティキャビネット",
        "Warning": "警告", "Caution": "注意",
        "Important": "重要", "Note": "注",
    },
    "de": {
        "Warning": "Warnung", "Caution": "Vorsicht",
        "Important": "Wichtig", "Note": "Hinweis",
        "Operating Instructions": "Bedienungsanleitung",
    },
    "fr": {
        "Warning": "Avertissement", "Caution": "Attention",
        "Important": "Important", "Note": "Remarque",
        "Operating Instructions": "Mode d'emploi",
    },
    "es": {
        "Warning": "Advertencia", "Caution": "Precaución",
        "Important": "Importante", "Note": "Nota",
        "Operating Instructions": "Instrucciones de funcionamiento",
    },
    "vi": {
        "Warning": "Cảnh báo", "Caution": "Thận trọng",
        "Important": "Quan trọng", "Note": "Lưu ý",
        "Operating Instructions": "Hướng dẫn vận hành",
        "Biological Safety Cabinet": "Tủ an toàn sinh học",
        "HEPA Filter": "Bộ lọc HEPA", "Calibration": "Hiệu chuẩn",
    },
    "ko": {
        "Warning": "경고", "Caution": "주의",
        "Important": "중요", "Note": "참고",
    },
    "pt": {
        "Warning": "Aviso", "Caution": "Cuidado",
        "Important": "Importante", "Note": "Nota",
    },
    "it": {
        "Warning": "Avvertenza", "Caution": "Attenzione",
        "Important": "Importante", "Note": "Nota",
    },
    "ru": {
        "Warning": "Предупреждение", "Caution": "Осторожно",
        "Important": "Важно", "Note": "Примечание",
    },
    "nl": {
        "Warning": "Waarschuwing", "Caution": "Let op",
        "Important": "Belangrijk", "Note": "Opmerking",
    },
    "pl": {
        "Warning": "Ostrzeżenie", "Caution": "Uwaga",
        "Important": "Ważne", "Note": "Uwaga",
    },
    "tr": {
        "Warning": "Uyarı", "Caution": "Dikkat",
        "Important": "Önemli", "Note": "Not",
    },
}

XML_NS   = "http://www.w3.org/XML/1998/namespace"
XML_LANG = f"{{{XML_NS}}}lang"
XML_SPC  = f"{{{XML_NS}}}space"


# ── Namespace helpers ────────────────────────────────────────────────────────
def _detect_ns(root) -> str:
    tag = root.tag
    if "{" in tag:
        return tag.split("}")[0].lstrip("{")
    return root.get("xmlns", "")


def _Q(tag: str, ns: str) -> str:
    return f"{{{ns}}}{tag}" if ns else tag


# ── Load XLIFF from bytes ────────────────────────────────────────────────────
def _load_xliff_bytes(data: bytes):
    parser = etree.XMLParser(remove_blank_text=False, resolve_entities=False, recover=True)
    root = etree.fromstring(data, parser)
    tree = root.getroottree()
    ns = _detect_ns(root)
    return tree, root, ns


# ── Segment extraction ───────────────────────────────────────────────────────
def _style_from_group(tu, ns: str) -> str:
    p = tu.getparent()
    while p is not None:
        tag = p.tag.split("}")[-1] if "}" in p.tag else p.tag
        if tag == "group":
            rn = p.get("resname", "")
            if rn:
                return rn
        p = p.getparent()
    return ""


def _inner_text(el) -> str:
    if el is None:
        return ""
    return re.sub(
        r"<[^>]+>", "",
        etree.tostring(el, encoding="unicode", with_tail=False),
    ).strip()


def _extract_units(root, ns: str) -> List[dict]:
    units: List[dict] = []
    tag_tu  = _Q("trans-unit", ns)
    tag_src = _Q("source",     ns)
    tag_seg = _Q("seg-source", ns)
    tag_mrk = _Q("mrk",        ns)

    for tu in root.iter(tag_tu):
        if tu.get("translate", "yes").lower() == "no":
            continue
        tu_id   = tu.get("id", f"tu_{len(units):04d}")
        resname = tu.get("resname", "")
        style   = _style_from_group(tu, ns) or resname
        src_el  = tu.find(tag_src)
        seg_el  = tu.find(tag_seg)

        if seg_el is not None:
            seg_mrks = [m for m in seg_el.iter(tag_mrk) if m.get("mtype") == "seg"]
            for mrk in seg_mrks:
                units.append({
                    "id":         f"{tu_id}::mrk::{mrk.get('mid', '')}",
                    "tu_id":      tu_id,
                    "mrk_mid":    mrk.get("mid", ""),
                    "element":    tu,
                    "seg_src_el": seg_el,
                    "source":     mrk.text or "",
                    "style":      style,
                    "restype":    tu.get("restype", ""),
                })
        else:
            text = _inner_text(src_el)
            units.append({
                "id":         tu_id,
                "tu_id":      tu_id,
                "mrk_mid":    None,
                "element":    tu,
                "seg_src_el": None,
                "source":     text,
                "style":      style,
                "restype":    tu.get("restype", ""),
            })
    return units


# ── Merging awkward FrameMaker fragments ─────────────────────────────────────
_TEMP_UNIT_RE = re.compile(
    r"^[\s]*[+\-]?\d+[\d.,]*\s*$"
    r"|^[\s]*[°℃℉]\s*[CF]?\s*$"
    r"|^[\s]*[CF]\s*$"
    r"|^[\s]*to\s*$"
    r"|^[\s]*[~–\-]\s*$"
    r"|^[\s]*[°%()±]\s*$",
    re.IGNORECASE,
)

_PAGE_REF_TAIL_RE = re.compile(
    r"\b(page|figure|fig|table|chapter|section|step|item|part)\s+$",
    re.IGNORECASE,
)


def _is_unit_fragment(u: dict) -> bool:
    s = u["source"].strip()
    if not s:
        return True
    if len(s) <= 5 and _TEMP_UNIT_RE.match(s):
        return True
    if len(s) <= 3 and re.match(r"^[°CF%()±\s]+$", s, re.IGNORECASE):
        return True
    return False


def _is_page_number_suffix(prev_source: str, curr_source: str) -> bool:
    curr = curr_source.strip()
    if not re.match(r"^\d+$", curr):
        return False
    return bool(_PAGE_REF_TAIL_RE.search(prev_source))


def _merge_units(units: List[dict]) -> List[dict]:
    if not units:
        return units

    def is_small_old(u: dict) -> bool:
        s = u["source"].strip()
        return (
            len(s) <= 3
            or bool(re.match(r"^[°CF%()]+$", s))
            or bool(re.match(r"^\d+°?$", s))
        )

    merged: List[dict] = []
    buffer: Optional[dict] = None
    for u in units:
        if buffer is None:
            buffer = u
            continue
        if u["tu_id"] == buffer["tu_id"] and (is_small_old(u) or is_small_old(buffer)):
            buffer["source"] = buffer["source"].rstrip() + " " + u["source"].lstrip()
        else:
            merged.append(buffer)
            buffer = u
    if buffer:
        merged.append(buffer)

    result: List[dict] = []
    i = 0
    while i < len(merged):
        u = merged[i]
        if _is_unit_fragment(u) and result:
            prev = result[-1]
            prev["source"] = prev["source"].rstrip() + " " + u["source"].lstrip()
        elif i + 1 < len(merged) and _is_unit_fragment(merged[i + 1]):
            combined = u["source"]
            j = i + 1
            while j < len(merged) and _is_unit_fragment(merged[j]):
                combined = combined.rstrip() + " " + merged[j]["source"].lstrip()
                j += 1
            u = dict(u)
            u["source"] = combined
            result.append(u)
            i = j
            continue
        else:
            result.append(u)
        i += 1

    final: List[dict] = []
    for u in result:
        curr_src = u["source"]
        if (
            final
            and re.match(r"^\d+$", curr_src.strip())
            and _is_page_number_suffix(final[-1]["source"], curr_src)
        ):
            final[-1]["source"] = final[-1]["source"].rstrip() + " " + curr_src.strip()
        else:
            final.append(u)

    return final


# ── Three-way segment classification ─────────────────────────────────────────
def _classify(unit: dict) -> str:
    src = unit["source"].strip()
    if not src:
        return "skip"
    if re.search(r"\d", src) and re.search(r"[A-Za-z°℃℉]", src):
        return "body"
    if re.match(r"^[\d\s.\/%°×xX±~≤≥<>]+$", src):
        if re.match(r"^\d+$", src):
            return "body"
        return "skip"
    if re.match(r"^https?://|^www\.", src):
        return "skip"
    if len(src) <= 1 and not src.isalpha():
        return "skip"
    if src.upper() in {d.upper() for d in DO_NOT_TRANSLATE}:
        return "skip"
    style = unit.get("style", "")
    if style and any(s in style for s in SAFETY_STYLES):
        return "safety"
    if SAFETY_RE.match(src):
        return "safety"
    return "body"


# ── OpenAI batch translation ─────────────────────────────────────────────────
SYS_PROMPT = """You are a professional technical translator for laboratory equipment manuals.
Translate the segments from English into {lang}.
Rules:
1. NEVER translate these -- return them verbatim: {dnt}
2. Return ONLY plain text values. No XML tags in your response.
3. Glossary (use these exact translations): {glossary}
4. Segments with [SAFETY] prefix are safety-critical -- translate with maximum fidelity.
5. For segments containing temperatures like "-20°C to +60°C" or "(-4°F to +140°F)",
   preserve the numeric values and unit symbols exactly; only translate surrounding words.
6. For segments that are pure numbers (e.g. "30", "25", "20") return them verbatim unchanged.
7. Respond with ONLY a JSON object: {{"id":"translation"}}. No markdown, no explanation."""


def _build_sys(target_lang: str) -> str:
    lang  = LANGUAGES.get(target_lang, target_lang)
    dnt   = ", ".join(f'"{t}"' for t in sorted(DO_NOT_TRANSLATE)[:12])
    gdict = GLOSSARY.get(target_lang, {})
    gloss = "; ".join(f'"{en}"->"{tr}"' for en, tr in list(gdict.items())[:12])
    return SYS_PROMPT.format(lang=lang, dnt=dnt or "(none)", glossary=gloss or "(none)")


def _translate_batch(
    batch: List[dict], target_lang: str, sys_prompt: str, model_to_use: str,
    client: "OpenAI",
) -> Dict[str, str]:
    payload = {
        (f"[SAFETY]{u['id']}" if u.get("_class") == "safety" else u["id"]): u["source"]
        for u in batch
    }
    user_msg = (
        f"Translate {len(batch)} segments into "
        f"{LANGUAGES.get(target_lang, target_lang)}.\n"
        f"Return ONLY JSON.\n\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )

    for attempt in range(1, 4):
        try:
            response = client.chat.completions.create(
                model=model_to_use,
                max_tokens=MAX_TOKENS,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user",   "content": user_msg},
                ],
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            result = json.loads(raw)
            return {k.replace("[SAFETY]", ""): v for k, v in result.items()}

        except json.JSONDecodeError as e:
            log.warning(f"Attempt {attempt} JSON error: {e}")
            if attempt == 3:
                return {u["id"]: u["source"] for u in batch}
            time.sleep(2 ** attempt)

        except Exception as e:
            err = str(e).lower()
            wait = 30 * attempt if ("rate" in err or "429" in err) else 5
            log.warning(f"Attempt {attempt}: {e} -- wait {wait}s")
            if attempt == 3:
                return {u["id"]: u["source"] for u in batch}
            time.sleep(wait)

    return {u["id"]: u["source"] for u in batch}


# ── seg-source strip + target injection ──────────────────────────────────────
def _strip_seg_source(root, ns: str) -> int:
    tag_seg = _Q("seg-source", ns)
    removed = 0
    for seg_el in root.findall(f".//{tag_seg}"):
        parent = seg_el.getparent()
        if parent is not None:
            prev = seg_el.getprevious()
            tail = seg_el.tail or ""
            if prev is not None:
                prev.tail = (prev.tail or "") + tail
            else:
                parent.text = (parent.text or "") + tail
            parent.remove(seg_el)
            removed += 1
    return removed


def _inject_translation_into_source_clone(tgt, seg_el, mid_map, tag_mrk, tag_g):
    parts: List[str] = []
    for child in seg_el:
        ctag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
        if ctag == "mrk" and child.get("mtype") == "seg":
            mid = child.get("mid", "")
            translated = mid_map.get(mid, child.text or "")
            parts.append(translated)
            if child.tail:
                parts.append(child.tail)
        else:
            inner_parts = []
            for m in child.iter(tag_mrk):
                if m.get("mtype") == "seg":
                    inner_parts.append(mid_map.get(m.get("mid", ""), m.text or ""))
            if inner_parts:
                parts.append(" ".join(inner_parts))
            if child.tail:
                parts.append(child.tail)

    translated_text = "".join(parts)
    children = list(tgt)

    if not children:
        tgt.text = translated_text
        return

    tgt_text_is_content = bool(tgt.text and tgt.text.strip())
    all_no_translate = all(
        c.get("translate", "yes").lower() == "no" for c in children
    )
    if all_no_translate:
        placed = False
        if tgt_text_is_content:
            tgt.text = translated_text
            placed = True
        if not placed:
            for child in children:
                if child.tail and child.tail.strip():
                    child.tail = translated_text
                    placed = True
                    break
        if not placed:
            children[-1].tail = translated_text
        return

    if tgt_text_is_content:
        tgt.text = translated_text
        for child in children:
            if child.get("translate", "yes").lower() == "no":
                child.text = None
                child.tail = None
        return

    tgt.text = None
    placed = False
    for i, child in enumerate(children):
        is_last = (i == len(children) - 1)
        translatable = child.get("translate", "yes").lower() != "no"

        if not placed:
            if child.tail and child.tail.strip():
                child.tail = translated_text
                placed = True
                for j in range(i + 1, len(children)):
                    c = children[j]
                    if c.get("translate", "yes").lower() == "no":
                        c.text = None
                    c.tail = None
                return
            elif translatable and child.text and child.text.strip():
                child.text = translated_text
                child.tail = None
                placed = True
                for j in range(i + 1, len(children)):
                    c = children[j]
                    if c.get("translate", "yes").lower() == "no":
                        c.text = None
                    c.tail = None
                return
            elif is_last:
                child.tail = translated_text
                placed = True
            else:
                if not translatable:
                    child.text = None
                child.tail = None
        else:
            if not translatable:
                child.text = None
            child.tail = None


def _write_back(units: List[dict], translations: Dict[str, str], ns: str, target_lang: str) -> int:
    tag_src    = _Q("source",     ns)
    tag_target = _Q("target",     ns)
    tag_mrk    = _Q("mrk",        ns)
    tag_g      = _Q("g",          ns)
    lang_code  = FM_LANG.get(target_lang, target_lang)

    by_tu = defaultdict(list)
    for u in units:
        by_tu[u["tu_id"]].append(u)

    updated = 0
    for tu_id, tu_units in by_tu.items():
        tu_el  = tu_units[0]["element"]
        seg_el = tu_units[0]["seg_src_el"]
        src_el = tu_el.find(tag_src)

        for old in tu_el.findall(tag_target):
            tu_el.remove(old)

        if seg_el is not None:
            mid_map: Dict[str, str] = {}
            for u in tu_units:
                if u["mrk_mid"] is not None:
                    t = translations.get(u["id"])
                    if t is not None:
                        mid_map[u["mrk_mid"]] = t
            if not mid_map:
                mid_map = {
                    u["mrk_mid"]: u["source"]
                    for u in tu_units if u["mrk_mid"] is not None
                }

            tgt = copy.deepcopy(src_el) if src_el is not None else etree.Element(tag_target)
            tgt.tag = tag_target
            if XML_LANG in tgt.attrib:
                del tgt.attrib[XML_LANG]
            if XML_SPC in tgt.attrib:
                del tgt.attrib[XML_SPC]
            tgt.set(XML_LANG, lang_code)
            tgt.set("state", "translated")
            tgt.tail = src_el.tail if src_el is not None else seg_el.tail

            _inject_translation_into_source_clone(tgt, seg_el, mid_map, tag_mrk, tag_g)

            ref_el = src_el if src_el is not None else seg_el
            ref_idx = list(tu_el).index(ref_el)
            tu_el.insert(ref_idx + 1, tgt)
            updated += 1
        else:
            u = tu_units[0]
            t = translations.get(u["id"])
            if t is None:
                continue
            if src_el is not None:
                tgt = copy.deepcopy(src_el)
                tgt.tag = tag_target
                if XML_LANG in tgt.attrib:
                    del tgt.attrib[XML_LANG]
                if XML_SPC in tgt.attrib:
                    del tgt.attrib[XML_SPC]
                tgt.set(XML_LANG, lang_code)
                tgt.set("state", "translated")
                tgt.tail = src_el.tail
                children = list(tgt)
                if not children:
                    tgt.text = t
                else:
                    tgt.text = None
                    placed = False
                    for child in children:
                        if not placed and child.tail and child.tail.strip():
                            child.tail = t
                            placed = True
                        elif (
                            not placed
                            and child.get("translate", "yes").lower() != "no"
                            and child.text and child.text.strip()
                        ):
                            child.text = t
                            child.tail = None
                            placed = True
                        else:
                            if child.get("translate", "yes").lower() == "no":
                                child.text = None
                            child.tail = None
                    if not placed:
                        if children:
                            children[-1].tail = t
                        else:
                            tgt.text = t
                idx = list(tu_el).index(src_el)
                tu_el.insert(idx + 1, tgt)
            else:
                tgt = etree.Element(tag_target)
                tgt.set(XML_LANG, lang_code)
                tgt.set("state", "translated")
                tgt.text = t
                tu_el.append(tgt)
            updated += 1
    return updated


def _validate_xml(tree) -> bool:
    try:
        raw = etree.tostring(tree.getroot(), encoding="unicode")
        etree.fromstring(raw.encode("utf-8"))
        return True
    except etree.XMLSyntaxError:
        return False


def _set_header_lang(root, ns: str, target_lang: str) -> None:
    lc = FM_LANG.get(target_lang, target_lang)
    for f in root.iter(_Q("file", ns)):
        f.set("target-language", lc)


def _tree_to_bytes(tree) -> bytes:
    buf = BytesIO()
    tree.write(buf, encoding="utf-8", xml_declaration=True, pretty_print=True)
    return buf.getvalue()


# ── Public entry point ───────────────────────────────────────────────────────
def translate_xliff_bytes(
    input_bytes: bytes,
    target_lang: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    batch_size: Optional[int] = None,
) -> bytes:
    """
    Translate a FrameMaker XLIFF file's text into `target_lang` and return the
    result as bytes (same XLIFF schema, same structure, with <target> elements
    populated). Graphics, images, and PDFs referenced by the XLIFF are NOT
    processed — this is the text-only pipeline.

    Parameters
    ----------
    input_bytes  : raw .xlf / .xliff content
    target_lang  : language code (must be a key in LANGUAGES)
    api_key      : OpenAI API key. If omitted, OPENAI_API_KEY env var is used.
    model        : OpenAI model name. Defaults to MODEL (gpt-4o).
    batch_size   : segments per OpenAI call. Defaults to BATCH_SIZE.

    Raises
    ------
    ValueError if target_lang is unsupported, no segments are found, or the
    output XML fails validation.
    """
    if target_lang not in LANGUAGES:
        raise ValueError(
            f"Unsupported target language: {target_lang!r}. "
            f"Supported: {sorted(LANGUAGES.keys())}"
        )

    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError(
            "OpenAI API key missing: set OPENAI_API_KEY env var or pass api_key."
        )

    used_model = (model or MODEL).strip() or MODEL
    used_batch = batch_size or BATCH_SIZE

    client = OpenAI(api_key=api_key)

    tree, root, ns = _load_xliff_bytes(input_bytes)
    units = _merge_units(_extract_units(root, ns))
    if not units:
        raise ValueError("No translatable segments found in the input XLIFF.")

    to_translate, skipped = [], []
    for u in units:
        cls = _classify(u)
        u["_class"] = cls
        (skipped if cls == "skip" else to_translate).append(u)

    sys_p = _build_sys(target_lang)
    batches = [
        to_translate[i:i + used_batch]
        for i in range(0, len(to_translate), used_batch)
    ]

    all_trans: Dict[str, str] = {}
    for i, batch in enumerate(batches, 1):
        log.info(f"Batch {i}/{len(batches)} ({len(batch)} segs)")
        result = _translate_batch(batch, target_lang, sys_p, used_model, client)
        all_trans.update(result)
        if i < len(batches):
            time.sleep(BATCH_DELAY)

    for u in skipped:
        all_trans[u["id"]] = u["source"]

    _set_header_lang(root, ns, target_lang)
    _write_back(units, all_trans, ns, target_lang)
    _strip_seg_source(root, ns)

    if not _validate_xml(tree):
        raise ValueError("Output XML failed validation after translation.")

    return _tree_to_bytes(tree)


def supported_languages() -> Dict[str, str]:
    """Return the dict of supported language codes → human-readable labels."""
    return dict(LANGUAGES)
