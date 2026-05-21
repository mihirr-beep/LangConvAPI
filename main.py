"""
FastAPI service: FrameMaker XLIFF translation — text + graphics.

Designed to be called as a ChatGPT Custom GPT Action. Endpoints:

  GET  /              — service info / liveness (also accepts HEAD)
  GET  /health        — minimal liveness probe (also accepts HEAD)
  GET  /test          — built-in single-page HTML test harness
  GET  /languages     — list supported target language codes
  POST /translate     — upload .xlf [+ optional graphics.zip] + language list
                        → ZIP of per-language deliverable folders

The OpenAPI schema FastAPI generates at /openapi.json is suitable to paste
straight into ChatGPT's Action configuration.

What the response ZIP looks like
────────────────────────────────
Single language:
    <stem>_<lang>.zip
        translated_<lang>/
            graphics/graphics/<subfolder>/<file>      (only if graphics.zip provided)
            translated_<lang>/<stem>_<lang>.xlf

Multiple languages:
    <stem>_translated_<N>langs.zip
        translated_<lang1>/<…same as above…>
        translated_<lang2>/<…same as above…>
        …

This layout matches the strict spec FrameMaker re-import expects: from
the XLF directory `translated_<lang>/`, every <ImportObFile> reference
in the embedded MIF blob walks `../graphics/graphics/<…>/<file>` to find
the translated asset.

Authentication
──────────────
If the env var `API_KEY` is set, every request must carry it in the
`X-API-Key` header. If unset, the endpoint is open.

Concurrency / size limits
─────────────────────────
Each request is processed inline (no Celery, no queue) so the calling
client holds the connection open while OpenAI batches stream. For typical
XLIFFs (<10k segments, <5 MB text + <50 MB graphics) this is fine. For
larger inputs, deploy behind a load balancer with a 10-minute read
timeout, or split the request per language.
"""

from __future__ import annotations

import io
import logging
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import List, Optional

# ── Load .env BEFORE any module that reads os.environ at import time ─────────
# Reads `chatgpt_api/.env` (next to this file) regardless of where the user
# launched uvicorn from. Existing env vars (e.g. from `docker run --env-file`
# or a real shell export) win — `override=False` — so deployment-provided
# secrets aren't silently shadowed by a stale .env left on disk.
from dotenv import load_dotenv
_ENV_PATH = Path(__file__).resolve().parent / ".env"
if _ENV_PATH.exists():
    load_dotenv(_ENV_PATH, override=False)

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Security,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.security.api_key import APIKeyHeader

# IMPORTANT: these imports happen AFTER load_dotenv() above, because
# translator.py and image_ocr_translator.py read MODEL / BATCH_SIZE /
# OPENAI_API_KEY from os.environ at import time. Reordering breaks them.
from translator import LANGUAGES, supported_languages
from pipeline import translate_project

# Inline-load the developer test UI so a deployed image only needs main.py +
# translator.py + image_ocr_translator.py + pipeline.py + the HTML file
# (no static-files mount, no extra route prefix).
_HERE = Path(__file__).resolve().parent
_TEST_HTML_PATH = _HERE / "test.html"
try:
    _TEST_HTML = _TEST_HTML_PATH.read_text(encoding="utf-8")
except FileNotFoundError:
    _TEST_HTML = (
        "<h1>test.html missing</h1>"
        "<p>Place test.html next to main.py to enable the /test page.</p>"
    )

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
)
log = logging.getLogger("xliff_api")

# Startup sanity: surface immediately whether the OpenAI key is actually
# visible to this process. Mask all but the prefix so the log line is safe
# to leave in CI / Render logs.
_API_KEY = os.environ.get("OPENAI_API_KEY", "")
if _API_KEY:
    _masked = _API_KEY[:8] + "…" + _API_KEY[-4:] if len(_API_KEY) > 14 else "(set)"
    log.info(
        "OPENAI_API_KEY loaded: %s (from %s)",
        _masked, ".env" if _ENV_PATH.exists() else "shell",
    )
else:
    log.warning(
        "OPENAI_API_KEY is NOT set. Translation requests will fail with 400. "
        "Put the key in %s or export it before launching uvicorn.",
        _ENV_PATH,
    )

# Allowed file extensions for the uploaded source XLIFF.
ALLOWED_XLF_EXTS = {".xlf", ".xliff"}
ALLOWED_ZIP_EXTS = {".zip"}

# Comma-separated list of allowed origins (use "*" to allow any).
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",")
    if o.strip()
]

# Optional API-key gate. If unset, the endpoint is open.
EXPECTED_API_KEY = os.environ.get("API_KEY", "").strip()
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _require_api_key(api_key: str = Security(_api_key_header)) -> None:
    """Reject requests that don't carry the configured X-API-Key. No-op if no key configured."""
    if not EXPECTED_API_KEY:
        return
    if not api_key or api_key != EXPECTED_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid X-API-Key header.",
        )


# ── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="FrameMaker XLIFF Translator",
    version="2.0.0",
    description=(
        "Translate Adobe FrameMaker XLIFF exports into a target language, with "
        "optional GPT-4o vision OCR + layout-preserving redraw of every "
        "referenced image / PDF. Returns a ZIP archive containing one "
        "`translated_<lang>/` deliverable folder per requested language, each "
        "laid out exactly as FrameMaker expects for re-import:\n\n"
        "    translated_<lang>/\n"
        "        graphics/graphics/<subfolder>/<file>      ← translated assets\n"
        "        translated_<lang>/<stem>_<lang>.xlf       ← translated XLIFF\n\n"
        "Translated XLIFFs are FrameMaker-18 import-safe (use the "
        "<source>-cloning rule that avoids 'Internal Error 18004'). Reference "
        "rewriting points <ImportObFile> at the new asset path while leaving "
        "<ImportObFileDI> (the device-independent original) untouched."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS or ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
)


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.api_route(
    "/",
    methods=["GET", "HEAD"],
    summary="Service info / liveness probe",
    tags=["meta"],
)
def root() -> dict:
    """
    Service identification + liveness probe.

    Accepts both `GET` and `HEAD` so platform health checks (Render, Fly,
    Cloud Run, etc.) succeed regardless of which verb they use.
    """
    return {
        "service": "FrameMaker XLIFF Translator (text + graphics)",
        "version": app.version,
        "endpoints": [
            "GET /languages",
            "POST /translate",
            "GET /health",
            "GET /test",
        ],
        "auth_required": bool(EXPECTED_API_KEY),
    }


@app.api_route(
    "/health",
    methods=["GET", "HEAD"],
    summary="Minimal liveness probe",
    tags=["meta"],
    include_in_schema=False,
)
def health() -> dict:
    """
    Dedicated liveness endpoint — returns 200 / OK without touching any
    downstream services. Prefer this over `/` as the Render "Health Check
    Path".
    """
    return {"ok": True}


@app.get(
    "/test",
    summary="Developer test UI",
    tags=["meta"],
    response_class=HTMLResponse,
    include_in_schema=False,   # don't pollute the OpenAPI schema ChatGPT reads
)
def test_ui() -> str:
    """
    Single-page HTML harness that calls /languages and /translate against this
    same host. Useful for eyeballing the response, debugging auth, and
    confirming the OpenAPI shape before pasting the spec into a ChatGPT
    Custom Action.
    """
    return _TEST_HTML


@app.get(
    "/languages",
    summary="List supported target languages",
    tags=["meta"],
)
def list_languages() -> dict:
    """
    Returns the dictionary of supported target language codes mapped to their
    human-readable labels. Use the codes (keys) as values for the `languages`
    field on `POST /translate`.
    """
    return {"languages": supported_languages()}


# ── Helpers ──────────────────────────────────────────────────────────────────
def _safe_extract_zip(zip_bytes: bytes, dest: Path) -> None:
    """
    Zip-slip-safe extraction. Rejects any member whose resolved destination
    falls outside `dest`. Required because we're extracting an attacker-
    controlled archive (the uploaded graphics.zip).
    """
    dest = dest.resolve()
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        for member in zf.infolist():
            # Drop empty / leading-separator names.
            name = member.filename
            if not name or name.endswith("/") and "../" not in name:
                continue
            target = (dest / name).resolve()
            if not str(target).startswith(str(dest)):
                raise HTTPException(
                    status.HTTP_400_BAD_REQUEST,
                    f"Unsafe path in graphics.zip: {name!r}",
                )
        zf.extractall(dest)


def _zip_folder(folder: Path, zip_arcroot: str, zf: zipfile.ZipFile) -> int:
    """
    Append every file under `folder` into the open zipfile `zf`, rooted at
    `zip_arcroot/`. Returns the number of files added.
    """
    folder = folder.resolve()
    count = 0
    for path in folder.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(folder)
        arcname = f"{zip_arcroot}/{rel.as_posix()}".replace("//", "/")
        zf.write(path, arcname)
        count += 1
    return count


# ── /translate ───────────────────────────────────────────────────────────────
@app.post(
    "/translate",
    summary="Translate an XLIFF (+ optional graphics ZIP) into one or more target languages",
    tags=["translation"],
    response_class=StreamingResponse,
    responses={
        200: {
            "content": {"application/zip": {}},
            "description": (
                "ZIP archive containing one `translated_<lang>/` deliverable "
                "folder per requested language."
            ),
        },
        400: {"description": "Bad input (unsupported file type, unknown language, empty list, …)."},
        401: {"description": "Missing or invalid `X-API-Key` (when authentication is enabled)."},
        413: {"description": "Uploaded file exceeds the size limit."},
        500: {"description": "Translation failed (OpenAI error, malformed XLIFF, etc.)."},
    },
    dependencies=[Depends(_require_api_key)],
)
async def translate_endpoint(
    file: UploadFile = File(
        ...,
        description="An Adobe FrameMaker XLIFF export (`.xlf` or `.xliff`).",
    ),
    languages: str = Form(
        ...,
        description=(
            "Target language codes, comma-separated (e.g. `de,fr,ja`). "
            "Call GET /languages to see the supported codes."
        ),
        examples=["de", "de,fr,ja"],
    ),
    graphics_zip: Optional[UploadFile] = File(
        None,
        description=(
            "Optional ZIP of the FrameMaker Graphics folder referenced by the "
            "XLIFF. When provided, every <ImportObFileDI> in the XLIFF's "
            "embedded MIF blob is looked up inside this archive (subfolder-"
            "anchored, then root-anchored, then recursive), translated via "
            "GPT-4o vision OCR (images) or per-span PDF text-layer rewriting "
            "(PDFs), and saved under "
            "`translated_<lang>/graphics/graphics/<subfolder>/<file>`. "
            "When omitted, only text is translated and graphic references "
            "are left untouched."
        ),
    ),
) -> StreamingResponse:
    """
    Translate the uploaded XLIFF into every requested target language.

    Returns a ZIP archive whose top-level entries are one
    `translated_<lang>/` folder per language, each containing:

        translated_<lang>/
            graphics/graphics/<subfolder>/<file>      (only if graphics_zip given)
            translated_<lang>/<stem>_<lang>.xlf

    Languages are processed sequentially (each runs a separate OpenAI
    batch loop); if any single language fails, the entire request fails
    with a 500. Partial results are NOT returned.
    """
    # ── Validate XLF extension ───────────────────────────────────────────────
    if not file.filename:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Missing filename on uploaded XLIFF.")
    xlf_ext = Path(file.filename).suffix.lower()
    if xlf_ext not in ALLOWED_XLF_EXTS:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"Unsupported XLIFF file type {xlf_ext!r}. Allowed: {sorted(ALLOWED_XLF_EXTS)}.",
        )

    # ── Validate graphics_zip extension (if present) ─────────────────────────
    if graphics_zip is not None and graphics_zip.filename:
        zip_ext = Path(graphics_zip.filename).suffix.lower()
        if zip_ext not in ALLOWED_ZIP_EXTS:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                f"graphics_zip must be a .zip file (got {zip_ext!r}).",
            )

    # ── Parse + validate languages ───────────────────────────────────────────
    lang_list: List[str] = [code.strip() for code in languages.split(",") if code.strip()]
    if not lang_list:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "At least one target language code is required (comma-separated).",
        )
    unknown = [l for l in lang_list if l not in LANGUAGES]
    if unknown:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"Unsupported language code(s): {unknown}. "
            f"Call GET /languages for the supported list.",
        )

    # ── Read uploads ─────────────────────────────────────────────────────────
    xlf_data = await file.read()
    if not xlf_data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Uploaded XLIFF is empty.")

    max_mb = int(os.environ.get("MAX_UPLOAD_MB", "20"))
    if len(xlf_data) > max_mb * 1024 * 1024:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            f"XLIFF exceeds {max_mb} MB limit.",
        )

    zip_data: Optional[bytes] = None
    if graphics_zip is not None:
        zip_data = await graphics_zip.read()
        if not zip_data:
            # User attached an empty file — treat as if they hadn't attached one.
            log.info("graphics_zip is empty; falling back to text-only translation.")
            zip_data = None
        else:
            max_zip_mb = int(os.environ.get("MAX_GRAPHICS_MB", "200"))
            if len(zip_data) > max_zip_mb * 1024 * 1024:
                raise HTTPException(
                    status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    f"graphics_zip exceeds {max_zip_mb} MB limit.",
                )

    log.info(
        "Translate request: xlf=%s xlf_size=%dKB graphics_zip=%s langs=%s",
        file.filename,
        len(xlf_data) // 1024,
        f"{(graphics_zip.filename if graphics_zip else None)} ({len(zip_data)//1024 if zip_data else 0} KB)"
        if zip_data else "none",
        lang_list,
    )

    # ── Spin up a request-scoped work dir; auto-cleaned on the way out ───────
    with tempfile.TemporaryDirectory(prefix="xliff_api_") as _work:
        work_dir = Path(_work)

        # 1. Unpack the graphics ZIP once (shared across all languages).
        graphics_source_dir: Optional[Path] = None
        if zip_data:
            graphics_source_dir = work_dir / "_graphics_src"
            graphics_source_dir.mkdir(parents=True, exist_ok=True)
            try:
                _safe_extract_zip(zip_data, graphics_source_dir)
            except zipfile.BadZipFile:
                raise HTTPException(
                    status.HTTP_400_BAD_REQUEST,
                    "graphics_zip is not a valid ZIP archive.",
                )
            log.info("Graphics ZIP extracted to %s", graphics_source_dir)

        # 2. Run the pipeline per language.
        per_lang_roots: dict[str, Path] = {}
        for lang in lang_list:
            lang_work = work_dir / f"_run_{lang}"
            lang_work.mkdir(parents=True, exist_ok=True)
            try:
                root_dir = translate_project(
                    xlf_bytes=xlf_data,
                    xlf_filename=file.filename,
                    target_lang=lang,
                    work_dir=lang_work,
                    graphics_source_dir=graphics_source_dir,
                    on_log=lambda msg, level="info": getattr(log, level if level != "success" else "info", log.info)(msg),
                )
            except ValueError as e:
                log.warning("Translate %s failed: %s", lang, e)
                raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))
            except Exception as e:
                log.exception("Translate %s crashed", lang)
                raise HTTPException(
                    status.HTTP_500_INTERNAL_SERVER_ERROR,
                    f"Translation to {lang} failed: {e}",
                )

            per_lang_roots[lang] = root_dir
            log.info("  ✓ %s — deliverable at %s", lang, root_dir)

        # 3. Pack one combined response ZIP.
        stem = Path(file.filename).stem.rstrip(".") or "translated"
        zip_buf = io.BytesIO()
        total_files = 0
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for lang, root_dir in per_lang_roots.items():
                arcroot = root_dir.name   # "translated_<lang>"
                total_files += _zip_folder(root_dir, arcroot, zf)
        zip_buf.seek(0)
        log.info("Packed %d file(s) into response ZIP", total_files)

    # work_dir is now deleted; zip_buf holds everything we need.

    zip_filename = (
        f"{stem}_{lang_list[0]}.zip"
        if len(lang_list) == 1
        else f"{stem}_translated_{len(lang_list)}langs.zip"
    )

    return StreamingResponse(
        zip_buf,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{zip_filename}"',
            "X-Translated-Languages": ",".join(lang_list),
            "X-Source-Filename": file.filename,
            "X-Graphics-Processed": "yes" if zip_data else "no",
        },
    )
