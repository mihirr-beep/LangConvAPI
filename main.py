"""
FastAPI service: text-only FrameMaker XLIFF translation.

Designed to be called as a ChatGPT Custom GPT Action. Two endpoints:

  GET  /languages              — list supported target language codes
  POST /translate              — upload .xlf + language list → ZIP of translated .xlf

The OpenAPI schema FastAPI generates at /openapi.json is suitable to paste
straight into ChatGPT's Action configuration.

Authentication
──────────────
If the env var `API_KEY` is set, every request must carry it in the
`X-API-Key` header. If unset, the endpoint is open. Configure the same key
on the ChatGPT Action side under "API Key → Custom → X-API-Key".

Concurrency / size limits
─────────────────────────
Each request is processed inline (no Celery, no queue) so the calling client
holds the connection open while OpenAI batches stream. For typical XLIFFs
(<10k segments, <5 MB) this is fine. For very large inputs, deploy behind a
load balancer with a generous read timeout (5–10 minutes).
"""

from __future__ import annotations

import io
import logging
import os
import zipfile
from pathlib import Path
from typing import List

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

# IMPORTANT: this import happens AFTER load_dotenv() above, because
# translator.py reads MODEL / BATCH_SIZE / MAX_TOKENS from os.environ at
# import time. Reordering breaks those defaults.
from translator import (
    LANGUAGES,
    supported_languages,
    translate_xliff_bytes,
)

# Inline-load the developer test UI so a deployed image only needs main.py +
# translator.py + the HTML file (no static-files mount, no extra route prefix).
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
# to leave in CI / Vercel logs.
_API_KEY = os.environ.get("OPENAI_API_KEY", "")
if _API_KEY:
    _masked = _API_KEY[:8] + "…" + _API_KEY[-4:] if len(_API_KEY) > 14 else "(set)"
    log.info("OPENAI_API_KEY loaded: %s (from %s)",
             _masked,
             ".env" if _ENV_PATH.exists() else "shell")
else:
    log.warning(
        "OPENAI_API_KEY is NOT set. Translation requests will fail with 400. "
        "Put the key in %s or export it before launching uvicorn.",
        _ENV_PATH,
    )

# Allowed file extensions for the uploaded source XLIFF.
ALLOWED_EXTS = {".xlf", ".xliff"}

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
    version="1.0.0",
    description=(
        "Translate Adobe FrameMaker XLIFF exports into a target language. "
        "Text-only — referenced images, PDFs, and graphics are NOT processed. "
        "Returns a ZIP archive containing one translated .xlf per language. "
        "Translated XLIFFs are FrameMaker-18 import-safe (use the <source>-cloning "
        "rule that avoids 'Internal Error 18004')."
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
    Cloud Run, etc.) succeed regardless of which verb they use. FastAPI's
    plain `@app.get` only handles GET → HEAD returns 405 → Render treats
    the deploy as unhealthy and shuts it down, which is exactly what was
    happening before this route was widened.
    """
    return {
        "service": "FrameMaker XLIFF Translator (text only)",
        "version": app.version,
        "endpoints": ["GET /languages", "POST /translate", "GET /health"],
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
    Path" because it's guaranteed never to grow heavier dependencies.
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


@app.post(
    "/translate",
    summary="Translate an XLIFF into one or more target languages",
    tags=["translation"],
    response_class=StreamingResponse,
    responses={
        200: {
            "content": {"application/zip": {}},
            "description": "ZIP archive containing one translated `.xlf` per requested language.",
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
) -> StreamingResponse:
    """
    Translate the uploaded XLIFF into every requested target language and
    return a ZIP archive. Inside the archive:

        <original-stem>_<lang>.xlf

    one entry per language. Languages are processed sequentially (each runs a
    separate OpenAI batch loop); if any single language fails, the entire
    request fails with a 500.

    This endpoint **does not** process image references inside the XLIFF —
    `<ImportObFile>` and `<ImportObFileDI>` paths are passed through unchanged.
    Use the full Streamlit + Celery stack if you need OCR + graphics
    translation.
    """
    # ── Validate file extension ──────────────────────────────────────────────
    if not file.filename:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Missing filename on uploaded file.")
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"Unsupported file type {ext!r}. Allowed: {sorted(ALLOWED_EXTS)}.",
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

    # ── Read upload ──────────────────────────────────────────────────────────
    data = await file.read()
    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Uploaded file is empty.")

    max_mb = int(os.environ.get("MAX_UPLOAD_MB", "20"))
    if len(data) > max_mb * 1024 * 1024:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            f"File exceeds {max_mb} MB limit.",
        )

    log.info(
        "Translate request: file=%s size=%dKB langs=%s",
        file.filename, len(data) // 1024, lang_list,
    )

    # ── Translate per language, collect bytes ────────────────────────────────
    results: dict[str, bytes] = {}
    for lang in lang_list:
        try:
            translated = translate_xliff_bytes(data, target_lang=lang)
        except ValueError as e:
            log.warning("Translate %s failed: %s", lang, e)
            raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))
        except Exception as e:
            log.exception("Translate %s crashed", lang)
            raise HTTPException(
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                f"Translation to {lang} failed: {e}",
            )
        results[lang] = translated
        log.info("  ✓ %s — %d KB", lang, len(translated) // 1024)

    # ── Pack ZIP ─────────────────────────────────────────────────────────────
    stem = Path(file.filename).stem.rstrip(".")
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for lang, translated_bytes in results.items():
            arcname = f"{stem}_{lang}{ext}"
            zf.writestr(arcname, translated_bytes)
    zip_buf.seek(0)

    zip_filename = (
        f"{stem}_{lang_list[0]}.xlf.zip"
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
        },
    )
