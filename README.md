# FrameMaker XLIFF Translator — FastAPI

A self-contained REST service that translates Adobe FrameMaker XLIFF exports
**and optionally every graphic they reference** into any of 20 target
languages. Built for use as a ChatGPT Enterprise / Custom GPT Action.

This folder is independent of the Streamlit app — it can be deployed alone.

## What's in this folder

```
chatgpt_api/
├── main.py                  ← FastAPI app (endpoints + ZIP packaging)
├── translator.py            ← XLIFF text translator (FM-18-safe target injection)
├── image_ocr_translator.py  ← image + PDF OCR / layout-preserving redraw
├── pipeline.py              ← orchestrator: text + graphics + ref rewrite
├── test.html                ← in-browser debug harness (served at /test)
├── requirements.txt
├── Dockerfile               ← one-stage image with fonts + opencv libs
├── .env.example
└── README.md
```

No `lxml` namespace madness, no Celery, no Redis. One process, four endpoints.

## What it does

When `graphics_zip` is omitted from a `/translate` request — text-only:

1. `lxml`-parse the XLIFF, extract `<source>` + `<seg-source>/<mrk>` segments.
2. Merge FrameMaker's awkwardly-split fragments (`"°C"`, `"to"`, page-number
   tails after `"page"/"figure"/"table"`).
3. Classify each segment as `skip` / `safety` / `body`.
4. Batch through OpenAI (`response_format=json_object`, 3× retries on
   JSON / rate-limit errors).
5. Inject translations into `<target>` elements cloned from `<source>` (NOT
   `<seg-source>` — that triggers FrameMaker 18 *Internal Error 18004*).
6. Strip every `<seg-source>` from the output.

When `graphics_zip` is attached — full pipeline:

7. Base64-decode + gunzip the `<internal-file>` blob to recover the
   embedded MIF.
8. For every `<ImportObFileDI>` in the MIF, locate the source asset inside
   the uploaded Graphics folder. Lookup is forgiving: subfolder-anchored
   first (`<root>/<DI-derived-subfolder>/<basename>`), then root-anchored
   (`<root>/<basename>`), then recursive `rglob` over the whole archive.
9. Translate each image via GPT-4o vision OCR — bounding-box + bold-flag
   structured output, alignment + baseline + bg/text colour sampled from
   the source pixels, content-aware text removal via `cv2.inpaint`,
   binary-search font fitting for word-wrapped translations.
10. Translate each PDF page either at the text-layer (span-level redact +
    re-insert, coordinates preserved exactly) or as a raster (image OCR
    fallback for scanned pages).
11. Rewrite every `<ImportObFile>` in the MIF to point at the new asset.
    `<ImportObFileDI>` is left untouched (it's the device-independent
    original, FM's identity anchor).

## Response layout (strict)

The ZIP returned by `POST /translate` contains one folder per requested
language:

```
translated_<lang>/
    graphics/
        graphics/
            <subfolder>/
                <file>           ← translated image / PDF (original filename kept)
    translated_<lang>/
        <stem>_<lang>.xlf        ← translated XLIFF
```

From the XLF directory `translated_<lang>/`, every `<ImportObFile>` in the
embedded MIF walks `../graphics/graphics/<subfolder>/<file>` to find the
translated asset — which is exactly what FrameMaker expects on re-import.

## Quick start (local)

```bash
cd chatgpt_api
python -m venv .venv
. .venv/bin/activate          # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

cp .env.example .env          # then fill in OPENAI_API_KEY

uvicorn main:app --host 0.0.0.0 --port 8000
```

Open <http://localhost:8000/test> for the in-browser test harness or
<http://localhost:8000/docs> for the interactive Swagger UI.

## Quick start (Docker)

```bash
cd chatgpt_api
docker build -t xliff-api .
docker run --rm -p 8000:8000 --env-file .env xliff-api
```

The image installs `fonts-dejavu-core`, `fonts-noto-core`, and
`fonts-noto-cjk` so the image-redraw step has glyphs for every supported
language. If you trim the font set, CJK / Arabic translations will render
as tofu boxes.

## Endpoints

### `GET /languages`

Returns the supported target languages.

```bash
curl http://localhost:8000/languages
```

### `POST /translate`

**Text only** — translate the XLIFF, leave MIF references untouched:

```bash
curl -X POST http://localhost:8000/translate \
  -F "file=@manual.xlf" \
  -F "languages=de,fr,ja" \
  -H "X-API-Key: $API_KEY" \
  -o translated.zip
```

**With graphics** — also translate every referenced image / PDF and rewrite
the MIF refs:

```bash
zip -r graphics.zip Graphics/

curl -X POST http://localhost:8000/translate \
  -F "file=@manual.xlf" \
  -F "graphics_zip=@graphics.zip" \
  -F "languages=de,fr,ja" \
  -H "X-API-Key: $API_KEY" \
  -o translated.zip
```

`graphics_zip` can be the bare `Graphics/` folder or your whole FrameMaker
project — the lookup will find files at any nesting level.

## Authentication

Set `API_KEY` in `.env`. Every request then has to include the same value
in an `X-API-Key` header. Leave `API_KEY` blank for an open API (useful
during development).

ChatGPT Action setup:
1. Action authentication → **API Key**
2. Custom header name → `X-API-Key`
3. Paste the value of `API_KEY`

## Wire up to ChatGPT Enterprise (Custom GPT Action)

1. Deploy this service to a public HTTPS URL (Fly.io, Render, Cloud Run,
   your own server — anywhere that can host a Docker container).
2. In your Custom GPT → **Actions** → **Create new action**.
3. **Import from URL** → `https://<your-host>/openapi.json`.
   FastAPI generates the spec automatically; ChatGPT will discover
   `/languages` and `/translate`.
4. Set authentication as above (X-API-Key header).
5. Test from the GPT builder: attach a FrameMaker XLIFF (and optionally a
   Graphics ZIP), ask "translate this into German and French." The GPT
   will call `POST /translate` and surface the ZIP it gets back.

## Configuration

All knobs are env-var-driven. See [.env.example](.env.example) for the
full list. The variables that matter:

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | _(required)_ | OpenAI key used for every translation |
| `OPENAI_MODEL` | `gpt-4o` | Model passed to `chat.completions.create` (text + vision) |
| `BATCH_SIZE` | `40` | Segments per OpenAI text call |
| `API_KEY` | _(blank)_ | If set, every request needs `X-API-Key: <this>` |
| `MAX_UPLOAD_MB` | `20` | Reject XLFs larger than this |
| `MAX_GRAPHICS_MB` | `200` | Reject graphics_zip larger than this |
| `ALLOWED_ORIGINS` | `*` | CORS origins (comma-separated) |
| `LANGFUSE_*` | _(blank)_ | Auto-traces text-pipeline OpenAI calls if set |

## What this DOESN'T do

- **No async / queue.** The HTTP connection stays open for the whole
  translation. Image-heavy requests can take 5–15 minutes; tune your
  reverse-proxy / Render timeout accordingly. For very large jobs, prefer
  the full Streamlit + Celery stack in the parent directory.
- **No batching across files.** Each request handles one XLIFF. The
  `languages` field lets you fan out to multiple target languages in one
  call — but the file is the same throughout.
- **No partial results.** If any single language fails, the entire
  request fails with a 500. By design — partial ZIPs make it too easy
  to ship broken localised builds without noticing.
