# FrameMaker XLIFF Translator — FastAPI

A self-contained REST service that translates Adobe FrameMaker XLIFF exports
into any of 20 target languages. **Text only** — referenced images, PDFs,
and graphics are passed through unchanged. Built for use as a ChatGPT
Enterprise / Custom GPT Action.

This folder is independent of the Streamlit app — it can be deployed alone.

## What's in this folder

```
chatgpt_api/
├── main.py             ← FastAPI app (2 endpoints)
├── translator.py       ← self-contained XLIFF text translator
├── requirements.txt    ← fastapi, uvicorn, openai, lxml
├── Dockerfile          ← one-stage image
├── .env.example
└── README.md
```

No `lxml` namespace madness, no Celery, no Redis. One process, two endpoints.

## Quick start (local)

```bash
cd chatgpt_api
python -m venv .venv
. .venv/bin/activate          # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

cp .env.example .env          # then fill in OPENAI_API_KEY
export $(cat .env | xargs)    # or use python-dotenv / load env however you prefer

uvicorn main:app --host 0.0.0.0 --port 8000
```

Open <http://localhost:8000/docs> for the interactive Swagger UI.

## Quick start (Docker)

```bash
cd chatgpt_api
docker build -t xliff-api .
docker run --rm -p 8000:8000 --env-file .env xliff-api
```

## Endpoints

### `GET /languages`

Returns the supported target languages.

```bash
curl http://localhost:8000/languages
```

Response:

```json
{
  "languages": {
    "de": "German (Deutsch)",
    "fr": "French (Français)",
    "ja": "Japanese (日本語)",
    "...": "..."
  }
}
```

### `POST /translate`

Translate a single uploaded XLIFF into one or more target languages. Returns
a ZIP containing one translated `.xlf` per language.

```bash
curl -X POST http://localhost:8000/translate \
  -F "file=@manual.xlf" \
  -F "languages=de,fr,ja" \
  -H "X-API-Key: $API_KEY" \
  -o translated.zip
```

Inside `translated.zip`:

```
manual_de.xlf
manual_fr.xlf
manual_ja.xlf
```

Each is a fully-formed, FrameMaker-18-importable XLIFF (uses the
`<source>`-cloning rule that avoids *Internal Error 18004*).

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
   your own server — anywhere that can host a Docker container or Python
   process).
2. In your Custom GPT → **Actions** → **Create new action**.
3. **Import from URL** → `https://<your-host>/openapi.json`.
   FastAPI generates the spec automatically; ChatGPT will discover both
   `/languages` and `/translate`.
4. Set authentication as above (X-API-Key header).
5. Test from the GPT builder: ask it to "translate `<attached file>` into
   German and French" — the GPT will call `POST /translate` and surface
   the ZIP it gets back.

## Configuration

All knobs are env-var-driven. See [.env.example](.env.example) for the full
list. The variables that matter:

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | _(required)_ | OpenAI key used for every translation |
| `OPENAI_MODEL` | `gpt-4o` | Model passed to `chat.completions.create` |
| `BATCH_SIZE` | `40` | Segments per OpenAI call |
| `API_KEY` | _(blank)_ | If set, every request needs `X-API-Key: <this>` |
| `MAX_UPLOAD_MB` | `20` | Reject XLFs larger than this |
| `ALLOWED_ORIGINS` | `*` | CORS origins (comma-separated) |
| `LANGFUSE_*` | _(blank)_ | Auto-traces OpenAI calls if set |

## What this DOESN'T do

- **No image / PDF OCR.** `<ImportObFile>` and `<ImportObFileDI>` paths in
  the XLIFF are passed through unchanged. If your downstream FrameMaker
  import needs translated graphics, run the full Streamlit + Celery stack
  in the parent directory.
- **No async / queue.** The HTTP connection stays open for the whole
  translation (typically 30 s – 5 min depending on file size). Tune your
  reverse-proxy timeout accordingly. For long jobs, prefer the full stack.
- **No batching across files.** Each request handles one XLIFF. The
  `languages` field lets you fan out to multiple target languages in one
  call — but the file is the same throughout.

## How the XLIFF translation works

Per language, the translator:

1. `lxml`-parses the input bytes, picks up the namespace if present.
2. Walks every `<trans-unit translate!="no">`, pulls out `<source>` (and
   per-`<mrk mtype="seg">` segments inside `<seg-source>`).
3. Merges FrameMaker's awkwardly-split fragments (lone units like `"°C"`,
   `"to"`, isolated numbers after `"page"/"figure"/"table"`).
4. Classifies each segment as `skip` (pure numbers / URLs / known
   do-not-translate entries) / `safety` (parent group's `resname` matches
   Warning/Caution/…) / `body`.
5. Batches the `safety`+`body` segments at `BATCH_SIZE`, calls OpenAI with
   strict `response_format=json_object`, retries 3× on JSON / rate-limit
   errors.
6. Writes each translation into a fresh `<target>` cloned from `<source>`
   (NOT `<seg-source>` — that triggers FrameMaker 18 Internal Error 18004).
7. Strips every `<seg-source>` from the output before serialising back to
   bytes.

The full logic lives in [translator.py](translator.py).
