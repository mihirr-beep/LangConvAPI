FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Fonts are needed at runtime: the image_ocr_translator step word-wraps
# translated strings and rasterises them on top of the redacted image.
# Without these packages, CJK / Arabic / extended-Latin glyphs fall back
# to tofu boxes. libglib2.0-0 covers the runtime opencv-python-headless
# occasionally still asks for on slim images.
RUN apt-get update && apt-get install -y --no-install-recommends \
        fonts-dejavu-core \
        fonts-noto-core \
        fonts-noto-cjk \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Run as non-root.
RUN useradd --create-home --uid 1000 appuser

# Copy every source file the service imports plus the static test page.
COPY --chown=appuser:appuser \
    translator.py \
    image_ocr_translator.py \
    pipeline.py \
    main.py \
    test.html \
    ./

USER appuser

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
