# Stage 1 — build React UI (only copy files needed to build)
FROM node:20-bookworm-slim AS frontend-build
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci || npm install
COPY frontend/public ./public
COPY frontend/src ./src
COPY frontend/postcss.config.js frontend/tailwind.config.js ./
ENV CI=false
RUN npm run build \
    && test -f build/index.html \
    || (echo "ERROR: build/index.html missing" && exit 1)

# Stage 2 — Python API + ffmpeg
FROM python:3.12-slim-bookworm
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY vs.py .
COPY --from=frontend-build /app/frontend/build ./frontend/build
RUN test -f frontend/build/index.html \
    || (echo "ERROR: frontend/build missing in final image" && exit 1)
ENV PORT=5000
EXPOSE 5000
CMD ["sh", "-c", "gunicorn -w 1 -k gthread --threads 8 --timeout 600 --bind 0.0.0.0:${PORT} vs:app"]
