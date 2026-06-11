# Stage 1 — build React UI
FROM node:20-bookworm-slim AS frontend-build
WORKDIR /app
COPY frontend/package.json frontend/package-lock.json ./frontend/
RUN npm ci --prefix frontend || npm install --prefix frontend
COPY frontend/ ./frontend/
# Render sets CI=true; don't fail the image build on ESLint warnings
ENV CI=false
RUN npm run build --prefix frontend \
    && test -f frontend/build/index.html \
    || (echo "ERROR: frontend/build/index.html missing after npm run build" && exit 1)

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
    || (echo "ERROR: frontend/build not copied into final image" && exit 1)
ENV PORT=5000
EXPOSE 5000
CMD ["sh", "-c", "gunicorn -w 1 -k gthread --threads 8 --timeout 600 --bind 0.0.0.0:${PORT} vs:app"]
