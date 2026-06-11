# Stage 1 — build React UI
FROM node:20-bookworm-slim AS frontend-build
WORKDIR /app
COPY frontend/package.json frontend/package-lock.json ./frontend/
RUN npm ci --prefix frontend
COPY frontend/ ./frontend/
RUN npm run build --prefix frontend

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
ENV PORT=5000
EXPOSE 5000
CMD ["sh", "-c", "gunicorn -w 1 -k gthread --threads 8 --timeout 600 --bind 0.0.0.0:${PORT} vs:app"]
