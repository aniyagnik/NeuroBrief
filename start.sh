#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [[ ! -d frontend/build ]] || [[ ! -f frontend/build/index.html ]]; then
  echo "Building frontend..."
  (cd frontend && npm install && npm run build)
fi

if [[ -d .venv ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

PORT="${PORT:-5000}"
if [[ "${FLASK_DEBUG:-0}" == "1" ]]; then
  echo "Starting Flask dev server on port ${PORT}..."
  exec python vs.py
fi

echo "Starting gunicorn on port ${PORT}..."
exec gunicorn -w 1 -k gthread --threads 8 --timeout 600 --bind "0.0.0.0:${PORT}" vs:app
