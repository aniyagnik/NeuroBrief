# NeuroBrief

Upload a video or paste a YouTube link to get an AI-generated **summary**, **quiz**, and **transcript**.

Single repo: React frontend + Flask backend. In production the backend serves the built UI from `frontend/build/`.

## Project layout

```
.
├── frontend/           # React app (Create React App)
│   ├── src/            # UI components
│   ├── public/
│   └── package.json
├── vs.py               # Flask API, background jobs, static file serving
├── requirements.txt    # Python dependencies
├── start.sh            # Production start (build UI + gunicorn)
├── Procfile            # Heroku / Railway deploy
├── .env.example        # Copy to .env and add API keys
└── package.json        # Root scripts (build, dev helpers)
```

## Prerequisites

- Python 3.10+
- Node.js 18+
- ffmpeg (for uploaded video files)
- API keys: [AssemblyAI](https://www.assemblyai.com/) and [Hugging Face](https://huggingface.co/)

## Setup

```bash
# 1. Clone and enter the repo
git clone <your-repo-url>
cd video-summarization

# 2. Python environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Frontend dependencies
npm run install:frontend

# 4. Configure secrets
cp .env.example .env
# Edit .env — at minimum set ASSEMBLYAI_API_KEY and HF_API_KEY
```

## Development

Run **two terminals** (UI on :3000, API on :5000; CRA proxies API calls):

```bash
# Terminal 1 — backend
source .venv/bin/activate
FLASK_DEBUG=1 python vs.py

# Terminal 2 — frontend
npm run dev:ui
```

Open [http://localhost:3000](http://localhost:3000).

## Production (local)

Builds the React app and serves everything from one port:

```bash
source .venv/bin/activate
./start.sh
```

Open [http://localhost:5000](http://localhost:5000).

## Deploy

1. Set environment variables from `.env.example` on your host.
2. Build frontend: `npm run build`
3. Start with gunicorn (see `Procfile` or `start.sh`).

Health check: `GET /api/health`

## API keys (.env)

| Variable | Required | Purpose |
|----------|----------|---------|
| `ASSEMBLYAI_API_KEY` | Yes | Speech-to-text |
| `HF_API_KEY` | Yes | Summary & quiz (Hugging Face) |
| `HF_MODEL` | No | Default `meta-llama/Meta-Llama-3-8B-Instruct` |
| `PORT` | No | Default `5000` |
| `JOB_WORKERS` | No | Parallel jobs (default `2`) |

## License

MIT (or your choice — update before publishing).
