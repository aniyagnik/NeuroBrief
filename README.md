# NeuroBrief

Upload a video or paste a YouTube link to get an AI-generated **summary**, **quiz**, and **transcript**.

Single repo: React frontend + Flask backend. In production the backend serves the built UI from `frontend/build/`.

## Project layout

```
.
├── frontend/src/App.js   # React UI + quiz
├── vs.py                 # Flask API + pipeline
├── requirements.txt    # Python dependencies
├── start.sh            # Local production start
├── Dockerfile          # Render / Docker deploy (ffmpeg + UI build)
├── render.yaml           # Render Blueprint
├── Procfile              # Heroku / Railway
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

## Deploy on Render (recommended)

The repo includes a **Dockerfile** (ffmpeg + built React UI) and **`render.yaml`** for one-click deploy.

### Option A — Blueprint (easiest)

1. Push this repo to GitHub (do **not** commit `.env`).
2. Go to [Render Dashboard](https://dashboard.render.com/) → **New** → **Blueprint**.
3. Connect your GitHub repo — Render reads `render.yaml`.
4. When prompted, set:
   - `ASSEMBLYAI_API_KEY`
   - `HF_API_KEY`
5. Click **Apply**. First deploy takes ~5–10 minutes.

### Option B — Manual Web Service

1. **New** → **Web Service** → connect repo.
2. **Runtime:** Docker
3. **Health check path:** `/api/health`
4. Add environment variables from `.env.example` (at minimum `ASSEMBLYAI_API_KEY`, `HF_API_KEY`).
5. Deploy.

Your app will be live at `https://<service-name>.onrender.com` (UI + API on one URL).

### Render notes

| Topic | Detail |
|-------|--------|
| Free tier | Service sleeps after ~15 min idle; first request may be slow |
| Job status | Stored in memory — lost if the instance restarts |
| Uploads | Ephemeral disk — fine per request, not long-term storage |
| YouTube | Often **blocked on cloud IPs** — use **file upload**, or add `YTDLP_COOKIES` (see below) |
| File uploads | Supported (ffmpeg is included in the Docker image) |

### YouTube blocked on Render (`Sign in to confirm you're not a bot`)

YouTube frequently blocks datacenter IPs (Render, Railway, etc.). **Easiest fix: upload the video file** instead of pasting a URL.

To try YouTube links on Render:

1. Install **"Get cookies.txt LOCALLY"** in Chrome/Firefox.
2. Open [youtube.com](https://www.youtube.com) while logged in → export cookies (Netscape `.txt` format).
3. On your computer, base64-encode the file (recommended — Render handles one line better):
   ```bash
   base64 -w0 youtube_cookies.txt
   ```
4. Render Dashboard → your service → **Environment** → **Add variable**
   - **Key:** `YTDLP_COOKIES_BASE64`
   - **Value:** paste the long base64 string from step 3
5. **Save** → **Manual Deploy** (must redeploy after env changes).
6. Check: `curl https://YOUR-APP.onrender.com/api/health` → `"youtube_cookies_configured": true`

Alternative: set **`YTDLP_COOKIES`** to the raw file contents (multiline). Base64 is more reliable on Render.

Locally use `YTDLP_COOKIES_BROWSER=chrome` in `.env` instead.

Verify deploy: `curl https://<your-app>.onrender.com/api/health`

### Troubleshooting `frontend_built: false`

This means the React UI was **not** built into the server. Common cause: the service was created as **Python** instead of **Docker**.

**Fix:**

1. Render Dashboard → your service → **Settings**
2. Under **Build & Deploy**, set **Runtime** to **Docker** (not Python)
3. **Dockerfile path:** `./Dockerfile`
4. **Docker build context:** `.` (repo root)
5. **Manual Deploy** → **Clear build cache & deploy**

After redeploy, `/api/health` should show `"frontend_built": true` and `"runtime": "docker"`.

If you still see `"runtime": "native"`, Docker is not being used.

### `argument list too long` during Docker build

Render sends your repo to Docker as a **build context**. If `node_modules`, `.venv`, or video files were ever committed, the context becomes huge and the build fails.

**Fix (already in repo):** `.dockerignore` uses a **whitelist** — only `vs.py`, `requirements.txt`, and frontend source are sent to Docker.

**Also clean Git** (run once locally, then push):

```bash
git rm -r --cached frontend/node_modules node_modules .venv uploads 2>/dev/null || true
git commit -m "Stop tracking node_modules and local artifacts"
git push
```

Then **Manual Deploy → Clear build cache & deploy** on Render.

## Deploy (other platforms)

1. Set environment variables from `.env.example` on your host.
2. Build frontend: `npm run build`
3. Start with gunicorn (see `Procfile`, `start.sh`, or `Dockerfile`).

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
