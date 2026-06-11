import logging
import os
import re
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from urllib.parse import urlparse

import ffmpeg
import requests
import yt_dlp
from dotenv import load_dotenv
from flask import Flask, g, jsonify, request, send_file, send_from_directory
from flask_cors import CORS
from sqlalchemy import Column, Integer, String, Text, create_engine, inspect, text
from sqlalchemy.orm import declarative_base, sessionmaker
from werkzeug.utils import secure_filename

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s %(levelname)-5s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger("neurobrief")
logging.getLogger("werkzeug").setLevel(logging.WARNING)

ALLOWED_VIDEO_EXT = {".mp4", ".webm", ".mkv", ".mov", ".avi", ".m4v"}
MAX_FILE_SIZE = int(os.getenv("MAX_UPLOAD_BYTES", str(100 * 1024 * 1024)))
JOB_WORKERS = max(1, int(os.getenv("JOB_WORKERS", "2")))
HF_API_URL = "https://router.huggingface.co/v1/chat/completions"
HF_MODEL = os.getenv("HF_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
UUID_RE = re.compile(r"^[0-9a-f-]{36}$", re.I)
DOWNLOAD_TYPES = frozenset({"summary", "quiz", "transcript"})
STAGE_LABELS = {
    "extracting": "Extracting audio from video…",
    "downloading": "Downloading audio from YouTube…",
    "transcribing": "Transcribing audio (can take several minutes)…",
    "summarizing": "Generating summary and quiz…",
}


def _path(*parts):
    return os.path.join(BASE_DIR, *parts)


def _static_folder():
    for candidate in (_path("frontend", "build"), _path("build")):
        if os.path.isdir(candidate):
            return candidate
    return _path("build")


UPLOAD_FOLDER = _path(os.getenv("UPLOAD_FOLDER", "uploads"))
SUMMARY_FOLDER = _path(os.getenv("SUMMARY_FOLDER", "summaries"))
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SUMMARY_FOLDER, exist_ok=True)

Base = declarative_base()


class Transcript(Base):
    __tablename__ = "transcripts"
    id = Column(Integer, primary_key=True)
    video_filename = Column(Text)
    transcript_text = Column(Text)
    summary_text = Column(Text)


class Job(Base):
    __tablename__ = "jobs"
    id = Column(String(36), primary_key=True)
    status = Column(String(20), nullable=False, default="pending")
    error = Column(Text, nullable=True)
    summary = Column(Text, nullable=True)
    quiz = Column(Text, nullable=True)
    source_type = Column(String(20), nullable=True)
    source_detail = Column(Text, nullable=True)
    level = Column(String(20), nullable=True)
    stage = Column(String(40), nullable=True)


db_engine = create_engine(
    os.getenv("DATABASE_URL", f"sqlite:///{_path('feedback.db')}"),
    connect_args={"check_same_thread": False},
)
Base.metadata.create_all(db_engine)
_insp = inspect(db_engine)
if _insp.has_table("jobs") and "stage" not in {c["name"] for c in _insp.get_columns("jobs")}:
    with db_engine.begin() as conn:
        conn.execute(text("ALTER TABLE jobs ADD COLUMN stage TEXT"))
SessionLocal = sessionmaker(bind=db_engine)
_job_sem = threading.Semaphore(JOB_WORKERS)


@contextmanager
def db_session():
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _env(key):
    return (os.getenv(key) or "").strip().strip('"').strip("'")


def _unlink(path):
    try:
        if path and os.path.isfile(path):
            os.unlink(path)
    except OSError:
        pass


def _status_msg(status, stage):
    if status == "pending":
        return "Queued…"
    if status == "failed":
        return None
    if status == "completed":
        return "Done!"
    return STAGE_LABELS.get(stage or "", "Processing…")


def _update_job(job_id, **fields):
    with db_session() as session:
        row = session.get(Job, job_id)
        if row:
            for k, v in fields.items():
                setattr(row, k, v)


def _set_stage(job_id, stage):
    _update_job(job_id, status="processing", stage=stage)
    logger.info("[job:%s] %s", job_id[:8], STAGE_LABELS.get(stage, stage))


def _llm(prompt, max_tokens=1024, temperature=0.7, top_p=None):
    api_key = _env("HF_API_KEY")
    if not api_key:
        raise RuntimeError("HF_API_KEY is not set.")
    body = {
        "model": HF_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if top_p is not None:
        body["top_p"] = top_p
    r = requests.post(
        HF_API_URL,
        json=body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        timeout=120,
    )
    if not r.ok:
        raise RuntimeError(f"Hugging Face API error ({r.status_code}): {(r.text or r.reason)[:500]}")
    choices = r.json().get("choices") or []
    text_out = ((choices[0].get("message") or {}).get("content") or choices[0].get("text") or "").strip()
    if not text_out:
        raise RuntimeError("LLM returned an empty response.")
    return text_out


def _transcribe(audio_path):
    api_key = _env("ASSEMBLYAI_API_KEY")
    if not api_key:
        raise RuntimeError("ASSEMBLYAI_API_KEY is not set in the environment.")
    headers = {"authorization": api_key}
    base = "https://api.assemblyai.com"
    with open(audio_path, "rb") as f:
        up = requests.post(f"{base}/v2/upload", headers=headers, data=f, timeout=300)
    if not up.ok:
        raise RuntimeError(f"AssemblyAI upload failed ({up.status_code})")
    audio_url = up.json().get("upload_url")
    models = [m.strip() for m in _env("ASSEMBLYAI_SPEECH_MODELS").split(",") if m.strip()]
    if not models:
        models = ["universal-3-pro", "universal-2"]
    tr = requests.post(
        f"{base}/v2/transcript",
        json={"audio_url": audio_url, "speech_models": models},
        headers=headers,
        timeout=60,
    )
    if not tr.ok:
        raise RuntimeError(f"AssemblyAI transcript failed ({tr.status_code})")
    poll_url = f"{base}/v2/transcript/{tr.json()['id']}"
    while True:
        res = requests.get(poll_url, headers=headers, timeout=60).json()
        if res.get("status") == "completed":
            return res.get("text") or ""
        if res.get("status") == "error":
            raise RuntimeError(f"Transcription failed: {res.get('error', 'unknown')}")
        time.sleep(3)


def _clean_summary(text):
    lines = []
    for line in text.strip().split("\n"):
        s = line.strip()
        if not s:
            lines.append("")
            continue
        if re.match(r"^(here is|below is|this is|summary:)\b", s, re.I):
            continue
        if re.match(r"^(\*\*)?summary(\*\*)?\s*:?\s*$", s, re.I):
            continue
        lines.append(line)
    out = "\n".join(lines).strip()
    return re.sub(r"\n{3,}", "\n\n", out)


def _summarize(text):
    prompt = f"""Write a summary of the content below.

STRICT RULES — follow every rule:
1. Output ONLY 3 paragraphs separated by one blank line. No title, no intro, no outro.
2. Do NOT write phrases like "Here is a summary", "The transcript", "The speaker", or "The video".
3. State ideas directly (e.g. "Anthropic is..." not "The content explains that Anthropic is...").
4. Use only facts present in the content. Do not invent statistics, quotes, or events.
5. Neutral professional tone. No slang or jokes.
6. Do not use markdown headings or bullet lists.

Content:
{text}"""
    raw = _llm(prompt, max_tokens=900, temperature=0.4)
    cleaned = _clean_summary(raw)
    return cleaned or raw.strip()


def _extract_quiz(raw):
    for pattern in (r"(Question Type:\s*.+)", r"(Type:\s*.+)"):
        match = re.search(pattern, raw, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return raw.strip()


def _generate_quiz(summary, level="medium"):
    levels = {"easy": "beginner", "medium": "intermediate", "hard": "advanced"}
    rules = f"""Create a quiz from the summary below.
Difficulty: {levels.get(level, 'intermediate')}.

STRICT RULES — follow every rule:
1. Output ONLY questions in the exact format shown. No intro, no section titles, no markdown (**), no numbering.
2. Include exactly 3 Multiple Choice and 3 True or False questions (6 total).
3. Multiple Choice only: exactly 4 options on one "Options:" line (A) B) C) D)).
4. True or False: no Options line; Answer must be exactly True or False.
5. Use only facts from the summary. Do not invent names, stats, or events.
6. Do not prefix questions with "Multiple Choice:" or "True or False:" — use Question Type: only.

EXACT FORMAT (copy this structure):

Question Type: Multiple Choice
Question: <question text>
Options: A) <option> B) <option> C) <option> D) <option>
Answer: <letter>) <option text>

Question Type: True or False
Question: <statement>
Answer: True

Summary:
{summary}"""
    raw = _llm(rules, max_tokens=900, temperature=0.3, top_p=0.85)
    cleaned = _extract_quiz(raw)
    if not re.search(r"Question Type:", cleaned, re.I):
        retry = _llm(
            rules + "\n\nYour last reply ignored the format. Reply again using ONLY the Question Type / Question / Options / Answer template. No other text.",
            max_tokens=900,
            temperature=0.2,
        )
        cleaned = _extract_quiz(retry)
    return cleaned or "Quiz generation failed."


def _save_results(job_id, label, transcript, summary, quiz):
    for kind, content in (("summary", summary), ("quiz", quiz), ("transcript", transcript)):
        with open(os.path.join(SUMMARY_FOLDER, f"{job_id}_{kind}.txt"), "w") as f:
            f.write(content)
    with db_session() as session:
        job = session.get(Job, job_id)
        if job:
            job.status, job.summary, job.quiz, job.error, job.stage = "completed", summary, quiz, None, None
        session.add(Transcript(video_filename=label, transcript_text=transcript, summary_text=summary))


def _youtube_ok(url):
    try:
        u = urlparse(url.strip())
        return u.scheme in ("http", "https") and (u.hostname or "").lower() in {
            "youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com", "youtu.be", "www.youtu.be"
        }
    except Exception:
        return False


def _download_youtube(url, job_id):
    out = os.path.join(UPLOAD_FOLDER, job_id)
    opts = {
        "format": "bestaudio/best",
        "outtmpl": out + ".%(ext)s",
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "128"}],
        "extractor_args": {"youtube": {"player_client": ["android", "web"]}},
        "quiet": True,
        "no_warnings": True,
    }
    browser = _env("YTDLP_COOKIES_BROWSER")
    if browser:
        opts["cookiesfrombrowser"] = (browser,)
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])
    mp3 = out + ".mp3"
    if not os.path.isfile(mp3):
        raise RuntimeError("YouTube download finished but MP3 was not found.")
    return mp3


def _run_job(app, job_id, level, label, get_audio_path):
    with app.app_context():
        extras = []
        try:
            audio = get_audio_path()
            extras.append(audio)
            _set_stage(job_id, "transcribing")
            transcript = _transcribe(audio)
            _set_stage(job_id, "summarizing")
            summary = _summarize(transcript)
            quiz = _generate_quiz(summary, level if level != "high" else "hard")
            _save_results(job_id, label, transcript, summary, quiz)
            logger.info("[job:%s] Completed", job_id[:8])
        except Exception as e:
            logger.exception("[job:%s] Failed: %s", job_id[:8], e)
            _update_job(job_id, status="failed", error=str(e), stage=None)
        finally:
            for p in extras:
                _unlink(p)


def _start_job(fn, *args):
    if not _job_sem.acquire(blocking=False):
        return False

    def run():
        try:
            fn(*args)
        finally:
            _job_sem.release()

    threading.Thread(target=run, daemon=True).start()
    return True


app = Flask(__name__, static_folder=_static_folder())
CORS(app)


@app.before_request
def _before():
    g._t = time.perf_counter()


@app.after_request
def _after(resp):
    if not (request.method == "GET" and request.path.startswith("/api/jobs/")):
        logger.info("%s %s -> %s (%.0fms)", request.method, request.path, resp.status_code,
                    (time.perf_counter() - g._t) * 1000)
    return resp


@app.route("/api/health")
def health():
    ok = bool(_env("ASSEMBLYAI_API_KEY")) and bool(_env("HF_API_KEY"))
    built = os.path.isfile(os.path.join(app.static_folder or "", "index.html"))
    return jsonify({"status": "ok" if ok and built else "degraded", "frontend_built": built}), 200 if ok else 503


@app.route("/api/jobs/<job_id>")
def get_job(job_id):
    if not UUID_RE.match(job_id):
        return jsonify({"error": "Invalid job id"}), 400
    with db_session() as session:
        row = session.get(Job, job_id)
        if not row:
            return jsonify({"error": "Job not found"}), 404
        stage = None if row.status == "completed" else row.stage
        data = {
            "job_id": row.id, "uid": row.id, "status": row.status, "stage": stage,
            "status_message": _status_msg(row.status, stage), "error": row.error,
        }
        if row.status == "completed":
            data["summary"], data["quiz"] = row.summary or "", row.quiz or ""
        resp = jsonify(data)
        resp.headers["Cache-Control"] = "no-store"
        return resp


@app.route("/process", methods=["POST"])
def process():
    url = (request.form.get("youtube_url") or "").strip()
    f = request.files.get("video_file")
    if f and not f.filename:
        f = None
    if f and url:
        return jsonify({"error": "Send either a video file or a YouTube URL, not both."}), 400
    if not f and not url:
        return jsonify({"error": "Provide a video file or a youtube_url."}), 400

    level = (request.form.get("level") or "medium").lower()
    if level == "high":
        level = "hard"
    job_id = str(uuid.uuid4())

    if f:
        ext = os.path.splitext(f.filename)[1].lower()
        if ext not in ALLOWED_VIDEO_EXT:
            return jsonify({"error": f"Unsupported file type: {ext}"}), 400
        video_path = os.path.join(UPLOAD_FOLDER, f"{job_id}_{secure_filename(f.filename) or 'video' + ext}")
        f.save(video_path)
        if os.path.getsize(video_path) > MAX_FILE_SIZE:
            _unlink(video_path)
            return jsonify({"error": "File too large"}), 400
        with db_session() as s:
            s.add(Job(id=job_id, status="pending", source_type="upload", source_detail=f.filename, level=level))

        def get_audio():
            _set_stage(job_id, "extracting")
            audio = os.path.join(UPLOAD_FOLDER, f"{job_id}.mp3")
            ffmpeg.input(video_path).output(audio, format="mp3", acodec="libmp3lame", ac=1, ar="16k").overwrite_output().run(quiet=True)
            _unlink(video_path)
            return audio

        if not _start_job(_run_job, app, job_id, level, f.filename, get_audio):
            _unlink(video_path)
            _update_job(job_id, status="failed", error="Server busy.")
            return jsonify({"error": "Server is busy. Try again shortly."}), 503
    else:
        if not _youtube_ok(url):
            return jsonify({"error": "Invalid YouTube URL."}), 400
        with db_session() as s:
            s.add(Job(id=job_id, status="pending", source_type="youtube", source_detail=url, level=level))

        def get_audio():
            _set_stage(job_id, "downloading")
            return _download_youtube(url, job_id)

        if not _start_job(_run_job, app, job_id, level, url, get_audio):
            _update_job(job_id, status="failed", error="Server busy.")
            return jsonify({"error": "Server is busy. Try again shortly."}), 503

    return jsonify({"job_id": job_id, "uid": job_id, "status": "pending", "status_message": "Queued…"})


@app.route("/download/<uid>/<filetype>")
def download(uid, filetype):
    if not UUID_RE.match(uid) or filetype not in DOWNLOAD_TYPES:
        return jsonify({"error": "Not found"}), 404
    path = os.path.join(SUMMARY_FOLDER, f"{uid}_{filetype}.txt")
    if not os.path.isfile(path):
        return jsonify({"error": "File not found"}), 404
    return send_file(path, as_attachment=True, download_name=f"{filetype}.txt", mimetype="text/plain")


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve(path):
    root = app.static_folder
    if path.startswith("api/"):
        return jsonify({"error": "Not found"}), 404
    if path and root and os.path.exists(os.path.join(root, path)):
        return send_from_directory(root, path)
    if root and os.path.isfile(os.path.join(root, "index.html")):
        return send_from_directory(root, "index.html")
    return jsonify({"error": "Frontend build missing. Run: cd frontend && npm run build"}), 503


logger.info("NeuroBrief starting — UI: %s", app.static_folder)
if not _env("ASSEMBLYAI_API_KEY"):
    logger.warning("ASSEMBLYAI_API_KEY not set")
if not _env("HF_API_KEY"):
    logger.warning("HF_API_KEY not set")

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug, host="0.0.0.0", port=port, threaded=True, use_reloader=os.getenv("FLASK_USE_RELOADER", "0") == "1")
