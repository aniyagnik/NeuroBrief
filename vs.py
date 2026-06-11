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
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from sqlalchemy import Column, String, Text, create_engine, inspect, text
from sqlalchemy.orm import declarative_base, sessionmaker
from werkzeug.utils import secure_filename

BASE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE, ".env"))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout, force=True)
log = logging.getLogger("neurobrief")

UPLOAD = os.path.join(BASE, os.getenv("UPLOAD_FOLDER", "uploads"))
SUMMARY = os.path.join(BASE, os.getenv("SUMMARY_FOLDER", "summaries"))
STATIC = next((p for p in (os.path.join(BASE, "frontend", "build"), os.path.join(BASE, "build")) if os.path.isdir(p)), os.path.join(BASE, "build"))
os.makedirs(UPLOAD, exist_ok=True)
os.makedirs(SUMMARY, exist_ok=True)

HF_URL = "https://router.huggingface.co/v1/chat/completions"
HF_MODEL = os.getenv("HF_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
VIDEO_EXT = {".mp4", ".webm", ".mkv", ".mov", ".avi", ".m4v"}
MAX_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(100 * 1024 * 1024)))
UUID = re.compile(r"^[0-9a-f-]{36}$", re.I)
STAGES = {
    "extracting": "Extracting audio…",
    "downloading": "Downloading from YouTube…",
    "transcribing": "Transcribing…",
    "summarizing": "Generating summary and quiz…",
}

Base = declarative_base()


class Job(Base):
    __tablename__ = "jobs"
    id = Column(String(36), primary_key=True)
    status = Column(String(20), default="pending")
    error = Column(Text)
    summary = Column(Text)
    quiz = Column(Text)
    stage = Column(String(40))
    level = Column(String(20))


engine = create_engine(os.getenv("DATABASE_URL", f"sqlite:///{os.path.join(BASE, 'feedback.db')}"), connect_args={"check_same_thread": False})
Base.metadata.create_all(engine)
if inspect(engine).has_table("jobs") and "stage" not in {c["name"] for c in inspect(engine).get_columns("jobs")}:
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE jobs ADD COLUMN stage TEXT"))
Session = sessionmaker(bind=engine)
sem = threading.Semaphore(max(1, int(os.getenv("JOB_WORKERS", "2"))))


@contextmanager
def db():
    s = Session()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


def env(k):
    return (os.getenv(k) or "").strip().strip('"').strip("'")


def llm(prompt, tokens=900, temp=0.4):
    key = env("HF_API_KEY")
    if not key:
        raise RuntimeError("HF_API_KEY is not set.")
    r = requests.post(HF_URL, json={"model": HF_MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": tokens, "temperature": temp},
                      headers={"Authorization": f"Bearer {key}"}, timeout=120)
    if not r.ok:
        raise RuntimeError(f"LLM error {r.status_code}: {(r.text or '')[:300]}")
    out = ((r.json().get("choices") or [{}])[0].get("message") or {}).get("content", "").strip()
    if not out:
        raise RuntimeError("Empty LLM response.")
    return out


def transcribe(path):
    key = env("ASSEMBLYAI_API_KEY")
    if not key:
        raise RuntimeError("ASSEMBLYAI_API_KEY is not set.")
    h = {"authorization": key}
    with open(path, "rb") as f:
        up = requests.post("https://api.assemblyai.com/v2/upload", headers=h, data=f, timeout=300)
    if not up.ok:
        raise RuntimeError("AssemblyAI upload failed.")
    models = [m.strip() for m in env("ASSEMBLYAI_SPEECH_MODELS").split(",") if m.strip()] or ["universal-3-pro", "universal-2"]
    tr = requests.post("https://api.assemblyai.com/v2/transcript", json={"audio_url": up.json()["upload_url"], "speech_models": models}, headers=h, timeout=60)
    if not tr.ok:
        raise RuntimeError("AssemblyAI transcript failed.")
    url = f"https://api.assemblyai.com/v2/transcript/{tr.json()['id']}"
    while True:
        res = requests.get(url, headers=h, timeout=60).json()
        if res.get("status") == "completed":
            return res.get("text") or ""
        if res.get("status") == "error":
            raise RuntimeError(res.get("error", "Transcription failed"))
        time.sleep(3)


def summarize(text):
    raw = llm(
        "Write exactly 3 paragraphs (blank line between each). No title or intro. "
        "No words: transcript, speaker, video. Only facts from the text. Professional tone.\n\n" + text,
        temp=0.4,
    )
    lines = [ln for ln in raw.split("\n") if not re.match(r"^(here is|below is|this is|summary:)", ln.strip(), re.I)]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines).strip())


QUIZ_FMT = """Output ONLY 6 questions (3 MCQ + 3 True/False) in this exact format. No intro, markdown, or numbering.

Question Type: Multiple Choice
Question: ...
Options: A) ... B) ... C) ... D) ...
Answer: B) ...

Question Type: True or False
Question: ...
Answer: True

Use only facts from this summary:
"""


def make_quiz(summary, level="medium"):
    prompt = QUIZ_FMT + summary + f"\nDifficulty: {level}."
    raw = llm(prompt, tokens=900, temp=0.3)
    m = re.search(r"(Question Type:.*)", raw, re.I | re.S)
    text_out = (m.group(1) if m else raw).strip()
    if not re.search(r"Question Type:", text_out, re.I):
        raw = llm(prompt + "\nUse ONLY the Question Type / Question / Options / Answer format.", tokens=900, temp=0.2)
        m = re.search(r"(Question Type:.*)", raw, re.I | re.S)
        text_out = (m.group(1) if m else raw).strip()
    return text_out or "Quiz generation failed."


def set_job(job_id, **kw):
    with db() as s:
        row = s.get(Job, job_id)
        if row:
            for k, v in kw.items():
                setattr(row, k, v)


def run_pipeline(app, job_id, level, label, audio_fn):
    with app.app_context():
        audio = None
        try:
            audio = audio_fn()
            set_job(job_id, status="processing", stage="transcribing")
            log.info("[%s] transcribing", job_id[:8])
            transcript = transcribe(audio)
            set_job(job_id, stage="summarizing")
            summary = summarize(transcript)
            quiz = make_quiz(summary, "hard" if level == "high" else level)
            for kind, body in (("summary", summary), ("quiz", quiz), ("transcript", transcript)):
                open(os.path.join(SUMMARY, f"{job_id}_{kind}.txt"), "w").write(body)
            set_job(job_id, status="completed", summary=summary, quiz=quiz, error=None, stage=None)
            log.info("[%s] done", job_id[:8])
        except Exception as e:
            log.exception("[%s] failed", job_id[:8])
            set_job(job_id, status="failed", error=str(e), stage=None)
        finally:
            if audio and os.path.isfile(audio):
                os.unlink(audio)


def start_job(fn, *args):
    if not sem.acquire(blocking=False):
        return False

    def run():
        try:
            fn(*args)
        finally:
            sem.release()

    threading.Thread(target=run, daemon=True).start()
    return True


app = Flask(__name__, static_folder=STATIC)
CORS(app)


@app.route("/api/jobs/<job_id>")
def job_status(job_id):
    if not UUID.match(job_id):
        return jsonify({"error": "Invalid job id"}), 400
    with db() as s:
        row = s.get(Job, job_id)
        if not row:
            return jsonify({"error": "Not found"}), 404
        stage = None if row.status == "completed" else row.stage
        msg = "Queued…" if row.status == "pending" else ("Done!" if row.status == "completed" else STAGES.get(stage or "", "Processing…"))
        data = {"job_id": row.id, "uid": row.id, "status": row.status, "stage": stage, "status_message": msg, "error": row.error}
        if row.status == "completed":
            data["summary"], data["quiz"] = row.summary or "", row.quiz or ""
        r = jsonify(data)
        r.headers["Cache-Control"] = "no-store"
        return r


@app.route("/process", methods=["POST"])
def process():
    url = (request.form.get("youtube_url") or "").strip()
    f = request.files.get("video_file")
    if f and not f.filename:
        f = None
    if bool(f) == bool(url):
        return jsonify({"error": "Send a video file OR a YouTube URL."}), 400

    level = (request.form.get("level") or "medium").lower()
    job_id = str(uuid.uuid4())
    with db() as s:
        s.add(Job(id=job_id, status="pending", level=level))

    if f:
        ext = os.path.splitext(f.filename)[1].lower()
        if ext not in VIDEO_EXT:
            return jsonify({"error": "Unsupported file type."}), 400
        video = os.path.join(UPLOAD, f"{job_id}_{secure_filename(f.filename)}")
        f.save(video)
        if os.path.getsize(video) > MAX_BYTES:
            os.unlink(video)
            return jsonify({"error": "File too large"}), 400

        def audio_fn():
            set_job(job_id, status="processing", stage="extracting")
            out = os.path.join(UPLOAD, f"{job_id}.mp3")
            ffmpeg.input(video).output(out, format="mp3", acodec="libmp3lame", ac=1, ar="16k").overwrite_output().run(quiet=True)
            os.unlink(video)
            return out

        if not start_job(run_pipeline, app, job_id, level, f.filename, audio_fn):
            os.unlink(video)
            set_job(job_id, status="failed", error="Server busy.")
            return jsonify({"error": "Server busy."}), 503
    else:
        host = urlparse(url).hostname or ""
        if host.lower() not in {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be", "www.youtu.be"}:
            return jsonify({"error": "Invalid YouTube URL."}), 400

        def audio_fn():
            set_job(job_id, status="processing", stage="downloading")
            out = os.path.join(UPLOAD, job_id)
            opts = {"format": "bestaudio/best", "outtmpl": out + ".%(ext)s",
                    "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3"}],
                    "quiet": True, "no_warnings": True}
            if env("YTDLP_COOKIES_BROWSER"):
                opts["cookiesfrombrowser"] = (env("YTDLP_COOKIES_BROWSER"),)
            yt_dlp.YoutubeDL(opts).download([url])
            mp3 = out + ".mp3"
            if not os.path.isfile(mp3):
                raise RuntimeError("YouTube download failed.")
            return mp3

        if not start_job(run_pipeline, app, job_id, level, url, audio_fn):
            set_job(job_id, status="failed", error="Server busy.")
            return jsonify({"error": "Server busy."}), 503

    return jsonify({"job_id": job_id, "uid": job_id, "status": "pending", "status_message": "Queued…"})


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def spa(path):
    if path.startswith("api/"):
        return jsonify({"error": "Not found"}), 404
    if path and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    if os.path.isfile(os.path.join(app.static_folder, "index.html")):
        return send_from_directory(app.static_folder, "index.html")
    return jsonify({"error": "Run: cd frontend && npm run build"}), 503


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=os.getenv("FLASK_DEBUG") == "1", threaded=True)
