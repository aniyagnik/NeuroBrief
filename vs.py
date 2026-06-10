import logging
import os
import re
import sys
import uuid
import time
import threading
import requests
from contextlib import contextmanager
from urllib.parse import urlparse

import yt_dlp
from flask import Flask, request, send_file, send_from_directory, jsonify, g
from werkzeug.utils import secure_filename
import ffmpeg
from sqlalchemy import create_engine, Column, Integer, Text, String, inspect, text
from sqlalchemy.orm import declarative_base, sessionmaker
from flask_cors import CORS
from dotenv import load_dotenv
from werkzeug.datastructures import FileStorage

# ========== Setup ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FORMAT = os.getenv(
    "LOG_FORMAT",
    "%(asctime)s %(levelname)-5s [%(name)s] %(message)s",
)
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format=LOG_FORMAT,
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger("neurobrief")
# Quiet noisy third-party loggers in production.
logging.getLogger("werkzeug").setLevel(logging.WARNING)

ALLOWED_VIDEO_EXT = {".mp4", ".webm", ".mkv", ".mov", ".avi", ".m4v"}
MAX_FILE_SIZE = int(os.getenv("MAX_UPLOAD_BYTES", str(100 * 1024 * 1024)))
JOB_WORKERS = max(1, int(os.getenv("JOB_WORKERS", "2")))
TOGETHER_API_URL = "https://api.together.ai/v1/chat/completions"
TOGETHER_MODEL = os.getenv("TOGETHER_MODEL", "mistralai/Mixtral-8x7B-Instruct-v0.1")
HF_API_URL = "https://router.huggingface.co/v1/chat/completions"
HF_MODEL = os.getenv("HF_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto").lower()
UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I
)
DOWNLOAD_TYPES = frozenset({"summary", "quiz", "transcript"})
LEVEL_ALIASES = {"high": "hard"}
_job_semaphore = threading.Semaphore(JOB_WORKERS)


def _abs_path(*parts):
    return os.path.join(BASE_DIR, *parts)


def _resolve_static_folder():
    for candidate in (_abs_path("frontend", "build"), _abs_path("build")):
        if os.path.isdir(candidate):
            return candidate
    return _abs_path("build")


app = Flask(__name__, static_folder=_resolve_static_folder())
CORS(app)

UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", _abs_path("uploads"))
SUMMARY_FOLDER = os.getenv("SUMMARY_FOLDER", _abs_path("summaries"))
if not os.path.isabs(UPLOAD_FOLDER):
    UPLOAD_FOLDER = _abs_path(UPLOAD_FOLDER)
if not os.path.isabs(SUMMARY_FOLDER):
    SUMMARY_FOLDER = _abs_path(SUMMARY_FOLDER)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SUMMARY_FOLDER, exist_ok=True)

# ========== Database Setup ==========
Base = declarative_base()


class Feedback(Base):
    __tablename__ = "feedback"
    id = Column(Integer, primary_key=True)
    summary = Column(Text)
    quiz = Column(Text)
    feedback = Column(Text)


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


_db_url = os.getenv("DATABASE_URL", f"sqlite:///{_abs_path('feedback.db')}")
db_engine = create_engine(
    _db_url,
    connect_args={"check_same_thread": False},
    pool_pre_ping=True,
)
Base.metadata.create_all(db_engine)


def _migrate_schema():
    insp = inspect(db_engine)
    if insp.has_table("jobs"):
        cols = {c["name"] for c in insp.get_columns("jobs")}
        if "stage" not in cols:
            with db_engine.begin() as conn:
                conn.execute(text("ALTER TABLE jobs ADD COLUMN stage TEXT"))


_migrate_schema()
SessionLocal = sessionmaker(bind=db_engine)


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


def _safe_unlink(path):
    try:
        if path and os.path.isfile(path):
            os.unlink(path)
    except OSError:
        pass


def _safe_glob_unlink(prefix_without_ext):
    """Remove job_id.* files under uploads (yt-dlp may leave multiple extensions)."""
    base = os.path.basename(prefix_without_ext)
    folder = os.path.dirname(prefix_without_ext) or UPLOAD_FOLDER
    try:
        for name in os.listdir(folder):
            if name.startswith(base + ".") or name == base:
                _safe_unlink(os.path.join(folder, name))
    except OSError:
        pass


def _normalize_level(level):
    if not level:
        return "medium"
    return LEVEL_ALIASES.get(level.lower(), level.lower())


def _is_allowed_youtube_url(url: str) -> bool:
    try:
        u = urlparse(url.strip())
        if u.scheme not in ("http", "https"):
            return False
        host = (u.hostname or "").lower()
        allowed = {
            "youtube.com",
            "www.youtube.com",
            "m.youtube.com",
            "music.youtube.com",
            "youtu.be",
            "www.youtu.be",
        }
        return host in allowed
    except Exception:
        return False


class LLMError(RuntimeError):
    pass


STAGE_LABELS = {
    "extracting": "Extracting audio from video…",
    "downloading": "Downloading audio from YouTube…",
    "transcribing": "Transcribing audio (can take several minutes)…",
    "summarizing": "Generating summary and quiz…",
}


def _clean_env(key):
    return (os.getenv(key) or "").strip().strip('"').strip("'")


def _parse_chat_response(result):
    choices = result.get("choices") or []
    if not choices:
        raise LLMError("LLM returned no choices.")
    msg = choices[0].get("message") or {}
    text = (msg.get("content") or "").strip()
    if not text:
        text = (choices[0].get("text") or "").strip()
    if not text:
        raise LLMError("LLM returned an empty response.")
    return text


def _together_chat(user_content: str, max_tokens: int, temperature: float, top_p=None):
    api_key = _clean_env("TOGETHER_API_KEY")
    if not api_key:
        raise LLMError("TOGETHER_API_KEY is not set.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": TOGETHER_MODEL,
        "messages": [{"role": "user", "content": user_content}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if top_p is not None:
        body["top_p"] = top_p

    response = requests.post(
        TOGETHER_API_URL, json=body, headers=headers, timeout=120
    )
    if not response.ok:
        detail = response.text[:500] if response.text else response.reason
        raise LLMError(f"Together API error ({response.status_code}): {detail}")
    return _parse_chat_response(response.json())


def _hf_chat(user_content: str, max_tokens: int, temperature: float, top_p=None):
    api_key = _clean_env("HF_API_KEY")
    if not api_key:
        raise LLMError("HF_API_KEY is not set.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": HF_MODEL,
        "messages": [{"role": "user", "content": user_content}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if top_p is not None:
        body["top_p"] = top_p

    response = requests.post(HF_API_URL, json=body, headers=headers, timeout=120)
    if not response.ok:
        detail = response.text[:500] if response.text else response.reason
        raise LLMError(f"Hugging Face API error ({response.status_code}): {detail}")
    return _parse_chat_response(response.json())


def _llm_chat(user_content: str, max_tokens: int, temperature: float, top_p=None):
    """Call configured LLM provider(s). auto = Together first, then Hugging Face."""
    provider = LLM_PROVIDER
    errors = []
    logger.info("LLM request provider=%s max_tokens=%d", provider, max_tokens)

    if provider in ("auto", "together"):
        try:
            result = _together_chat(user_content, max_tokens, temperature, top_p)
            logger.info("LLM response ok provider=together chars=%d", len(result))
            return result
        except LLMError as e:
            errors.append(str(e))
            logger.warning("Together LLM failed: %s", e)
            if provider == "together":
                raise

    if provider in ("auto", "huggingface", "hf"):
        try:
            result = _hf_chat(user_content, max_tokens, temperature, top_p)
            logger.info("LLM response ok provider=huggingface chars=%d", len(result))
            return result
        except LLMError as e:
            errors.append(str(e))
            logger.warning("Hugging Face LLM failed: %s", e)
            raise

    raise LLMError(
        errors[0]
        if errors
        else "No LLM provider configured. Set TOGETHER_API_KEY or HF_API_KEY."
    )


def _status_message(status, stage):
    if status == "pending":
        return "Queued…"
    if status == "failed":
        return None
    if status == "completed":
        return "Done!"
    if stage and stage in STAGE_LABELS:
        return STAGE_LABELS[stage]
    return "Processing…"


def _assemblyai_request_error(response, step: str):
    try:
        body = response.json()
        detail = body.get("error") or body.get("message") or body
    except Exception:
        detail = response.text or response.reason
    raise RuntimeError(
        f"AssemblyAI {step} failed ({response.status_code}): {detail}"
    )


def transcribe_audio_via_api(audio_path):
    api_key = (os.getenv("ASSEMBLYAI_API_KEY") or "").strip().strip('"').strip("'")
    if not api_key:
        raise RuntimeError("ASSEMBLYAI_API_KEY is not set in the environment.")

    if not os.path.isfile(audio_path) or os.path.getsize(audio_path) == 0:
        raise RuntimeError("Audio file is missing or empty before transcription.")

    models_raw = os.getenv(
        "ASSEMBLYAI_SPEECH_MODELS", "universal-3-pro,universal-2"
    )
    speech_models = [m.strip() for m in models_raw.split(",") if m.strip()]

    base_url = "https://api.assemblyai.com"
    headers = {"authorization": api_key}

    with open(audio_path, "rb") as f:
        response = requests.post(
            base_url + "/v2/upload", headers=headers, data=f, timeout=300
        )
    if not response.ok:
        _assemblyai_request_error(response, "upload")
    upload_payload = response.json()
    audio_url = upload_payload.get("upload_url")
    if not audio_url:
        raise RuntimeError(f"AssemblyAI upload failed: {upload_payload}")

    data = {
        "audio_url": audio_url,
        "speech_models": speech_models,
    }
    response = requests.post(
        base_url + "/v2/transcript", json=data, headers=headers, timeout=60
    )
    if not response.ok:
        _assemblyai_request_error(response, "transcript")
    transcript_id = response.json()["id"]
    polling_endpoint = base_url + "/v2/transcript/" + transcript_id
    logger.info("AssemblyAI transcript submitted id=%s", transcript_id)

    poll_count = 0
    while True:
        response = requests.get(polling_endpoint, headers=headers, timeout=60)
        if not response.ok:
            _assemblyai_request_error(response, "transcript status")
        transcription_result = response.json()
        status = transcription_result.get("status")
        poll_count += 1
        if poll_count == 1 or poll_count % 10 == 0:
            logger.info("AssemblyAI poll #%d status=%s id=%s", poll_count, status, transcript_id)
        if status == "completed":
            text_len = len(transcription_result.get("text") or "")
            logger.info(
                "AssemblyAI transcription done id=%s chars=%d polls=%d",
                transcript_id,
                text_len,
                poll_count,
            )
            return transcription_result.get("text") or ""
        if status == "error":
            err = transcription_result.get("error", "unknown error")
            raise RuntimeError(f"Transcription failed: {err}")
        time.sleep(3)


def summarize_long_text(text):
    prompt = f"""
        You are an intelligent assistant. Summarize the following transcript:

        Transcript:
        \"\"\"
        {text}
        \"\"\"

        Respond with:
        Summary:
        <paragraph here>
        ...
    """
    return _llm_chat(prompt.strip(), max_tokens=1024, temperature=0.7)


def generate_quiz(summary, level="medium"):
    level_prompt = {
        "easy": "Generate simple quiz questions suitable for beginners.",
        "medium": "Generate moderately difficult quiz questions.",
        "hard": "Generate challenging quiz questions requiring deeper understanding.",
    }
    prompt = f"""
        Based on the following summary, generate a quiz with 3 different types of questions:
        1. Multiple Choice (4 options + correct answer)
        2. True or False (with correct answer)
        3. Fill in the Blanks (with answer)

        {level_prompt.get(level, '')}

        Summary:
        {summary}

        Use this format only:
        Question Type: <Type>
        Question: <Text>
        Options: <A, B, C, D> (for MCQ only)
        Answer: <Answer>
    """

    raw = _llm_chat(prompt.strip(), max_tokens=512, temperature=0.7, top_p=0.9)
    match = re.search(r"(Question Type:.*)$", raw, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    cleaned = raw.strip()
    if cleaned:
        logger.warning("Quiz format parse miss — returning raw LLM output (%d chars)", len(cleaned))
        return cleaned
    return "Quiz generation returned an empty response."


def _update_job(job_id, **fields):
    with db_session() as session:
        row = session.get(Job, job_id)
        if row:
            for k, v in fields.items():
                setattr(row, k, v)


def _job_log(job_id, level, msg, *args, exc_info=False):
    logging.getLogger("neurobrief.jobs").log(
        level, "[job:%s] " + msg, job_id[:8], *args, exc_info=exc_info
    )


def _set_stage(job_id, stage):
    _update_job(job_id, status="processing", stage=stage)
    label = STAGE_LABELS.get(stage, stage)
    _job_log(job_id, logging.INFO, label)


def _create_job(job_id, source_type, source_detail, level):
    with db_session() as session:
        session.add(
            Job(
                id=job_id,
                status="pending",
                source_type=source_type,
                source_detail=source_detail,
                level=level,
            )
        )


def _finalize_success(job_id, video_label, transcript, summary, quiz):
    uid = job_id
    with open(os.path.join(SUMMARY_FOLDER, f"{uid}_summary.txt"), "w") as sf:
        sf.write(summary)
    with open(os.path.join(SUMMARY_FOLDER, f"{uid}_quiz.txt"), "w") as qf:
        qf.write(quiz)
    with open(os.path.join(SUMMARY_FOLDER, f"{uid}_transcript.txt"), "w") as tf:
        tf.write(transcript)

    with db_session() as session:
        job = session.get(Job, job_id)
        if job:
            job.status = "completed"
            job.summary = summary
            job.quiz = quiz
            job.error = None
            job.stage = None
        session.add(
            Transcript(
                video_filename=video_label,
                transcript_text=transcript,
                summary_text=summary,
            )
        )


def download_youtube_audio_mp3(url: str, job_id: str) -> str:
    """Download audio from YouTube and return path to the mp3 file."""
    out_base = os.path.join(UPLOAD_FOLDER, job_id)
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": out_base + ".%(ext)s",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "128",
            }
        ],
        "extractor_args": {"youtube": {"player_client": ["android", "web"]}},
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
    }
    cookies_file = os.getenv("YTDLP_COOKIES_FILE", "").strip()
    cookies_browser = os.getenv("YTDLP_COOKIES_BROWSER", "").strip()
    if cookies_file and os.path.isfile(cookies_file):
        ydl_opts["cookiefile"] = cookies_file
    elif cookies_browser:
        ydl_opts["cookiesfrombrowser"] = (cookies_browser,)

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except yt_dlp.utils.DownloadError as e:
        msg = str(e)
        if "Sign in to confirm" in msg or "not a bot" in msg:
            raise RuntimeError(
                "YouTube blocked the download. In .env set "
                "YTDLP_COOKIES_BROWSER=chrome (or firefox), be logged into "
                "YouTube in that browser, restart the server — or upload the video file."
            ) from e
        raise RuntimeError(f"YouTube download failed: {msg}") from e

    mp3_path = out_base + ".mp3"
    if os.path.isfile(mp3_path):
        return mp3_path
    raise RuntimeError(
        "YouTube download finished but MP3 was not found (is ffmpeg installed?)."
    )


def _run_file_job(app, job_id, video_path, level, source_label):
    with app.app_context():
        audio_path = os.path.join(UPLOAD_FOLDER, f"{job_id}.mp3")
        try:
            _set_stage(job_id, "extracting")
            (
                ffmpeg.input(video_path)
                .output(audio_path, format="mp3", acodec="libmp3lame", ac=1, ar="16k")
                .overwrite_output()
                .run(quiet=True)
            )
            _set_stage(job_id, "transcribing")
            transcript = transcribe_audio_via_api(audio_path)
            _set_stage(job_id, "summarizing")
            summary = summarize_long_text(transcript)
            quiz = generate_quiz(summary, level)
            _finalize_success(job_id, source_label, transcript, summary, quiz)
            _job_log(job_id, logging.INFO, "Completed successfully.")
        except Exception as e:
            _job_log(job_id, logging.ERROR, "Failed: %s", e, exc_info=True)
            _update_job(job_id, status="failed", error=str(e), stage=None)
        finally:
            _safe_unlink(video_path)
            _safe_unlink(audio_path)


def _run_youtube_job(app, job_id, youtube_url, level):
    with app.app_context():
        out_base = os.path.join(UPLOAD_FOLDER, job_id)
        try:
            _set_stage(job_id, "downloading")
            download_youtube_audio_mp3(youtube_url, job_id)
            _set_stage(job_id, "transcribing")
            audio_path = os.path.join(UPLOAD_FOLDER, f"{job_id}.mp3")
            transcript = transcribe_audio_via_api(audio_path)
            _set_stage(job_id, "summarizing")
            summary = summarize_long_text(transcript)
            quiz = generate_quiz(summary, level)
            _finalize_success(job_id, youtube_url, transcript, summary, quiz)
            _job_log(job_id, logging.INFO, "Completed successfully.")
        except Exception as e:
            _job_log(job_id, logging.ERROR, "Failed: %s", e, exc_info=True)
            _update_job(job_id, status="failed", error=str(e), stage=None)
        finally:
            _safe_glob_unlink(out_base)


def _start_background_job(target, *args):
    """Run a job in a daemon thread, limited by JOB_WORKERS semaphore."""

    def _wrapped():
        _job_log(args[1], logging.INFO, "Worker acquired — starting pipeline")
        try:
            target(*args)
        finally:
            _job_semaphore.release()
            _job_log(args[1], logging.DEBUG, "Worker released")

    if not _job_semaphore.acquire(blocking=False):
        return False
    threading.Thread(target=_wrapped, daemon=True, name=f"job-{args[1][:8]}").start()
    return True


def _config_status():
    together = bool(_clean_env("TOGETHER_API_KEY"))
    hf = bool(_clean_env("HF_API_KEY"))
    assembly = bool(_clean_env("ASSEMBLYAI_API_KEY"))
    static_ok = bool(
        app.static_folder
        and os.path.isfile(os.path.join(app.static_folder, "index.html"))
    )
    return {
        "assemblyai_configured": assembly,
        "llm_configured": together or hf,
        "llm_provider": LLM_PROVIDER,
        "frontend_built": static_ok,
        "static_folder": app.static_folder,
        "upload_folder": UPLOAD_FOLDER,
        "summary_folder": SUMMARY_FOLDER,
        "max_upload_mb": round(MAX_FILE_SIZE / (1024 * 1024), 1),
        "job_workers": JOB_WORKERS,
    }


def _log_startup():
    cfg = _config_status()
    logger.info("NeuroBrief backend starting")
    logger.info("Static UI: %s (built=%s)", cfg["static_folder"], cfg["frontend_built"])
    logger.info(
        "Storage: uploads=%s summaries=%s db=%s",
        UPLOAD_FOLDER,
        SUMMARY_FOLDER,
        _db_url,
    )
    logger.info(
        "Workers: job_workers=%d max_upload=%sMB llm_provider=%s",
        JOB_WORKERS,
        cfg["max_upload_mb"],
        LLM_PROVIDER,
    )
    if not cfg["assemblyai_configured"]:
        logger.warning("ASSEMBLYAI_API_KEY is not set — transcription will fail")
    if not cfg["llm_configured"]:
        logger.warning(
            "No LLM API key set (TOGETHER_API_KEY / HF_API_KEY) — summarization will fail"
        )
    if not cfg["frontend_built"]:
        logger.warning(
            "Frontend build missing — run: cd frontend && npm run build"
        )


@app.before_request
def _before_request():
    g._req_start = time.perf_counter()


@app.after_request
def _after_request(response):
    elapsed_ms = (time.perf_counter() - getattr(g, "_req_start", time.perf_counter())) * 1000
    path = request.path
    # Skip noisy job polling logs (every 2s from the browser).
    if request.method == "GET" and path.startswith("/api/jobs/"):
        return response
    logger.info(
        "%s %s -> %s (%.0fms)",
        request.method,
        path,
        response.status_code,
        elapsed_ms,
    )
    return response


# ========== Routes ==========
@app.route("/api/health", methods=["GET"])
def health():
    cfg = _config_status()
    ready = cfg["assemblyai_configured"] and cfg["llm_configured"] and cfg["frontend_built"]
    payload = {"status": "ok" if ready else "degraded", **cfg}
    return jsonify(payload), 200 if ready else 503


@app.route("/api/jobs/<job_id>", methods=["GET"])
def get_job(job_id):
    if not UUID_RE.match(job_id):
        return jsonify({"error": "Invalid job id"}), 400
    with db_session() as session:
        row = session.get(Job, job_id)
        if not row:
            return jsonify({"error": "Job not found"}), 404
        stage = None if row.status == "completed" else row.stage
        payload = {
            "job_id": row.id,
            "uid": row.id,
            "status": row.status,
            "stage": stage,
            "status_message": _status_message(row.status, stage),
            "error": row.error,
        }
        if row.status == "completed":
            payload["summary"] = row.summary or ""
            payload["quiz"] = row.quiz or ""
        response = jsonify(payload)
        response.headers["Cache-Control"] = "no-store"
        return response


@app.route("/process", methods=["POST"])
def process_file():
    """Start processing; returns job_id immediately. Poll GET /api/jobs/<id> for status."""
    youtube_url = (request.form.get("youtube_url") or "").strip()
    f: FileStorage | None = request.files.get("video_file")
    if f and (not f.filename):
        f = None

    has_file = f is not None
    has_url = bool(youtube_url)

    if has_file and has_url:
        return jsonify({"error": "Send either a video file or a YouTube URL, not both."}), 400
    if not has_file and not has_url:
        return jsonify({"error": "Provide a video file or a youtube_url."}), 400

    level = _normalize_level(request.form.get("level", "medium"))
    job_id = str(uuid.uuid4())

    if has_file:
        ext = os.path.splitext(f.filename)[1].lower()
        if ext not in ALLOWED_VIDEO_EXT:
            return jsonify(
                {"error": f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_VIDEO_EXT))}"}
            ), 400
        if request.content_length is not None and request.content_length > MAX_FILE_SIZE:
            return jsonify({"error": "File too large"}), 400

        safe_name = secure_filename(f.filename) or f"video{ext}"
        video_path = os.path.join(UPLOAD_FOLDER, f"{job_id}_{safe_name}")
        f.save(video_path)
        if os.path.getsize(video_path) > MAX_FILE_SIZE:
            _safe_unlink(video_path)
            return jsonify({"error": "File too large"}), 400

        _create_job(job_id, "upload", f.filename, level)
        _job_log(job_id, logging.INFO, "Queued (file upload)")
        if not _start_background_job(
            _run_file_job, app, job_id, video_path, level, f.filename
        ):
            _safe_unlink(video_path)
            _update_job(
                job_id,
                status="failed",
                error="All workers busy. Try again shortly.",
            )
            return jsonify(
                {"error": "Server is busy processing other videos. Try again in a minute."}
            ), 503
    else:
        if not _is_allowed_youtube_url(youtube_url):
            return jsonify({"error": "Invalid or unsupported YouTube URL."}), 400
        _create_job(job_id, "youtube", youtube_url, level)
        _job_log(job_id, logging.INFO, "Queued (YouTube)")
        if not _start_background_job(
            _run_youtube_job, app, job_id, youtube_url, level
        ):
            _update_job(
                job_id,
                status="failed",
                error="All workers busy. Try again shortly.",
            )
            return jsonify(
                {"error": "Server is busy processing other videos. Try again in a minute."}
            ), 503

    return jsonify(
        {
            "job_id": job_id,
            "uid": job_id,
            "status": "pending",
            "status_message": "Queued…",
        }
    )


@app.route("/download/<uid>/<filetype>")
def download(uid, filetype):
    if not UUID_RE.match(uid):
        return jsonify({"error": "Invalid download id"}), 400
    if filetype not in DOWNLOAD_TYPES:
        return jsonify({"error": "Invalid file type"}), 400
    path = os.path.join(SUMMARY_FOLDER, f"{uid}_{filetype}.txt")
    if not os.path.isfile(path):
        return jsonify({"error": "File not found"}), 404
    return send_file(
        path,
        as_attachment=True,
        download_name=f"{filetype}.txt",
        mimetype="text/plain",
    )


@app.route("/feedback", methods=["POST"])
def feedback():
    summary = request.form.get("summary")
    quiz = request.form.get("quiz")
    fb = request.form.get("feedback")
    if not summary or not quiz or fb is None:
        return jsonify({"error": "summary, quiz, and feedback are required"}), 400
    with db_session() as session:
        session.add(Feedback(summary=summary, quiz=quiz, feedback=fb))
    return jsonify({"message": "Thank you for your feedback!"})


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve(path):
    static_root = app.static_folder
    if path.startswith("api/"):
        return jsonify({"error": "Not found"}), 404
    if path != "" and static_root and os.path.exists(os.path.join(static_root, path)):
        return send_from_directory(static_root, path)
    if not static_root or not os.path.isfile(os.path.join(static_root, "index.html")):
        return (
            jsonify(
                {
                    "error": "Frontend build not found. "
                    "Run the React build into frontend/build or build/."
                }
            ),
            503,
        )
    return send_from_directory(static_root, "index.html")


_log_startup()

# ========== Run ==========
if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    # Reloader spawns a second process and kills background job threads mid-flight.
    use_reloader = os.getenv("FLASK_USE_RELOADER", "0") == "1"
    logger.info("Listening on http://0.0.0.0:%d (debug=%s)", port, debug)
    app.run(
        debug=debug,
        host="0.0.0.0",
        port=port,
        threaded=True,
        use_reloader=use_reloader,
    )
