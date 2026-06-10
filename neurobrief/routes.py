"""Flask application and HTTP routes."""

import logging
import os
import re
import time
import uuid

from flask import Flask, g, jsonify, request, send_file, send_from_directory
from flask_cors import CORS
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from neurobrief.config import (
    ALLOWED_VIDEO_EXT,
    DATABASE_URL,
    DOWNLOAD_TYPES,
    JOB_WORKERS,
    LLM_PROVIDER,
    MAX_FILE_SIZE,
    SUMMARY_FOLDER,
    UPLOAD_FOLDER,
    logger,
    resolve_static_folder,
)
from neurobrief.db import Feedback, Job, db_session
from neurobrief.llm import clean_env
from neurobrief.pipeline import (
    create_job,
    is_allowed_youtube_url,
    job_log,
    normalize_level,
    run_file_job,
    run_youtube_job,
    safe_unlink,
    start_background_job,
    status_message,
    update_job,
)

UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I
)

app = Flask(__name__, static_folder=resolve_static_folder())
CORS(app)


def config_status():
    together = bool(clean_env("TOGETHER_API_KEY"))
    hf = bool(clean_env("HF_API_KEY"))
    assembly = bool(clean_env("ASSEMBLYAI_API_KEY"))
    static_ok = bool(
        app.static_folder and os.path.isfile(os.path.join(app.static_folder, "index.html"))
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


def log_startup():
    cfg = config_status()
    logger.info("NeuroBrief backend starting")
    logger.info("Static UI: %s (built=%s)", cfg["static_folder"], cfg["frontend_built"])
    logger.info(
        "Storage: uploads=%s summaries=%s db=%s",
        UPLOAD_FOLDER,
        SUMMARY_FOLDER,
        DATABASE_URL,
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
        logger.warning("Frontend build missing — run: cd frontend && npm run build")


@app.before_request
def before_request():
    g._req_start = time.perf_counter()


@app.after_request
def after_request(response):
    elapsed_ms = (time.perf_counter() - getattr(g, "_req_start", time.perf_counter())) * 1000
    path = request.path
    if request.method == "GET" and path.startswith("/api/jobs/"):
        return response
    logger.info("%s %s -> %s (%.0fms)", request.method, path, response.status_code, elapsed_ms)
    return response


@app.route("/api/health", methods=["GET"])
def health():
    cfg = config_status()
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
            "status_message": status_message(row.status, stage),
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

    level = normalize_level(request.form.get("level", "medium"))
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
            safe_unlink(video_path)
            return jsonify({"error": "File too large"}), 400

        create_job(job_id, "upload", f.filename, level)
        job_log(job_id, logging.INFO, "Queued (file upload)")
        if not start_background_job(run_file_job, app, job_id, video_path, level, f.filename):
            safe_unlink(video_path)
            update_job(job_id, status="failed", error="All workers busy. Try again shortly.")
            return jsonify(
                {"error": "Server is busy processing other videos. Try again in a minute."}
            ), 503
    else:
        if not is_allowed_youtube_url(youtube_url):
            return jsonify({"error": "Invalid or unsupported YouTube URL."}), 400
        create_job(job_id, "youtube", youtube_url, level)
        job_log(job_id, logging.INFO, "Queued (YouTube)")
        if not start_background_job(run_youtube_job, app, job_id, youtube_url, level):
            update_job(job_id, status="failed", error="All workers busy. Try again shortly.")
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


log_startup()
