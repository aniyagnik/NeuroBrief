"""Video processing pipeline: jobs, YouTube download, summary, and quiz."""

import logging
import os
import re
import threading
from urllib.parse import urlparse

import ffmpeg
import yt_dlp

from neurobrief.config import (
    JOB_WORKERS,
    LEVEL_ALIASES,
    STAGE_LABELS,
    SUMMARY_FOLDER,
    UPLOAD_FOLDER,
    logger,
)
from neurobrief.db import Job, Transcript, db_session
from neurobrief.llm import llm_chat
from neurobrief.transcription import transcribe_audio

_job_semaphore = threading.Semaphore(JOB_WORKERS)


def normalize_level(level):
    if not level:
        return "medium"
    return LEVEL_ALIASES.get(level.lower(), level.lower())


def is_allowed_youtube_url(url):
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


def status_message(status, stage):
    if status == "pending":
        return "Queued…"
    if status == "failed":
        return None
    if status == "completed":
        return "Done!"
    if stage and stage in STAGE_LABELS:
        return STAGE_LABELS[stage]
    return "Processing…"


def safe_unlink(path):
    try:
        if path and os.path.isfile(path):
            os.unlink(path)
    except OSError:
        pass


def safe_glob_unlink(prefix_without_ext):
    base = os.path.basename(prefix_without_ext)
    folder = os.path.dirname(prefix_without_ext) or UPLOAD_FOLDER
    try:
        for name in os.listdir(folder):
            if name.startswith(base + ".") or name == base:
                safe_unlink(os.path.join(folder, name))
    except OSError:
        pass


def update_job(job_id, **fields):
    with db_session() as session:
        row = session.get(Job, job_id)
        if row:
            for k, v in fields.items():
                setattr(row, k, v)


def job_log(job_id, level, msg, *args, exc_info=False):
    logging.getLogger("neurobrief.jobs").log(
        level, "[job:%s] " + msg, job_id[:8], *args, exc_info=exc_info
    )


def set_stage(job_id, stage):
    update_job(job_id, status="processing", stage=stage)
    label = STAGE_LABELS.get(stage, stage)
    job_log(job_id, logging.INFO, label)


def create_job(job_id, source_type, source_detail, level):
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


def summarize_text(text):
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
    return llm_chat(prompt.strip(), max_tokens=1024, temperature=0.7)


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

    raw = llm_chat(prompt.strip(), max_tokens=512, temperature=0.7, top_p=0.9)
    match = re.search(r"(Question Type:.*)$", raw, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    cleaned = raw.strip()
    if cleaned:
        logger.warning("Quiz format parse miss — returning raw LLM output (%d chars)", len(cleaned))
        return cleaned
    return "Quiz generation returned an empty response."


def finalize_success(job_id, video_label, transcript, summary, quiz):
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


def download_youtube_audio(url, job_id):
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
    raise RuntimeError("YouTube download finished but MP3 was not found (is ffmpeg installed?).")


def run_file_job(app, job_id, video_path, level, source_label):
    with app.app_context():
        audio_path = os.path.join(UPLOAD_FOLDER, f"{job_id}.mp3")
        try:
            set_stage(job_id, "extracting")
            (
                ffmpeg.input(video_path)
                .output(audio_path, format="mp3", acodec="libmp3lame", ac=1, ar="16k")
                .overwrite_output()
                .run(quiet=True)
            )
            set_stage(job_id, "transcribing")
            transcript = transcribe_audio(audio_path)
            set_stage(job_id, "summarizing")
            summary = summarize_text(transcript)
            quiz = generate_quiz(summary, level)
            finalize_success(job_id, source_label, transcript, summary, quiz)
            job_log(job_id, logging.INFO, "Completed successfully.")
        except Exception as e:
            job_log(job_id, logging.ERROR, "Failed: %s", e, exc_info=True)
            update_job(job_id, status="failed", error=str(e), stage=None)
        finally:
            safe_unlink(video_path)
            safe_unlink(audio_path)


def run_youtube_job(app, job_id, youtube_url, level):
    with app.app_context():
        out_base = os.path.join(UPLOAD_FOLDER, job_id)
        try:
            set_stage(job_id, "downloading")
            download_youtube_audio(youtube_url, job_id)
            set_stage(job_id, "transcribing")
            audio_path = os.path.join(UPLOAD_FOLDER, f"{job_id}.mp3")
            transcript = transcribe_audio(audio_path)
            set_stage(job_id, "summarizing")
            summary = summarize_text(transcript)
            quiz = generate_quiz(summary, level)
            finalize_success(job_id, youtube_url, transcript, summary, quiz)
            job_log(job_id, logging.INFO, "Completed successfully.")
        except Exception as e:
            job_log(job_id, logging.ERROR, "Failed: %s", e, exc_info=True)
            update_job(job_id, status="failed", error=str(e), stage=None)
        finally:
            safe_glob_unlink(out_base)


def start_background_job(target, *args):
    def wrapped():
        job_log(args[1], logging.INFO, "Worker acquired — starting pipeline")
        try:
            target(*args)
        finally:
            _job_semaphore.release()
            job_log(args[1], logging.DEBUG, "Worker released")

    if not _job_semaphore.acquire(blocking=False):
        return False
    threading.Thread(target=wrapped, daemon=True, name=f"job-{args[1][:8]}").start()
    return True
