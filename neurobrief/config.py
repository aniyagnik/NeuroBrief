"""App configuration, paths, and logging."""

import logging
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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
logging.getLogger("werkzeug").setLevel(logging.WARNING)

ALLOWED_VIDEO_EXT = {".mp4", ".webm", ".mkv", ".mov", ".avi", ".m4v"}
MAX_FILE_SIZE = int(os.getenv("MAX_UPLOAD_BYTES", str(100 * 1024 * 1024)))
JOB_WORKERS = max(1, int(os.getenv("JOB_WORKERS", "2")))
TOGETHER_API_URL = "https://api.together.ai/v1/chat/completions"
TOGETHER_MODEL = os.getenv("TOGETHER_MODEL", "mistralai/Mixtral-8x7B-Instruct-v0.1")
HF_API_URL = "https://router.huggingface.co/v1/chat/completions"
HF_MODEL = os.getenv("HF_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto").lower()
DOWNLOAD_TYPES = frozenset({"summary", "quiz", "transcript"})
LEVEL_ALIASES = {"high": "hard"}

STAGE_LABELS = {
    "extracting": "Extracting audio from video…",
    "downloading": "Downloading audio from YouTube…",
    "transcribing": "Transcribing audio (can take several minutes)…",
    "summarizing": "Generating summary and quiz…",
}


def abs_path(*parts):
    return os.path.join(BASE_DIR, *parts)


def resolve_static_folder():
    for candidate in (abs_path("frontend", "build"), abs_path("build")):
        if os.path.isdir(candidate):
            return candidate
    return abs_path("build")


UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", abs_path("uploads"))
SUMMARY_FOLDER = os.getenv("SUMMARY_FOLDER", abs_path("summaries"))
if not os.path.isabs(UPLOAD_FOLDER):
    UPLOAD_FOLDER = abs_path(UPLOAD_FOLDER)
if not os.path.isabs(SUMMARY_FOLDER):
    SUMMARY_FOLDER = abs_path(SUMMARY_FOLDER)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SUMMARY_FOLDER, exist_ok=True)

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{abs_path('feedback.db')}")
