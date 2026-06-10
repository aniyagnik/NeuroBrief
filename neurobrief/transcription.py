"""AssemblyAI transcription."""

import os
import time

import requests

from neurobrief.config import logger


def assemblyai_request_error(response, step):
    try:
        body = response.json()
        detail = body.get("error") or body.get("message") or body
    except Exception:
        detail = response.text or response.reason
    raise RuntimeError(f"AssemblyAI {step} failed ({response.status_code}): {detail}")


def transcribe_audio(audio_path):
    api_key = (os.getenv("ASSEMBLYAI_API_KEY") or "").strip().strip('"').strip("'")
    if not api_key:
        raise RuntimeError("ASSEMBLYAI_API_KEY is not set in the environment.")

    if not os.path.isfile(audio_path) or os.path.getsize(audio_path) == 0:
        raise RuntimeError("Audio file is missing or empty before transcription.")

    models_raw = os.getenv("ASSEMBLYAI_SPEECH_MODELS", "universal-3-pro,universal-2")
    speech_models = [m.strip() for m in models_raw.split(",") if m.strip()]

    base_url = "https://api.assemblyai.com"
    headers = {"authorization": api_key}

    with open(audio_path, "rb") as f:
        response = requests.post(base_url + "/v2/upload", headers=headers, data=f, timeout=300)
    if not response.ok:
        assemblyai_request_error(response, "upload")
    upload_payload = response.json()
    audio_url = upload_payload.get("upload_url")
    if not audio_url:
        raise RuntimeError(f"AssemblyAI upload failed: {upload_payload}")

    data = {"audio_url": audio_url, "speech_models": speech_models}
    response = requests.post(base_url + "/v2/transcript", json=data, headers=headers, timeout=60)
    if not response.ok:
        assemblyai_request_error(response, "transcript")
    transcript_id = response.json()["id"]
    polling_endpoint = base_url + "/v2/transcript/" + transcript_id
    logger.info("AssemblyAI transcript submitted id=%s", transcript_id)

    poll_count = 0
    while True:
        response = requests.get(polling_endpoint, headers=headers, timeout=60)
        if not response.ok:
            assemblyai_request_error(response, "transcript status")
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
