#!/usr/bin/env python3
"""Keep only YouTube-related cookies (smaller file for Render env vars / secrets)."""
import gzip
import base64
import sys
from pathlib import Path

KEEP = ("youtube.com", "youtu.be", "googlevideo.com", "google.com", "accounts.google.com")


def trim(text: str) -> str:
    out = ["# Netscape HTTP Cookie File"]
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            if s.startswith("# Netscape"):
                continue
            continue
        parts = line.split("\t")
        if len(parts) < 7:
            continue
        domain = parts[0].lstrip(".").lower()
        if any(k in domain for k in KEEP):
            out.append(line.rstrip())
    return "\n".join(out) + "\n"


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("youtube_cookies.txt")
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("youtube_cookies_trimmed.txt")
    raw = src.read_text(encoding="utf-8", errors="replace")
    trimmed = trim(raw)
    dst.write_text(trimmed, encoding="utf-8")
    print(f"Wrote {dst} ({len(trimmed)} bytes, was {len(raw)} bytes)")

    gz_b64 = base64.b64encode(gzip.compress(trimmed.encode("utf-8"))).decode("ascii")
    out_b64 = Path("youtube_cookies_gz_b64.txt")
    out_b64.write_text(gz_b64, encoding="utf-8")
    print(f"Wrote {out_b64} ({len(gz_b64)} chars) — paste as YTDLP_COOKIES_GZ_B64 on Render")


if __name__ == "__main__":
    main()
