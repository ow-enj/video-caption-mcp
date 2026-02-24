"""
Video Caption MCP Server
========================
An MCP server that accepts a video URL, transcribes the audio using
Groq's Whisper API (free tier), and burns stylized captions into the
video using FFmpeg. Designed for use with Poke (poke.com).

Deploy to Render, then add the MCP URL to Poke at:
  poke.com/settings/connections/integrations/new
"""

import os
import uuid
import json
import time
import asyncio
import tempfile
import subprocess
import logging
from pathlib import Path
from contextlib import asynccontextmanager

import httpx
from fastmcp import FastMCP
from pydantic import BaseModel, Field, ConfigDict

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "whisper-large-v3-turbo")
BASE_URL = os.environ.get("BASE_URL", "")
MAX_VIDEO_DURATION_SEC = int(os.environ.get("MAX_VIDEO_DURATION_SEC", "600"))
CLEANUP_AFTER_SEC = int(os.environ.get("CLEANUP_AFTER_SEC", "3600"))

OUTPUT_DIR = Path(tempfile.gettempdir()) / "caption_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("video-caption-mcp")

# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

mcp = FastMCP("video_caption_mcp")

# ---------------------------------------------------------------------------
# Pydantic Input Models
# ---------------------------------------------------------------------------

class CaptionVideoInput(BaseModel):
    """Input for the caption_video tool."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    video_url: str = Field(
        ...,
        description=(
            "Direct URL to a video file (mp4, mov, webm, mkv, etc). "
            "Must be a publicly accessible direct download link."
        ),
        min_length=10,
        max_length=2048,
    )
    language: str = Field(
        default="en",
        description="ISO-639-1 language code of the spoken audio (e.g. 'en', 'es', 'fr'). Default: 'en'.",
        min_length=2,
        max_length=5,
    )
    style: str = Field(
        default="tiktok",
        description=(
            "Caption style preset. Options: "
            "'tiktok' (Poppins font, white bold text on dark rounded box), "
            "'modern' (Poppins white with dark outline), "
            "'classic' (yellow text at bottom), "
            "'minimal' (small Poppins text, bottom-left), "
            "'bold' (large Impact font with heavy shadow). "
            "Default: 'tiktok'."
        ),
    )
    font_size: int = Field(
        default=24,
        description="Base font size for captions. Default: 24.",
        ge=12,
        le=72,
    )


class TranscriptionResult(BaseModel):
    """Internal model for a transcription segment."""
    start: float
    end: float
    text: str


# ---------------------------------------------------------------------------
# Caption Style Presets
# ---------------------------------------------------------------------------

CAPTION_STYLES = {
    "tiktok": (
        "FontName=Poppins,FontSize={size},PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&H00000000,BackColour=&HC0000000,"
        "Bold=1,BorderStyle=3,Outline=4,Shadow=0,MarginV=60,Alignment=2"
    ),
    "modern": (
        "FontName=Poppins,FontSize={size},PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&H00000000,BackColour=&H80000000,"
        "Bold=1,Outline=2,Shadow=1,MarginV=50,Alignment=2"
    ),
    "classic": (
        "FontName=Arial,FontSize={size},PrimaryColour=&H0000FFFF,"
        "OutlineColour=&H00000000,BackColour=&H80000000,"
        "Bold=1,Outline=1,Shadow=0,MarginV=30,Alignment=2"
    ),
    "minimal": (
        "FontName=Poppins,FontSize={size},PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&H00000000,BackColour=&H00000000,"
        "Bold=0,Outline=1,Shadow=0,MarginV=20,MarginL=20,Alignment=1"
    ),
    "bold": (
        "FontName=Impact,FontSize={size},PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&H00000000,BackColour=&H40000000,"
        "Bold=1,Outline=3,Shadow=2,MarginV=40,Alignment=2"
    ),
}


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def _cleanup_old_files():
    now = time.time()
    for f in OUTPUT_DIR.iterdir():
        if f.is_file() and (now - f.stat().st_mtime) > CLEANUP_AFTER_SEC:
            f.unlink(missing_ok=True)


def _srt_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _segments_to_srt(segments: list[TranscriptionResult]) -> str:
    lines = []
    for i, seg in enumerate(segments, 1):
        lines.append(str(i))
        lines.append(f"{_srt_timestamp(seg.start)} --> {_srt_timestamp(seg.end)}")
        lines.append(seg.text.strip())
        lines.append("")
    return "\n".join(lines)


def _run_ffmpeg(args: list[str], description: str = "FFmpeg") -> subprocess.CompletedProcess:
    cmd = ["ffmpeg", "-y"] + args
    logger.info(f"{description}: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        logger.error(f"{description} stderr: {result.stderr}")
        raise RuntimeError(f"{description} failed: {result.stderr[:500]}")
    return result


async def _download_video(url: str, dest: Path) -> None:
    async with httpx.AsyncClient(follow_redirects=True, timeout=120) as client:
        async with client.stream("GET", url) as resp:
            resp.raise_for_status()
            with open(dest, "wb") as f:
                async for chunk in resp.aiter_bytes(chunk_size=65536):
                    f.write(chunk)
    logger.info(f"Downloaded video: {dest.stat().st_size / 1024 / 1024:.1f} MB")


def _get_duration(video_path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(video_path)],
        capture_output=True, text=True, timeout=30,
    )
    info = json.loads(result.stdout)
    return float(info["format"]["duration"])


def _extract_audio(video_path: Path, audio_path: Path) -> None:
    _run_ffmpeg(
        ["-i", str(video_path), "-vn", "-ar", "16000", "-ac", "1", "-f", "wav", str(audio_path)],
        description="Extract audio",
    )


async def _transcribe_audio(audio_path: Path, language: str) -> list[TranscriptionResult]:
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY environment variable is not set. Get a free key at https://console.groq.com")

    file_size = audio_path.stat().st_size
    logger.info(f"Transcribing audio: {file_size / 1024 / 1024:.1f} MB")

    if file_size > 25 * 1024 * 1024:
        raise ValueError(f"Audio file is {file_size / 1024 / 1024:.1f} MB, exceeds Groq free tier limit of 25 MB.")

    async with httpx.AsyncClient(timeout=120) as client:
        with open(audio_path, "rb") as f:
            resp = await client.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                files={"file": ("audio.wav", f, "audio/wav")},
                data={
                    "model": WHISPER_MODEL,
                    "response_format": "verbose_json",
                    "timestamp_granularities[]": "segment",
                    "language": language,
                    "temperature": "0.0",
                },
            )
            resp.raise_for_status()

    data = resp.json()
    segments = data.get("segments", [])

    if not segments:
        raise ValueError("Whisper returned no segments. The audio might be silent or too noisy.")

    results = []
    for seg in segments:
        results.append(TranscriptionResult(start=seg["start"], end=seg["end"], text=seg["text"]))

    logger.info(f"Transcribed {len(results)} segments")
    return results


def _burn_captions(video_path: Path, srt_path: Path, output_path: Path, style: str, font_size: int) -> None:
    style_template = CAPTION_STYLES.get(style, CAPTION_STYLES["tiktok"])
    style_str = style_template.format(size=font_size)
    srt_escaped = str(srt_path).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")

    _run_ffmpeg(
        [
            "-i", str(video_path),
            "-vf", f"subtitles='{srt_escaped}':force_style='{style_str}'",
            "-c:a", "copy", "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            str(output_path),
        ],
        description="Burn captions",
    )


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------

@mcp.tool(name="caption_video")
async def caption_video(params: CaptionVideoInput) -> str:
    """Transcribe and burn stylized captions into a video.

    Downloads a video from a URL, extracts the audio, transcribes it using
    Groq Whisper AI, and burns captions directly into the video.

    Returns a download URL for the captioned video and the full transcript.
    """
    _cleanup_old_files()

    job_id = uuid.uuid4().hex[:12]
    work_dir = OUTPUT_DIR / job_id
    work_dir.mkdir(parents=True, exist_ok=True)

    video_path = work_dir / "input_video.mp4"
    audio_path = work_dir / "audio.wav"
    srt_path = work_dir / "captions.srt"
    output_path = work_dir / f"captioned_{job_id}.mp4"

    try:
        logger.info(f"[{job_id}] Downloading video from {params.video_url}")
        await _download_video(params.video_url, video_path)

        duration = _get_duration(video_path)
        if duration > MAX_VIDEO_DURATION_SEC:
            return json.dumps({"error": f"Video is {duration:.0f}s, max is {MAX_VIDEO_DURATION_SEC}s."})
        logger.info(f"[{job_id}] Video duration: {duration:.1f}s")

        logger.info(f"[{job_id}] Extracting audio...")
        await asyncio.to_thread(_extract_audio, video_path, audio_path)

        logger.info(f"[{job_id}] Transcribing with Whisper ({WHISPER_MODEL})...")
        segments = await _transcribe_audio(audio_path, params.language)

        srt_content = _segments_to_srt(segments)
        srt_path.write_text(srt_content, encoding="utf-8")
        logger.info(f"[{job_id}] Generated SRT with {len(segments)} segments")

        logger.info(f"[{job_id}] Burning captions (style={params.style})...")
        await asyncio.to_thread(_burn_captions, video_path, srt_path, output_path, params.style, params.font_size)

        if BASE_URL:
            download_url = f"{BASE_URL.rstrip('/')}/files/{job_id}/{output_path.name}"
        else:
            download_url = f"(BASE_URL not configured – file at {output_path})"

        full_text = " ".join(seg.text.strip() for seg in segments)

        video_path.unlink(missing_ok=True)
        audio_path.unlink(missing_ok=True)

        result = {
            "status": "success",
            "download_url": download_url,
            "transcript": full_text,
            "segment_count": len(segments),
            "duration_seconds": round(duration, 1),
            "style_used": params.style,
            "srt_preview": srt_content[:1000],
        }

        logger.info(f"[{job_id}] Done! Output: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
        return json.dumps(result, indent=2)

    except httpx.HTTPStatusError as e:
        return json.dumps({"error": f"HTTP error: {e.response.status_code} – {e.response.text[:300]}"})
    except subprocess.TimeoutExpired:
        return json.dumps({"error": "FFmpeg timed out. The video might be too large."})
    except Exception as e:
        logger.exception(f"[{job_id}] Error")
        return json.dumps({"error": f"{type(e).__name__}: {str(e)}"})


@mcp.tool(name="list_caption_styles")
async def list_caption_styles() -> str:
    """List all available caption style presets."""
    styles = [
        {"name": "tiktok", "description": "Poppins bold white on dark box – TikTok/Reels style", "default": True},
        {"name": "modern", "description": "Poppins white with outline and shadow"},
        {"name": "classic", "description": "Yellow text at bottom – traditional subtitles"},
        {"name": "minimal", "description": "Small Poppins text, bottom-left – clean and minimal"},
        {"name": "bold", "description": "Impact font, heavy shadow – maximum readability"},
    ]
    return json.dumps(styles, indent=2)


# ---------------------------------------------------------------------------
# File Serving + Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from starlette.applications import Starlette
    from starlette.responses import FileResponse, JSONResponse
    from starlette.routing import Route, Mount
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))

    async def serve_file(request):
        job_id = request.path_params["job_id"]
        filename = request.path_params["filename"]
        if ".." in job_id or ".." in filename or "/" in job_id:
            return JSONResponse({"error": "Invalid path"}, status_code=400)
        file_path = OUTPUT_DIR / job_id / filename
        if not file_path.exists():
            return JSONResponse({"error": "File not found or expired."}, status_code=404)
        return FileResponse(file_path, media_type="video/mp4", filename=filename)

    async def health_check(request):
        return JSONResponse({"status": "ok", "server": "video_caption_mcp"})

    # Get the MCP ASGI app from FastMCP
    mcp_app = mcp.http_app()

    # Compose: file routes + MCP app
    app = Starlette(
        routes=[
            Route("/health", health_check),
            Route("/files/{job_id}/{filename}", serve_file),
            Mount("/", app=mcp_app),
        ],
    )

    logger.info(f"Starting on port {port}")
    logger.info(f"MCP endpoint: http://0.0.0.0:{port}/mcp")
    logger.info(f"Health: http://0.0.0.0:{port}/health")

    uvicorn.run(app, host="0.0.0.0", port=port)
