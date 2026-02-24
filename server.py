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
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field, ConfigDict

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "whisper-large-v3-turbo")
BASE_URL = os.environ.get("BASE_URL", "")  # e.g. https://your-app.onrender.com
MAX_VIDEO_DURATION_SEC = int(os.environ.get("MAX_VIDEO_DURATION_SEC", "600"))  # 10 min default
CLEANUP_AFTER_SEC = int(os.environ.get("CLEANUP_AFTER_SEC", "3600"))  # 1 hour

OUTPUT_DIR = Path(tempfile.gettempdir()) / "caption_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("video-caption-mcp")

# ---------------------------------------------------------------------------
# Lifespan – cleanup old files periodically
# ---------------------------------------------------------------------------

@asynccontextmanager
async def app_lifespan():
    """Clean up old output files on startup."""
    _cleanup_old_files()
    yield {}

def _cleanup_old_files():
    """Remove output files older than CLEANUP_AFTER_SEC."""
    now = time.time()
    for f in OUTPUT_DIR.iterdir():
        if f.is_file() and (now - f.stat().st_mtime) > CLEANUP_AFTER_SEC:
            f.unlink(missing_ok=True)
            logger.info(f"Cleaned up old file: {f.name}")

# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "video_caption_mcp",
    lifespan=app_lifespan,
)

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
            "'tiktok' (Poppins font, white bold text on dark rounded box – like the screenshot style), "
            "'modern' (Poppins white with dark outline), "
            "'classic' (yellow text at bottom), "
            "'minimal' (small Poppins text, bottom-left), "
            "'bold' (large Impact font with heavy shadow). "
            "Default: 'tiktok'."
        ),
    )
    font_size: int = Field(
        default=24,
        description="Base font size for captions (will be scaled to video resolution). Default: 24.",
        ge=12,
        le=72,
    )


class TranscriptionResult(BaseModel):
    """Internal model for a transcription segment."""
    start: float
    end: float
    text: str


# ---------------------------------------------------------------------------
# Caption Style Presets (FFmpeg subtitle filter strings)
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

def _srt_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _segments_to_srt(segments: list[TranscriptionResult]) -> str:
    """Convert transcription segments to SRT subtitle format."""
    lines = []
    for i, seg in enumerate(segments, 1):
        lines.append(str(i))
        lines.append(f"{_srt_timestamp(seg.start)} --> {_srt_timestamp(seg.end)}")
        lines.append(seg.text.strip())
        lines.append("")
    return "\n".join(lines)


def _run_ffmpeg(args: list[str], description: str = "FFmpeg") -> subprocess.CompletedProcess:
    """Run an FFmpeg command and raise on failure."""
    cmd = ["ffmpeg", "-y"] + args
    logger.info(f"{description}: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        logger.error(f"{description} stderr: {result.stderr}")
        raise RuntimeError(f"{description} failed: {result.stderr[:500]}")
    return result


async def _download_video(url: str, dest: Path) -> None:
    """Download a video from a URL to a local path."""
    async with httpx.AsyncClient(follow_redirects=True, timeout=120) as client:
        async with client.stream("GET", url) as resp:
            resp.raise_for_status()
            with open(dest, "wb") as f:
                async for chunk in resp.aiter_bytes(chunk_size=65536):
                    f.write(chunk)
    logger.info(f"Downloaded video: {dest.stat().st_size / 1024 / 1024:.1f} MB")


def _get_duration(video_path: Path) -> float:
    """Get video duration in seconds using ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            str(video_path),
        ],
        capture_output=True, text=True, timeout=30,
    )
    info = json.loads(result.stdout)
    return float(info["format"]["duration"])


def _extract_audio(video_path: Path, audio_path: Path) -> None:
    """Extract audio from video as 16kHz mono WAV (optimal for Whisper)."""
    _run_ffmpeg(
        [
            "-i", str(video_path),
            "-vn",               # no video
            "-ar", "16000",      # 16kHz sample rate
            "-ac", "1",          # mono
            "-f", "wav",
            str(audio_path),
        ],
        description="Extract audio",
    )


async def _transcribe_audio(audio_path: Path, language: str) -> list[TranscriptionResult]:
    """Send audio to Groq Whisper API and return timestamped segments."""
    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY environment variable is not set. "
            "Get a free API key at https://console.groq.com"
        )

    file_size = audio_path.stat().st_size
    logger.info(f"Transcribing audio: {file_size / 1024 / 1024:.1f} MB")

    # Groq free tier: max 25 MB per file
    if file_size > 25 * 1024 * 1024:
        raise ValueError(
            f"Audio file is {file_size / 1024 / 1024:.1f} MB, "
            "which exceeds Groq free tier limit of 25 MB. "
            "Try a shorter video."
        )

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
        results.append(TranscriptionResult(
            start=seg["start"],
            end=seg["end"],
            text=seg["text"],
        ))

    logger.info(f"Transcribed {len(results)} segments")
    return results


def _burn_captions(
    video_path: Path,
    srt_path: Path,
    output_path: Path,
    style: str,
    font_size: int,
) -> None:
    """Burn SRT captions into video using FFmpeg subtitles filter with styling."""
    style_template = CAPTION_STYLES.get(style, CAPTION_STYLES["modern"])
    style_str = style_template.format(size=font_size)

    # Escape special characters in the SRT path for FFmpeg filter
    srt_escaped = str(srt_path).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")

    _run_ffmpeg(
        [
            "-i", str(video_path),
            "-vf", f"subtitles='{srt_escaped}':force_style='{style_str}'",
            "-c:a", "copy",          # keep original audio
            "-c:v", "libx264",       # re-encode video with captions
            "-preset", "fast",
            "-crf", "23",
            str(output_path),
        ],
        description="Burn captions",
    )


# ---------------------------------------------------------------------------
# MCP Tool
# ---------------------------------------------------------------------------

@mcp.tool(
    name="caption_video",
    annotations={
        "title": "Caption Video with AI Transcription",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    },
)
async def caption_video(params: CaptionVideoInput) -> str:
    """Transcribe and burn stylized captions into a video.

    Downloads a video from a URL, extracts the audio, transcribes it using
    Groq's Whisper AI (with timestamps), generates an SRT subtitle file,
    and burns the captions directly into the video with your chosen style.

    Returns a download URL for the captioned video and the full transcript.

    Args:
        params (CaptionVideoInput): Input containing:
            - video_url (str): Direct URL to a video file
            - language (str): ISO-639-1 language code (default: 'en')
            - style (str): Caption style preset (modern/classic/minimal/bold)
            - font_size (int): Base font size (default: 24)

    Returns:
        str: JSON with download_url, transcript text, segment count, and SRT content
    """
    job_id = uuid.uuid4().hex[:12]
    work_dir = OUTPUT_DIR / job_id
    work_dir.mkdir(parents=True, exist_ok=True)

    video_path = work_dir / "input_video.mp4"
    audio_path = work_dir / "audio.wav"
    srt_path = work_dir / "captions.srt"
    output_path = work_dir / f"captioned_{job_id}.mp4"

    try:
        # Step 1: Download
        logger.info(f"[{job_id}] Downloading video from {params.video_url}")
        await _download_video(params.video_url, video_path)

        # Step 2: Check duration
        duration = _get_duration(video_path)
        if duration > MAX_VIDEO_DURATION_SEC:
            return json.dumps({
                "error": (
                    f"Video is {duration:.0f}s ({duration/60:.1f} min), "
                    f"which exceeds the max of {MAX_VIDEO_DURATION_SEC}s. "
                    "Try a shorter video."
                )
            })
        logger.info(f"[{job_id}] Video duration: {duration:.1f}s")

        # Step 3: Extract audio
        logger.info(f"[{job_id}] Extracting audio...")
        await asyncio.to_thread(_extract_audio, video_path, audio_path)

        # Step 4: Transcribe with Groq Whisper
        logger.info(f"[{job_id}] Transcribing with Whisper ({WHISPER_MODEL})...")
        segments = await _transcribe_audio(audio_path, params.language)

        # Step 5: Generate SRT
        srt_content = _segments_to_srt(segments)
        srt_path.write_text(srt_content, encoding="utf-8")
        logger.info(f"[{job_id}] Generated SRT with {len(segments)} segments")

        # Step 6: Burn captions into video
        logger.info(f"[{job_id}] Burning captions (style={params.style})...")
        await asyncio.to_thread(
            _burn_captions, video_path, srt_path, output_path,
            params.style, params.font_size,
        )

        # Build download URL
        if BASE_URL:
            download_url = f"{BASE_URL.rstrip('/')}/files/{job_id}/{output_path.name}"
        else:
            download_url = f"(server BASE_URL not configured – file at {output_path})"

        # Full transcript
        full_text = " ".join(seg.text.strip() for seg in segments)

        # Clean up intermediate files (keep output + srt)
        video_path.unlink(missing_ok=True)
        audio_path.unlink(missing_ok=True)

        result = {
            "status": "success",
            "download_url": download_url,
            "transcript": full_text,
            "segment_count": len(segments),
            "duration_seconds": round(duration, 1),
            "style_used": params.style,
            "srt_preview": srt_content[:1000] + ("..." if len(srt_content) > 1000 else ""),
        }

        logger.info(f"[{job_id}] Done! Output: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
        return json.dumps(result, indent=2)

    except httpx.HTTPStatusError as e:
        return json.dumps({
            "error": f"HTTP error: {e.response.status_code} – {e.response.text[:300]}"
        })
    except subprocess.TimeoutExpired:
        return json.dumps({
            "error": "FFmpeg timed out. The video might be too large or complex."
        })
    except Exception as e:
        logger.exception(f"[{job_id}] Error")
        return json.dumps({"error": f"{type(e).__name__}: {str(e)}"})


@mcp.tool(
    name="list_caption_styles",
    annotations={
        "title": "List Available Caption Styles",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def list_caption_styles() -> str:
    """List all available caption style presets with descriptions.

    Returns details about each built-in caption style that can be used
    with the caption_video tool.

    Returns:
        str: JSON list of style presets with names and descriptions
    """
    styles = [
        {
            "name": "tiktok",
            "description": "Poppins bold white text on a dark semi-transparent box – the TikTok/Reels/Shorts look from the reference screenshot.",
            "preview": "Poppins, bold, white on dark box – social media standard",
            "default": True,
        },
        {
            "name": "modern",
            "description": "Poppins bold white text with dark outline and subtle shadow – clean and modern.",
            "preview": "Poppins, white, outlined, shadow – versatile",
        },
        {
            "name": "classic",
            "description": "Yellow text at the bottom of the screen – traditional subtitle look.",
            "preview": "Yellow, bold, outlined – classic TV/movie subtitles",
        },
        {
            "name": "minimal",
            "description": "Small Poppins text at bottom-left – unobtrusive and clean.",
            "preview": "Poppins, thin, bottom-left aligned – minimal and clean",
        },
        {
            "name": "bold",
            "description": "Large centered Impact font with heavy shadow – maximum readability.",
            "preview": "Impact font, large, heavy shadow – impossible to miss",
        },
    ]
    return json.dumps(styles, indent=2)


# ---------------------------------------------------------------------------
# File Serving (custom ASGI middleware for serving output files)
# ---------------------------------------------------------------------------

from starlette.applications import Starlette
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Route, Mount

async def serve_file(request):
    """Serve a captioned video file by job_id and filename."""
    job_id = request.path_params["job_id"]
    filename = request.path_params["filename"]

    # Sanitize
    if ".." in job_id or ".." in filename or "/" in job_id:
        return JSONResponse({"error": "Invalid path"}, status_code=400)

    file_path = OUTPUT_DIR / job_id / filename
    if not file_path.exists():
        return JSONResponse(
            {"error": "File not found. It may have been cleaned up."},
            status_code=404,
        )

    return FileResponse(
        file_path,
        media_type="video/mp4",
        filename=filename,
    )

async def health_check(request):
    """Health check endpoint."""
    return JSONResponse({"status": "ok", "server": "video_caption_mcp"})


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))

    # We need to serve both the MCP endpoint AND file downloads.
    # FastMCP's streamable HTTP mounts at /mcp. We add file routes alongside it.
    #
    # Strategy: Use Starlette to compose both the MCP app and file serving.

    from starlette.routing import Route, Mount
    from starlette.applications import Starlette
    import uvicorn

    # Get the MCP ASGI app
    mcp_app = mcp.streamable_http_app()

    # Compose into a parent Starlette app
    app = Starlette(
        routes=[
            Route("/health", health_check),
            Route("/files/{job_id}/{filename}", serve_file),
            Mount("/mcp", app=mcp_app),
        ],
    )

    logger.info(f"Starting server on port {port}")
    logger.info(f"MCP endpoint: http://0.0.0.0:{port}/mcp")
    logger.info(f"File serving:  http://0.0.0.0:{port}/files/{{job_id}}/{{filename}}")

    uvicorn.run(app, host="0.0.0.0", port=port)
