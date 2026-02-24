# 🎬 Video Caption MCP Server

An MCP (Model Context Protocol) server that automatically transcribes and burns stylized captions into videos. Designed to work with [Poke](https://poke.com) by Interaction Co.

**Upload a video → AI transcribes it → Stylized captions are burned in → Download the result.**

## How It Works

```
You → Poke: "Caption this video: https://example.com/video.mp4"
     ↓
Poke calls your MCP server's `caption_video` tool
     ↓
1. Downloads the video
2. FFmpeg extracts audio (16kHz mono WAV)
3. Groq Whisper API transcribes with timestamps (FREE!)
4. Generates SRT subtitle file
5. FFmpeg burns styled captions into the video
     ↓
Returns: download link + full transcript
```

## Caption Styles

| Style | Look | Best For |
|-------|------|----------|
| `tiktok` (**default**) | Poppins bold white on dark box | TikTok, Reels, Shorts |
| `modern` | Poppins white with outline | General purpose |
| `classic` | Yellow text, bottom | Movies, TV style |
| `minimal` | Small Poppins, bottom-left | Clean, professional |
| `bold` | Impact font, heavy shadow | Maximum readability |

## Setup (15 minutes)

### 1. Get a Free Groq API Key

1. Go to [console.groq.com](https://console.groq.com)
2. Sign up (no credit card needed)
3. Go to API Keys → Create new key
4. Copy the key – you'll need it in step 3

> Groq's free tier includes Whisper transcription at no cost with generous rate limits.

### 2. Deploy to Render

#### Option A: One-Click Deploy
[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

1. Click the button above
2. Connect your GitHub account
3. It will create a new repo from this template and deploy it

#### Option B: Manual Deploy
1. Fork this repo to your GitHub
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your forked repo
4. Render auto-detects the Dockerfile
5. Click "Create Web Service"

### 3. Set Environment Variables

In your Render dashboard → Environment:

| Variable | Value | Required |
|----------|-------|----------|
| `GROQ_API_KEY` | Your Groq API key | ✅ Yes |
| `BASE_URL` | `https://your-app-name.onrender.com` | ✅ Yes |
| `WHISPER_MODEL` | `whisper-large-v3-turbo` | No (default) |
| `MAX_VIDEO_DURATION_SEC` | `600` | No (default: 10 min) |
| `PORT` | `8000` | No (default) |

> ⚠️ **Important**: Set `BASE_URL` to your actual Render URL so download links work!

### 4. Connect to Poke

1. Go to [poke.com/settings/connections/integrations/new](https://poke.com/settings/connections/integrations/new)
2. Enter a name: `Video Captioner`
3. Enter the MCP Server URL: `https://your-app-name.onrender.com/mcp`
4. Click Create Integration

### 5. Test It!

Message Poke:
> "Use the Video Captioner integration's caption_video tool to caption this video: https://example.com/my-video.mp4"

Or more naturally:
> "Can you add captions to this video? https://example.com/my-video.mp4 Use the bold style."

## MCP Tools

### `caption_video`
Transcribes and burns captions into a video.

**Parameters:**
- `video_url` (required): Direct URL to a video file
- `language` (optional): ISO-639-1 code, default `"en"`
- `style` (optional): `modern`, `classic`, `minimal`, or `bold`
- `font_size` (optional): 12-72, default `24`

### `list_caption_styles`
Returns all available caption style presets with descriptions.

## Local Development

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/video-caption-mcp.git
cd video-caption-mcp

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Make sure FFmpeg is installed
ffmpeg -version  # should work

# Set environment variables
export GROQ_API_KEY="your-key-here"
export BASE_URL="http://localhost:8000"

# Run the server
python src/server.py
```

Test with the MCP Inspector:
```bash
npx @modelcontextprotocol/inspector
# Connect to http://localhost:8000/mcp using "Streamable HTTP" transport
```

## Architecture

```
video-caption-mcp/
├── src/
│   └── server.py          # FastMCP server + file serving
├── Dockerfile             # Python 3.13 + FFmpeg
├── render.yaml            # Render deployment config
├── requirements.txt       # Python dependencies
└── README.md
```

The server exposes:
- `POST /mcp` – MCP protocol endpoint (for Poke)
- `GET /files/{job_id}/{filename}` – Serves captioned video downloads
- `GET /health` – Health check

## Limitations

- **Video size**: Groq free tier accepts audio up to 25 MB (roughly 10-15 min of video audio)
- **Duration**: Default max 10 minutes (configurable via `MAX_VIDEO_DURATION_SEC`)
- **Render free tier**: May spin down after inactivity; first request after sleep takes ~30s
- **File cleanup**: Output files are auto-deleted after 1 hour
- **Direct URLs only**: The video URL must be a direct download link (not YouTube, etc.)

## Tips

- For YouTube/social media videos, use a service to get a direct download link first
- The `modern` style works best for vertical/short-form video
- Use `language` parameter for non-English videos for better accuracy
- Groq's Whisper is extremely fast – transcription usually takes just seconds

## License

MIT
