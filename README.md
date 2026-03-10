# TTS Audio Generator

Reads text from a CSV file and generates corresponding `.wav` audio files using Google Vertex AI (Gemini TTS).

## Features

- Uses `gemini-2.5-flash-preview-tts` model
- Voice, input filename, and project settings are configurable via `.env`
- Automatically skips existing audio files
- Exponential backoff retry on API rate limits (429), up to 10 attempts
- Filenames are auto-generated from text content (alphanumeric only, spaces to underscores, lowercase)

## Setup

```bash
uv sync
```

## Authentication

This tool uses Vertex AI authentication. Complete Google Cloud auth first:

```bash
gcloud auth application-default login
```

## Configuration

Copy `.env.example` to `.env` and fill in your settings:

```bash
cp .env.example .env
```

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GOOGLE_CLOUD_PROJECT` | ✅ | — | Google Cloud project ID |
| `GOOGLE_CLOUD_LOCATION` | | `us-central1` | Vertex AI region |
| `TTS_VOICE_NAME` | | `Aoede` | Voice name (Aoede, Charon, Fenrir, Kore, Puck, Leda, etc.) |
| `TTS_INPUT_CSV` | | `list.csv` | Input CSV filename |

## Usage

1. Prepare a CSV file (default `list.csv`), one sentence per line:

```csv
Hello, how are you?
The weather is nice today.
```

2. Run:

```bash
uv run main.py
```

3. Audio files are saved to `output/`, e.g. `output/hello_how_are_you.wav`

## FAQ

**Q: `ModuleNotFoundError: No module named 'google'`?**
A: Make sure you've run `uv sync` and use `uv run` to execute the script.

**Q: Getting `RESOURCE_EXHAUSTED` errors?**
A: The script retries automatically. If it keeps failing, wait a bit or check your Vertex AI quota.
