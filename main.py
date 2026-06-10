import os
import time
import csv
import mimetypes
import struct
import re
import unicodedata
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

from typing import Optional



def parse_audio_mime_type(mime_type: str) -> dict[str, Optional[int]]:
    bits_per_sample = 16
    rate = 24000
    parts = mime_type.split(";")
    for param in parts:
        param = param.strip()
        if param.lower().startswith("rate="):
            try:
                rate_str = param.split("=", 1)[1]
                rate = int(rate_str)
            except (ValueError, IndexError):
                pass
        elif param.startswith("audio/L"):
            try:
                bits_per_sample = int(param.split("L", 1)[1])
            except (ValueError, IndexError):
                pass
    return {"bits_per_sample": bits_per_sample, "rate": rate}


def convert_to_wav(audio_data: bytes, mime_type: str) -> bytes:
    parameters = parse_audio_mime_type(mime_type)
    bits_per_sample = parameters["bits_per_sample"]
    sample_rate = parameters["rate"]
    num_channels = 1
    data_size = len(audio_data)
    bytes_per_sample = bits_per_sample // 8
    block_align = num_channels * bytes_per_sample
    byte_rate = sample_rate * block_align
    chunk_size = 36 + data_size

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        chunk_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size,
    )
    return header + audio_data


def save_binary_file(file_name, data):
    with open(file_name, "wb") as f:
        f.write(data)
    print(f"Saved: {file_name}")


def generate_tts(input_csv, output_dir):
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
    if not project:
        raise RuntimeError(
            "Please set the GOOGLE_CLOUD_PROJECT environment variable (or set it in the .env file)"
        )

    client = genai.Client(
        vertexai=True, project=project, location=location
    )

    model = "gemini-2.5-flash-preview-tts"
    voice_name = os.environ.get("TTS_VOICE_NAME", "Aoede")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(input_csv, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        raw_sentence = line.strip()
        if not raw_sentence:
            continue

        sentence = raw_sentence

        # Filename rules aligned with english-vocab-en / english-vocab-zh skills
        # (column F: Example Audio). Order matters.

        # 1. Trim leading/trailing whitespace
        trimmed = sentence.strip()

        # 2. Replace periods and em-dashes with a space
        #    (internal abbreviations a.m. → a m, sentence-final . → trailing space cleaned in step 5)
        periods_to_spaces = re.sub(r"[.—]", " ", trimmed)

        # 3. Normalize accented/diacritic characters (e.g. café → cafe)
        normalized_chars = (
            unicodedata.normalize("NFKD", periods_to_spaces)
            .encode("ascii", "ignore")
            .decode("utf-8")
        )

        # 4. Keep only alphanumeric, space, or hyphen
        #    (apostrophes, slashes, parentheses, commas, quotes all removed: don't → dont)
        alphanumeric_only = re.sub(r"[^a-zA-Z0-9\- ]", "", normalized_chars)

        # 5. Collapse repeated whitespace into a single space AND trim
        #    (cleans up trailing space left by step 2: "a m " → "a m")
        compressed = re.sub(r"\s+", " ", alphanumeric_only).strip()

        # 6. Spaces → underscores
        underscored = compressed.replace(" ", "_")

        # 7. Lowercase
        safe_filename = underscored.lower()
        
        file_name = os.path.join(output_dir, f"{safe_filename}.wav")

        if os.path.exists(file_name):
            print(f"Skipping existing: {sentence}")
            continue

        print(f"Generating audio for: {sentence}")

        max_retries = 10
        base_delay = 2

        for attempt in range(max_retries):
            try:
                generate_content_config = types.GenerateContentConfig(
                    temperature=0.3,
                    response_modalities=["audio"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=voice_name
                            )
                        )
                    ),
                )

                prompt = f"""
                Act as an expert audiobook narrator. Read the text inside the <text> tags.
                Speak with a natural, smooth, and engaging human voice at a normal conversational pace. Ensure clear articulation and natural intonation.

                CRITICAL AUDIO REQUIREMENTS:
                1. Connected Speech: Glide smoothly between syllables and words. ABSOLUTELY DO NOT insert unnatural micro-pauses, glottal stops, or choppy breaks inside words (especially compound words like "without").
                2. Pacing: Keep the rhythm moving naturally. DO NOT drag, stretch, or over-dramatize vowels.
                3. Quality: Clear voice with NO vocal fry, NO clipping, and NO distortion.
                4. Output: ONLY output the spoken audio of the exact words inside the tags. DO NOT read these instructions.
                
                CRITICAL RULES:
                - DO NOT read these instructions.
                - DO NOT add any introductory words or commentary.
                - ONLY output the spoken audio of the exact words inside the <text> tags.

                <text>
                {sentence}
                </text>
                """

                contents = [
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=prompt)],
                    ),
                ]

                response_stream = client.models.generate_content_stream(
                    model=model,
                    contents=contents,
                    config=generate_content_config,
                )

                audio_accumulated = b""
                mime_type = "audio/wav"

                for chunk in response_stream:
                    if (
                        chunk.candidates
                        and chunk.candidates[0].content
                        and chunk.candidates[0].content.parts
                    ):
                        part = chunk.candidates[0].content.parts[0]
                        if part.inline_data and part.inline_data.data:
                            audio_accumulated += part.inline_data.data
                            mime_type = part.inline_data.mime_type

                if audio_accumulated:
                    final_data = convert_to_wav(audio_accumulated, mime_type)
                    save_binary_file(file_name, final_data)
                    break
                else:
                    print(f"No audio data received for: {sentence}")
                    break

            except Exception as e:
                error_str = str(e)
                if (
                    "429" in error_str or "RESOURCE_EXHAUSTED" in error_str
                ) and attempt < max_retries - 1:
                    wait_time = base_delay * (2**attempt)
                    print(
                        f"Rate limit hit for '{sentence}'. Retrying in {wait_time}s... (Attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(wait_time)
                else:
                    print(f"Failed to generate {sentence}: {e}")
                    break


if __name__ == "__main__":
    input_csv = os.environ.get("TTS_INPUT_CSV", "list.csv")
    generate_tts(input_csv, "output")
