import whisper
import faiss
import numpy as np
from scenedetect import detect, ContentDetector
from sentence_transformers import SentenceTransformer
import os
import logging
import subprocess
import requests
import base64
import cv2
import re

from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =========================
# CONFIG
# =========================

VISION_MODEL = "moondream:latest"

print("Loading AI models...")

embed_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# BETTER AUDIO TRANSCRIPTION
whisper_model = whisper.load_model(
    "small"
)

print("Models ready.")

# =========================
# AUDIO CHECK
# =========================

def has_audio(video_path: str) -> bool:

    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-af", "volumedetect",
        "-f", "null",
        "-"
    ]

    result = subprocess.run(
        cmd,
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True
    )

    output = result.stderr

    mean_volume = None

    for line in output.splitlines():

        if "mean_volume:" in line:

            try:

                mean_volume = float(
                    line.split("mean_volume:")[1]
                    .split(" dB")[0]
                    .strip()
                )

            except:
                pass

    if mean_volume is None:
        return False

    return mean_volume > -50

# =========================
# VIDEO DURATION
# =========================

def get_video_duration(video_path: str) -> float:

    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        video_path
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )

    try:

        return float(
            result.stdout.strip()
        )

    except:

        return 0.0

# =========================
# AUDIO EXTRACTION
# =========================

def extract_audio(video_path: str):

    os.makedirs(
        "data",
        exist_ok=True
    )

    audio_path = (
        "data/temp_audio.wav"
    )

    cmd = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        audio_path
    ]

    subprocess.run(
        cmd,
        capture_output=True
    )

    return audio_path

# =========================
# FRAME EXTRACTION
# =========================

def extract_frames(
    video_path: str,
    interval_seconds: int = 3
) -> List[Dict]:

    frames_dir = "data/frames"

    os.makedirs(
        frames_dir,
        exist_ok=True
    )

    duration = get_video_duration(
        video_path
    )

    frames = []

    timestamp = 0

    while timestamp < duration:

        frame_path = (
            f"{frames_dir}/"
            f"frame_{int(timestamp)}.jpg"
        )

        cmd = [
            "ffmpeg",
            "-y",
            "-ss", str(timestamp),
            "-i", video_path,
            "-vframes", "1",
            "-q:v", "8",
            frame_path
        ]

        subprocess.run(
            cmd,
            capture_output=True
        )

        if os.path.exists(frame_path):

            frame = cv2.imread(
                frame_path
            )

            if frame is not None:

                frame = cv2.resize(
                    frame,
                    (320, 320)
                )

                cv2.imwrite(
                    frame_path,
                    frame
                )

            frames.append({

                "path": frame_path,

                "timestamp": float(
                    timestamp
                )
            })

        timestamp += interval_seconds

    return frames

# =========================
# VISION MODEL
# =========================

def describe_frame_with_vision_model(
    frame_path: str
) -> str:

    with open(frame_path, "rb") as f:

        image_data = base64.b64encode(
            f.read()
        ).decode("utf-8")

    try:

        response = requests.post(

            "http://localhost:11434/api/generate",

            json={

                "model": VISION_MODEL,

                "prompt":
"""
You are analyzing a frame from a video.

Describe ONLY the MAIN visible action.

Rules:
- Focus on movement or activity.
- Mention the main subject.
- Ignore unnecessary background details.
- Do NOT hallucinate.
- Do NOT invent people or objects.
- Keep answer short and factual.

Good examples:
- "A chicken is crossing a road."
- "A dog runs through grass."
- "A man walks down a hallway."

Return ONLY one sentence.
""",

                "images": [image_data],

                "stream": False
            },

            timeout=180
        )

        response.raise_for_status()

        output = response.json().get(
            "response",
            ""
        ).strip()

        if (
            not output
            or "cannot" in output.lower()
            or "unable" in output.lower()
        ):

            return (
                "No meaningful action detected."
            )

        return output

    except Exception as e:

        logger.error(
            f"Vision model error: {e}"
        )

        return (
            "Frame could not be described."
        )

# =========================
# SILENT VIDEO PROCESSING
# =========================

def transcribe_silent_video(
    video_path: str
) -> List[Dict]:

    logger.info(
        "No audio detected. "
        "Using vision analysis..."
    )

    print("Step 1: Getting video duration")

    duration = get_video_duration(
        video_path
    )

    print(f"Step 2: Duration = {duration}")

    interval = (
        3 if duration < 60 else 5
    )

    print("Step 3: Extracting frames")

    frames = extract_frames(
        video_path,
        interval_seconds=interval
    )

    print(
        f"Step 4: Frame extraction complete. "
        f"Frames found: {len(frames)}"
    )

    segments = []

    print("Step 5: Starting vision model")

    with ThreadPoolExecutor(
        max_workers=1
    ) as executor:

        descriptions = list(

            executor.map(

                lambda frame:
                describe_frame_with_vision_model(
                    frame["path"]
                ),

                frames
            )
        )

    print("Step 6: Vision model completed")

    for frame, description in zip(
        frames,
        descriptions
    ):

        description = description.strip()

        if (
            "could not" in description.lower()
            or "no visible" in description.lower()
            or "no meaningful" in description.lower()
        ):
            continue

        segments.append({

            "text": description,

            "start": float(
                frame["timestamp"]
            ),

            "end": float(
                frame["timestamp"] +
                interval
            )
        })

    print("Step 7: Silent video processing complete")

    return segments
# =========================
# MAIN VIDEO PROCESSING
# =========================

def process_video(
    video_path: str
) -> Dict[str, Any]:

    if not os.path.exists(video_path):

        raise FileNotFoundError(
            f"File not found: {video_path}"
        )

    print("Step A: Video exists")

    logger.info(
        "Processing video..."
    )

    audio_present = has_audio(
        video_path
    )

    print(
        f"Step B: Audio present = {audio_present}"
    )

    if audio_present:

        print("Step C: Starting Whisper")

        audio_path = extract_audio(
            video_path
        )

        with ThreadPoolExecutor(
            max_workers=1
        ) as executor:

            transcribe_future = executor.submit(
                whisper_model.transcribe,
                audio_path,
                fp16=False,
                verbose=False,
                condition_on_previous_text=False
            )

            print(
                "Skipping scene detection temporarily"
            )

            # TEMP FIX
            scene_list = []

            result = (
                transcribe_future.result()
            )

            print(
                "Whisper complete"
            )

            raw_segments = result[
                "segments"
            ]

            segments = []

            for seg in raw_segments:

                text = seg[
                    "text"
                ].strip()

                if text:

                    segments.append({

                        "text": text,

                        "start": float(
                            seg["start"]
                        ),

                        "end": float(
                            seg["end"]
                        )
                    })
    else:

        print(
            "Step G: Silent video"
        )

        scene_list = detect(
            video_path,
            ContentDetector()
        )

        segments = (
            transcribe_silent_video(
                video_path
            )
        )

    print(
        "Step H: Building embeddings"
    )

    texts = [
        str(s["text"])
        for s in segments
        if s.get("text")
    ]

    embeddings = embed_model.encode(
        texts
    ).astype("float32")

    print(
        "Step I: FAISS indexing"
    )

    index = faiss.IndexFlatL2(
        embeddings.shape[1]
    )

    index.add(
        embeddings
    )

    print(
        "Step J: Processing complete"
    )

    return {
        "index": index,
        "segments": segments,
        "texts": texts,
        "chapters": [
            float(scene[0].get_seconds())
            for scene in scene_list
        ],
        "has_audio": audio_present
    }
# =========================
# SEARCH / QA
# =========================

def search_video(
    query: str,
    index,
    texts: List[str],
    segments: List[Dict],
    has_audio: bool
) -> str:

    if not query.strip():

        return (
            "Error: Empty question."
        )

    # =========================
    # TIMESTAMP QUERY
    # =========================

    timestamp_matches = re.findall(

        r'(\d{1,2})[:.]'
        r'(\d{2})[:.]'
        r'(\d{2})',

        query
    )

    if len(timestamp_matches) >= 2:

        h1, m1, s1 = map(
            int,
            timestamp_matches[0]
        )

        h2, m2, s2 = map(
            int,
            timestamp_matches[1]
        )

        start_seconds = (
            h1 * 3600 +
            m1 * 60 +
            s1
        )

        end_seconds = (
            h2 * 3600 +
            m2 * 60 +
            s2
        )

        matching_segments = []

        for seg in segments:

            if (
                seg["start"] >= start_seconds
                and seg["end"] <= end_seconds
            ):

                matching_segments.append(
                    seg
                )

        # EXPAND CONTEXT FOR AUDIO

        if has_audio:

            expanded_segments = []

            for seg in matching_segments:

                idx = segments.index(seg)

                start_idx = max(
                    0,
                    idx - 3
                )

                end_idx = min(
                    len(segments),
                    idx + 4
                )

                expanded_segments.extend(
                    segments[start_idx:end_idx]
                )

            seen = set()

            final_segments = []

            for seg in expanded_segments:

                key = (
                    seg["start"],
                    seg["text"]
                )

                if key not in seen:

                    seen.add(key)

                    final_segments.append(seg)

            matching_segments = (
                final_segments
            )

        context = "\n".join([

            f"[{seg['start']:.1f}s] "
            f"{seg['text']}"

            for seg in matching_segments
        ])

    elif len(timestamp_matches) == 1:

        h, m, s = map(
            int,
            timestamp_matches[0]
        )

        target_seconds = (
            h * 3600 +
            m * 60 +
            s
        )

        nearby_segments = []

        for seg in segments:

            if abs(
                seg["start"] -
                target_seconds
            ) <= 10:

                nearby_segments.append(
                    seg
                )

        context = "\n".join([

            f"[{seg['start']:.1f}s] "
            f"{seg['text']}"

            for seg in nearby_segments
        ])

    else:

        # =========================
        # AUDIO RETRIEVAL
        # =========================

        if has_audio:

            query_vec = embed_model.encode(
                [query]
            ).astype("float32")

            D, I = index.search(
                query_vec,
                k=4
            )

            context_chunks = []

            used = set()

            for idx in I[0]:

                if idx in used:
                    continue

                used.add(idx)

                # BIGGER AUDIO CONTEXT WINDOW

                start = max(
                    0,
                    idx - 3
                )

                end = min(
                    len(texts),
                    idx + 4
                )

                window = " ".join(
                    texts[start:end]
                )
                if window in context_chunks:
                    continue            

                context_chunks.append(

                    f"[{segments[idx]['start']:.1f}s] "
                    f"{window}"
                )

            context = "\n".join(
                context_chunks
            )

        # =========================
        # SILENT VIDEO RETRIEVAL
        # =========================

        else:

            query_vec = embed_model.encode(
                [query]
            ).astype("float32")

            D, I = index.search(
                query_vec,
                k=2
            )

            context_chunks = []

            used = set()

            for idx in I[0]:

                if idx in used:
                    continue

                used.add(idx)

                context_chunks.append(

                    f"[{segments[idx]['start']:.1f}s] "
                    f"{texts[idx]}"
                )

            context = "\n".join(
                context_chunks
            )

    # =========================
    # AUDIO PROMPT
    # =========================

    if has_audio:

        prompt = f"""
You are an intelligent video assistant.

Rules:
- Answer ONLY using the provided context.
- Explain events naturally and accurately.
- Maintain chronological flow.
- Preserve important names, places, and details.
- Use proper spelling and grammar.
- Avoid hallucinating missing information.
- If information is unclear, say so.

Video Context:
{context}

Question:
{query}
"""

    # =========================
    # SILENT VIDEO PROMPT
    # =========================

    else:

        prompt = f"""
You are a video understanding assistant.

Your task:
Summarize the MAIN visible action happening in the video.

Rules:
- Use ONLY the provided context.
- Keep the answer SHORT.
- Focus on the primary action only.
- Treat similar captions as ONE continuous action.
- Do NOT count repeated frames as repeated events.
- Do NOT invent extra activity.
- Return 1-2 sentences maximum.

Video Context:
{context}

Question:
{query}
"""

    try:

        res = requests.post(

            "http://localhost:11434/api/generate",

            json={

                "model": "qwen2.5:3b",

                "prompt": prompt,

                "stream": False
            },

            timeout=180
        )

        res.raise_for_status()

        answer = res.json()[
            "response"
        ].strip()

        return answer

    except requests.exceptions.ConnectionError:

        return (
            "Error: Ollama is not running. "
            "Start it using: ollama serve"
        )

    except Exception as e:

        logger.error(
            f"LLM error: {e}"
        )

        return (
            "Error: Something went wrong "
            "with the LLM."
        )