from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

from app.core_engine import process_video
from app.topic_engine import generate_topic_timeline
from app.state import video_memory, embed_model

from app.utils.downloader import (
    download_video_from_url
)

import shutil
import os
import re
import time
import subprocess

router = APIRouter()

# =========================
# SUPPORTED VIDEO FORMATS
# =========================

ALLOWED_EXTENSIONS = {

    '.mp4', '.avi', '.mov', '.mkv', '.webm',
    '.flv', '.wmv', '.m4v', '.3gp', '.ts',
    '.mpeg', '.mpg', '.ogv', '.rm', '.rmvb',
    '.divx', '.f4v', '.asf', '.vob'
}

# =========================
# VIDEO CONVERSION
# =========================

def convert_to_mp4(input_path: str) -> str:
    """
    Convert any video format to mp4 using FFmpeg.
    """

    output_path = (
        input_path.rsplit('.', 1)[0]
        + '_converted.mp4'
    )

    cmd = [

        "ffmpeg",
        "-y",

        "-i", input_path,

        "-c:v", "libx264",

        "-c:a", "aac",

        "-strict", "experimental",

        output_path
    ]

    result = subprocess.run(

        cmd,

        capture_output=True,

        text=True
    )

    if result.returncode != 0:

        raise RuntimeError(

            f"FFmpeg conversion failed: "
            f"{result.stderr}"
        )

    return output_path

# =========================
# FORMAT CHECK
# =========================

def is_mp4_compatible(
    file_path: str
) -> bool:
    """
    Check if file already works well.
    """

    ext = os.path.splitext(
        file_path
    )[1].lower()

    return ext in {

        '.mp4',
        '.webm'
    }

# =========================
# PROCESS VIDEO API
# =========================

@router.post("/process-video")

async def process_video_api(

    file: UploadFile = File(None),

    url: str = Form(None)
):

    # =========================
    # URL INPUT
    # =========================

    if url and url != "string":

        try:

            processing_path = (
                download_video_from_url(
                    url
                )
            )

        except Exception as e:

            raise HTTPException(

                status_code=500,

                detail=(
                    f"Video download failed: "
                    f"{str(e)}"
                )
            )

        filename = os.path.basename(
            processing_path
        )

        ext = os.path.splitext(
            filename
        )[1].lower()

        converted = False

    # =========================
    # FILE UPLOAD INPUT
    # =========================

    elif file:

        filename = (
            file.filename
            or "unnamed_video"
        )

        ext = os.path.splitext(
            filename
        )[1].lower()

        if ext not in ALLOWED_EXTENSIONS:

            raise HTTPException(

                status_code=400,

                detail=(
                    f"Unsupported format "
                    f"'{ext}'. "
                    f"Supported: "
                    f"{', '.join(sorted(ALLOWED_EXTENSIONS))}"
                )
            )

        os.makedirs(
            "data",
            exist_ok=True
        )

        safe_name = re.sub(

            r'[^\w\-.]',

            '_',

            os.path.basename(
                filename
            )
        )

        file_path = (
            f"data/{safe_name}"
        )

        # =========================
        # HANDLE DUPLICATES
        # =========================

        base, extension = os.path.splitext(
            safe_name
        )

        counter = 1

        while os.path.exists(
            file_path
        ):

            file_path = (
                f"data/"
                f"{base}_{counter}"
                f"{extension}"
            )

            counter += 1

        # =========================
        # SAVE UPLOADED FILE
        # =========================

        with open(
            file_path,
            "wb"
        ) as f:

            shutil.copyfileobj(
                file.file,
                f
            )

        processing_path = file_path

        converted = False

        # =========================
        # CONVERT IF NEEDED
        # =========================

        if not is_mp4_compatible(
            file_path
        ):

            try:

                processing_path = (
                    convert_to_mp4(
                        file_path
                    )
                )

                converted = True

            except RuntimeError as e:

                os.remove(
                    file_path
                )

                raise HTTPException(

                    status_code=500,

                    detail=str(e)
                )

    # =========================
    # NO INPUT
    # =========================

    else:

        raise HTTPException(

            status_code=400,

            detail=(
                "No file or URL provided."
            )
        )

    # =========================
    # PROCESS VIDEO
    # =========================

    try:

        data = process_video(
            processing_path
        )

        data['timestamp'] = (
            time.time()
        )

        data['file_path'] = (
            os.path.abspath(
                processing_path
            )
        )

        print(
            "Saving to memory:",
            data["file_path"]
        )

        print(
            "Exists while saving:",
            os.path.exists(
                data["file_path"]
            )
        )

        data['original_format'] = ext

        data['topics'] = (
            generate_topic_timeline(
                data['segments'],
                embed_model
            )
        )

        video_id = os.path.basename(
            processing_path
        )

        video_memory[video_id] = data

        response = {

            "status": "success",

            "video_id": video_id,

            "original_format": ext,

            "converted_to_mp4": converted,

            "has_audio": data.get(
                "has_audio",
                True
            ),

            "analysis_method":

                "whisper"

                if data.get(
                    "has_audio",
                    True
                )

                else

                "llava_visual",

            "chapters": data[
                "chapters"
            ],

            "topics": data[
                "topics"
            ],

            "segment_count": len(
                data["segments"]
            )
        }

        return response

    except Exception as e:

        raise HTTPException(

            status_code=500,

            detail=str(e)
        )

    finally:

        # =========================
        # KEEP VIDEO FOR CLIP EXPORT
        # =========================

        # Do not delete the downloaded video.
        # Clip export needs the original file path.

        pass

# =========================
# LIST VIDEOS
# =========================

@router.get("/videos")

async def list_videos():

    return {

        "videos": [

            {

                "video_id": vid,

                "timestamp": data.get(
                    "timestamp"
                ),

                "original_format": data.get(
                    "original_format",
                    "unknown"
                ),

                "topics": data.get(
                    "topics",
                    []
                )
            }

            for vid, data in video_memory.items()
        ]
    }

# =========================
# DELETE VIDEO
# =========================

@router.delete("/video/{video_id}")

async def delete_video(
    video_id: str
):

    if video_id in video_memory:

        del video_memory[
            video_id
        ]

        return {

            "status": "deleted",

            "video_id": video_id
        }

    raise HTTPException(

        status_code=404,

        detail="Video not found"
    )