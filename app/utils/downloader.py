import yt_dlp
import os
import uuid

DOWNLOAD_DIR = "downloads"

os.makedirs(
    DOWNLOAD_DIR,
    exist_ok=True
)

def download_video_from_url(url: str):

    unique_name = (
        f"{uuid.uuid4()}.mp4"
    )

    output_path = os.path.join(
        DOWNLOAD_DIR,
        unique_name
    )

    ydl_opts = {

        "format": "mp4",

        "outtmpl": output_path,

        "quiet": False,

        "noplaylist": True
    }

    with yt_dlp.YoutubeDL(
        ydl_opts
    ) as ydl:

        ydl.download([url])

    return output_path