import yt_dlp
import os
import uuid
import time


# Always use the backend folder as base path
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

DOWNLOAD_DIR = os.path.join(
    BASE_DIR,
    "downloads"
)

os.makedirs(
    DOWNLOAD_DIR,
    exist_ok=True
)


def download_video_from_url(url: str):

    unique_name = f"{uuid.uuid4()}.mp4"

    output_path = os.path.join(
        DOWNLOAD_DIR,
        unique_name
    )

    ydl_opts = {
        "format": "bestvideo+bestaudio/best",
        "merge_output_format": "mp4",
        "outtmpl": output_path,
        "quiet": False,
        "noplaylist": True
    }

    with yt_dlp.YoutubeDL(
        ydl_opts
    ) as ydl:

        ydl.download([url])

    timeout = 30
    start = time.time()

    while True:

        if os.path.exists(output_path):
            break

        print(
            f"Waiting for download: {output_path}"
        )

        time.sleep(1)

        if time.time() - start > timeout:

            raise Exception(
                "Download timeout"
            )

    print(
        f"Downloaded successfully: {output_path}"
    )

    return output_path