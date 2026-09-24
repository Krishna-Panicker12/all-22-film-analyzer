from pathlib import Path
import subprocess
import sys


YOUTUBE_URL = "https://www.youtube.com/watch?v=Y722eYPrsts"
GAME_NAME = "game_01"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"


def main():
    if YOUTUBE_URL == "PASTE_YOUTUBE_URL_HERE":
        raise SystemExit("Paste the YouTube URL into YOUTUBE_URL first.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_template = OUTPUT_DIR / f"{GAME_NAME}.%(ext)s"

    command = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--no-playlist",
        "--continue",
        "--retries", "10",
        "--fragment-retries", "10",
        "--concurrent-fragments", "4",
        "-f",
        "bv*[ext=mp4][height<=1080]+ba[ext=m4a]/b[ext=mp4][height<=1080]",
        "--merge-output-format", "mp4",
        "-o", str(output_template),
        YOUTUBE_URL,
    ]

    print(f"Downloading {GAME_NAME}...")
    print(f"Output folder: {OUTPUT_DIR}\n")

    subprocess.run(command, check=True)

    expected_file = OUTPUT_DIR / f"{GAME_NAME}.mp4"

    if expected_file.exists():
        print(f"\nDownload complete: {expected_file}")
    else:
        print(f"\nDownload finished. Check: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
