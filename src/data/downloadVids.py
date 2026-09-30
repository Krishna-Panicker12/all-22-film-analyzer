from pathlib import Path
import subprocess
import sys
import argparse


YOUTUBE_URL = "https://www.youtube.com/watch?v=Y722eYPrsts"
GAME_NAME = "game_01"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"


def main():
    parser=argparse.ArgumentParser(description='Download a supplied URL into the game.mp4 ingestion slot.')
    parser.add_argument('url')
    parser.add_argument('--output',type=Path,default=OUTPUT_DIR/'game.mp4')
    args=parser.parse_args()
    if args.output.exists(): raise SystemExit('Output already exists; choose another --output to preserve it.')
    args.output.parent.mkdir(parents=True,exist_ok=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_template = args.output.with_suffix('.%(ext)s')

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
        args.url,
    ]

    print(f"Downloading {GAME_NAME}...")
    print(f"Output folder: {OUTPUT_DIR}\n")

    subprocess.run(command, check=True)

    expected_file = args.output

    if expected_file.exists():
        print(f"\nDownload complete: {expected_file}")
    else:
        print(f"\nDownload finished. Check: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
