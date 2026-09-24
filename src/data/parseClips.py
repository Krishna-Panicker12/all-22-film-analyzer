from pathlib import Path
import csv
import shutil
import subprocess


GAME_NAME = "game_01"

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_VIDEO = PROJECT_ROOT / "data" / "raw" / f"{GAME_NAME}.mp4"
METADATA_CSV = PROJECT_ROOT / "data" / "metadata" / f"{GAME_NAME}_md.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "plays" / GAME_NAME

REQUIRED_COLUMNS = {
    "play_id",
    "clip_file",
    "clip_start",
    "tracking",
    "clip_end",
}


def timestamp_to_seconds(timestamp):
    timestamp = timestamp.strip()
    parts = timestamp.split(":")

    if len(parts) != 3:
        raise ValueError(
            f"Invalid timestamp {timestamp!r}. Expected HH:MM:SS."
        )

    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])

    if hours < 0:
        raise ValueError(f"Invalid hour value in {timestamp!r}")

    if not 0 <= minutes < 60:
        raise ValueError(f"Invalid minute value in {timestamp!r}")

    if not 0 <= seconds < 60:
        raise ValueError(f"Invalid second value in {timestamp!r}")

    return hours * 3600 + minutes * 60 + seconds


def get_video_duration(video_path):
    command = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True,
    )

    return float(result.stdout.strip())


def load_and_validate_metadata(csv_path, video_duration):
    with csv_path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError("Metadata CSV has no header.")

        missing_columns = REQUIRED_COLUMNS - set(reader.fieldnames)

        if missing_columns:
            raise ValueError(
                f"Metadata CSV is missing required columns: "
                f"{sorted(missing_columns)}"
            )

        rows = []
        seen_play_ids = set()
        seen_clip_files = set()

        for line_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(
                    f"CSV line {line_number} has extra fields. "
                    "Check for two plays joined onto one line "
                    "or an unexpected comma."
                )

            play_id = row["play_id"].strip()
            clip_file = row["clip_file"].strip()

            if not play_id:
                raise ValueError(
                    f"Line {line_number}: missing play_id."
                )

            if not clip_file:
                raise ValueError(
                    f"Line {line_number}: missing clip_file."
                )

            if play_id in seen_play_ids:
                raise ValueError(
                    f"Duplicate play_id found: {play_id}"
                )

            if clip_file in seen_clip_files:
                raise ValueError(
                    f"Duplicate clip_file found: {clip_file}"
                )

            seen_play_ids.add(play_id)
            seen_clip_files.add(clip_file)

            start = timestamp_to_seconds(row["clip_start"])
            tracking = timestamp_to_seconds(row["tracking"])
            end = timestamp_to_seconds(row["clip_end"])

            if not start < tracking < end:
                raise ValueError(
                    f"\nInvalid timing for {play_id}\n"
                    f"clip_start: {row['clip_start']}\n"
                    f"tracking:   {row['tracking']}\n"
                    f"clip_end:   {row['clip_end']}\n\n"
                    "Expected: clip_start < tracking < clip_end"
                )

            if end > video_duration + 0.5:
                raise ValueError(
                    f"{play_id}: clip_end {row['clip_end']} "
                    "is beyond the source video's duration."
                )

            rows.append(row)

    return rows


def cut_clip(source_video, row, output_path):
    start_seconds = timestamp_to_seconds(row["clip_start"])
    end_seconds = timestamp_to_seconds(row["clip_end"])
    duration = end_seconds - start_seconds

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-ss", row["clip_start"],
        "-i", str(source_video),
        "-t", f"{duration:.3f}",
        "-map", "0:v:0",
        "-map", "0:a:0?",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "160k",
        "-movflags", "+faststart",
        str(output_path),
    ]

    subprocess.run(command, check=True)


def main():
    if shutil.which("ffmpeg") is None:
        raise SystemExit("FFmpeg was not found on PATH.")

    if shutil.which("ffprobe") is None:
        raise SystemExit("ffprobe was not found on PATH.")

    if not SOURCE_VIDEO.exists():
        raise SystemExit(f"Source video not found:\n{SOURCE_VIDEO}")

    if not METADATA_CSV.exists():
        raise SystemExit(f"Metadata CSV not found:\n{METADATA_CSV}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    video_duration = get_video_duration(SOURCE_VIDEO)

    print(f"Source video: {SOURCE_VIDEO.name}")
    print(f"Duration: {video_duration:.2f} seconds\n")

    rows = load_and_validate_metadata(
        METADATA_CSV,
        video_duration,
    )

    print(f"Metadata validated: {len(rows)} plays")
    print(f"Output folder: {OUTPUT_DIR}\n")

    for index, row in enumerate(rows, start=1):
        output_path = OUTPUT_DIR / row["clip_file"].strip()

        clip_start = timestamp_to_seconds(row["clip_start"])
        tracking = timestamp_to_seconds(row["tracking"])
        tracking_offset = tracking - clip_start

        print(
            f"[{index:02d}/{len(rows):02d}] "
            f"{row['play_id']} -> {output_path.name}"
        )
        print(
            f"     clip: {row['clip_start']} to {row['clip_end']}"
        )
        print(
            f"     tracking starts {tracking_offset:.2f}s into clip"
        )

        cut_clip(
            SOURCE_VIDEO,
            row,
            output_path,
        )

    print(f"\nFinished. Created {len(rows)} clips in:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
