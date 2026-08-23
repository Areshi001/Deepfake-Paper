"""
Download FF-C23 from Kaggle and extract frames using GPU-accelerated FFmpeg.
"""

import json
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from src.utils import DIRS


# ---------------------------------------------------------------------------
# FFmpeg helpers
# ---------------------------------------------------------------------------
def get_duration(vid_path: Path) -> float:
    """Return video duration in seconds via ffprobe."""
    r = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            str(vid_path),
        ],
        capture_output=True,
        text=True,
    )
    try:
        return float(r.stdout.strip())
    except (ValueError, AttributeError):
        return 0.0


def extract_frames_gpu(vid_path: Path, out_dir: Path, label: int, n_frames: int = 20):
    """Extract *n_frames* evenly spaced frames from a video using GPU-accelerated FFmpeg."""
    out_dir.mkdir(parents=True, exist_ok=True)
    duration = get_duration(vid_path)
    if duration == 0:
        return []
    fps = n_frames / duration
    cmd = [
        "ffmpeg",
        "-loglevel", "error",
        "-hwaccel", "cuda",
        "-i", str(vid_path),
        "-vf", f"fps={fps},scale=224:224",
        "-vframes", str(n_frames),
        "-q:v", "2",
        str(out_dir / f"{vid_path.stem}_%03d.jpg"),
        "-y",
    ]
    subprocess.run(cmd, capture_output=True)

    rows = []
    for f in sorted(out_dir.glob(f"{vid_path.stem}_*.jpg"))[:n_frames]:
        rows.append({"path": str(f), "label": label, "video_id": vid_path.stem})
    return rows


# ---------------------------------------------------------------------------
# High-level extraction
# ---------------------------------------------------------------------------
def extract_all_frames(
    real_dir: Path,
    fake_dir: Path,
    n_frames: int = 20,
    max_workers: int = 8,
    sample_per_class: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Extract frames from all real and fake videos.

    Parameters
    ----------
    real_dir : Path
        Directory containing original (.mp4) videos.
    fake_dir : Path
        Directory containing DeepFake (.mp4) videos.
    n_frames : int
        Number of frames to extract per video.
    max_workers : int
        Thread-pool size.
    sample_per_class : int or None
        If set, randomly sample this many videos per class.
    seed : int
        Random seed for subsampling.

    Returns
    -------
    pd.DataFrame
        Frame-level catalogue with columns ``path``, ``label``, ``video_id``.
    """
    import random

    real_vids = sorted(real_dir.glob("*.mp4"))
    fake_vids = sorted(fake_dir.glob("*.mp4"))

    if sample_per_class is not None:
        random.seed(seed)
        real_vids = random.sample(real_vids, min(sample_per_class, len(real_vids)))
        fake_vids = random.sample(fake_vids, min(sample_per_class, len(fake_vids)))

    tasks = (
        [(v, DIRS["frames"] / "real" / v.stem, 0) for v in real_vids]
        + [(v, DIRS["frames"] / "fake" / v.stem, 1) for v in fake_vids]
    )
    print(f"Total videos: {len(tasks)}")

    records: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(extract_frames_gpu, *t, n_frames): t for t in tasks}
        done = 0
        for f in as_completed(futures):
            records.extend(f.result())
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(tasks)} | {len(records)} frames")

    df = pd.DataFrame(records)
    df.to_csv(DIRS["splits"] / "frames.csv", index=False)
    print(f"\nExtraction done: {len(df)} frames")
    print(df.groupby("label").size())
    return df


# ---------------------------------------------------------------------------
# Train / Val / Test split
# ---------------------------------------------------------------------------
def split_dataset(frames_csv: Path) -> pd.DataFrame:
    """
    Perform a video-level stratified 75/15/10 train/val/test split.
    """
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(frames_csv)
    vid_df = df.drop_duplicates("video_id")[["video_id", "label"]].reset_index(drop=True)

    train_v, temp_v = train_test_split(
        vid_df, test_size=0.25, stratify=vid_df.label, random_state=42
    )
    val_v, test_v = train_test_split(
        temp_v, test_size=0.40, stratify=temp_v.label, random_state=42
    )

    split_map = {
        **dict.fromkeys(train_v.video_id, "train"),
        **dict.fromkeys(val_v.video_id, "val"),
        **dict.fromkeys(test_v.video_id, "test"),
    }
    df["split"] = df["video_id"].map(split_map)
    df.to_csv(DIRS["splits"] / "frames_split.csv", index=False)
    print("Split done")
    print(df.groupby(["split", "label"]).size())
    return df
