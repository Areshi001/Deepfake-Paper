"""
Extract facial regions (periocular, mouth, cheeks) from frames using dlib.
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.utils import DIRS

# ---------------------------------------------------------------------------
# Landmark indices (68-point model)
# ---------------------------------------------------------------------------
REGION_INDICES = {
    "periocular": list(range(36, 48)),
    "mouth": list(range(48, 68)),
    "cheeks": list(range(1, 6)) + list(range(11, 16)),
}

EXPANSION = 15  # pixels to expand bounding box


def _init_dlib():
    """Lazy-import dlib and load models."""
    import dlib

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    return detector, predictor


def crop_regions(row, detector, predictor):
    """
    Crop all facial regions from a single frame.

    Returns a list of dicts, one per successfully cropped region.
    """
    img = cv2.imread(row["path"])
    if img is None:
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)
    if not faces:
        return []

    shape = predictor(gray, faces[0])
    h, w = img.shape[:2]
    results = []

    for region_name, idxs in REGION_INDICES.items():
        pts = np.array([[shape.part(i).x, shape.part(i).y] for i in idxs])
        x1, y1 = pts.min(axis=0) - EXPANSION
        x2, y2 = pts.max(axis=0) + EXPANSION
        crop = img[max(0, y1) : min(h, y2), max(0, x1) : min(w, x2)]
        if crop.size == 0:
            continue
        crop = cv2.resize(crop, (224, 224))

        out_dir = (
            DIRS["regions"]
            / region_name
            / row["split"]
            / ("real" if row["label"] == 0 else "fake")
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = out_dir / Path(row["path"]).name.replace(".jpg", f"_{region_name}.jpg")
        cv2.imwrite(str(fname), crop)

        results.append(
            {
                "path": str(fname),
                "region": region_name,
                "label": row["label"],
                "split": row["split"],
                "video_id": row["video_id"],
            }
        )
    return results


def extract_all_regions(
    frames_split_csv: Path,
    max_workers: int = 4,
) -> pd.DataFrame:
    """
    Extract periocular, mouth, and cheek regions for every frame.

    Parameters
    ----------
    frames_split_csv : Path
        Path to the split frame catalogue CSV.
    max_workers : int
        Thread-pool size.

    Returns
    -------
    pd.DataFrame
        Region-level catalogue.
    """
    detector, predictor = _init_dlib()
    df = pd.read_csv(frames_split_csv)
    rows_list = list(df.itertuples(index=False))

    region_records: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(crop_regions, row._asdict(), detector, predictor)
            for row in rows_list
        ]
        for i, f in enumerate(as_completed(futures)):
            region_records.extend(f.result())
            if i % 500 == 0:
                print(f"  {i}/{len(futures)} frames processed")

    df_reg = pd.DataFrame(region_records)
    df_reg.to_csv(DIRS["metrics"] / "region_frames.csv", index=False)
    print("Region crops done")
    print(df_reg.groupby(["region", "split", "label"]).size())
    return df_reg
