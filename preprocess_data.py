import os
import re
import pandas as pd
from pathlib import Path
from typing import Optional, List

def load_dish_calories(dataset_path: str) -> pd.DataFrame:
    """
    returns DataFrame with columns: dish_id, calories
    """
    csv_path = os.path.join(dataset_path, "dish_nutrition_values.csv")
    df = pd.read_csv(csv_path)

    col_candidates = {
        "dish_id": ["dish_id", "dish", "id"],
        "calories": ["calories", "kcal", "energy", "calorie"],
    }

    def find_col(df_cols, options):
        for c in options:
            if c in df_cols:
                return c
        return None

    dish_col = find_col(df.columns, col_candidates["dish_id"])
    cal_col = find_col(df.columns, col_candidates["calories"])

    if dish_col is None or cal_col is None:
        raise ValueError(f"Could not find dish_id/calories columns in {csv_path}. Columns: {df.columns.tolist()}")

    out = df[[dish_col, cal_col]].copy()
    out.columns = ["dish_id", "calories"]
    out["dish_id"] = out["dish_id"].astype(str)
    out["calories"] = out["calories"].astype(float)
    return out


def extract_dish_id_from_path(p: Path) -> Optional[str]:
    """
    tries to extract dish_id from common Nutrition5k-like path patterns
    """
    s = str(p)

    # Case 1: dish_id is the parent directory name: .../overhead/<dish_id>/image.jpg
    parent = p.parent.name
    if re.fullmatch(r"\d+", parent):
        return parent
    if re.fullmatch(r"dish[_-]?\d+", parent, flags=re.IGNORECASE):
        return re.sub(r"\D", "", parent)

    # Case 2: dish_id is in filename: <dish_id>.jpg or dish_<dish_id>_*.jpg
    stem = p.stem  # filename without extension
    m = re.search(r"(\d+)", stem)
    if m:
        return m.group(1)

    return None


def collect_image_label_table(
    dataset_path: str,
    use_overhead: bool = True,
    use_side_angles: bool = False,
    max_images_total: Optional[int] = 20000,
    max_images_per_dish: int = 2,
    seed: int = 42
) -> pd.DataFrame:
    """
    Builds a table with columns: image_path, calories, dish_id
    - max_images_total: cap overall images for speed (None = no cap)
    - max_images_per_dish: if side_angles produces many frames, cap per dish
    """
    rng = pd.Series(range(10))  # placeholder

    dish_cal = load_dish_calories(dataset_path)
    dish_to_cal = dict(zip(dish_cal["dish_id"], dish_cal["calories"]))

    img_paths: List[Path] = []
    imagery_root = Path(dataset_path) / "imagery"

    if use_overhead:
        overhead_dir = imagery_root / "realsense_overhead"
        if overhead_dir.exists():
            img_paths += list(overhead_dir.rglob("*.jpg")) + list(overhead_dir.rglob("*.png"))

    if use_side_angles:
        side_dir = imagery_root / "side_angles"
        if side_dir.exists():
            img_paths += list(side_dir.rglob("*.jpg")) + list(side_dir.rglob("*.png"))

    rows = []
    for p in img_paths:
        dish_id = extract_dish_id_from_path(p)
        if dish_id is None:
            continue
        if dish_id not in dish_to_cal:
            continue
        rows.append({"image_path": str(p), "dish_id": dish_id, "calories": dish_to_cal[dish_id]})

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(
            "No image/label pairs created. You likely need to tweak extract_dish_id_from_path() "
            "based on the actual filenames/folders."
        )

    # if side angles included, cap per dish to avoid 100s of similar frames
    if use_side_angles and max_images_per_dish is not None:
        df = (
            df.groupby("dish_id", group_keys=False)
              .apply(lambda g: g.sample(n=min(len(g), max_images_per_dish), random_state=seed))
              .reset_index(drop=True)
        )

    # shuffle
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # cap overall images
    if max_images_total is not None and len(df) > max_images_total:
        df = df.iloc[:max_images_total].copy()

    return df