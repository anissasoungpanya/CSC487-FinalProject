import pandas as pd
from pathlib import Path
from typing import Optional


def load_dish_calories(dataset_path: str) -> pd.DataFrame:
    csv_path = Path(dataset_path) / "dish_nutrition_values.csv"
    df = pd.read_csv(csv_path)

    dish_col = None
    cal_col = None

    for c in "dish_id":
        if c in df.columns:
            dish_col = c
            break

    for c in "calories":
        if c in df.columns:
            cal_col = c
            break

    if dish_col is None:
        dish_col = df.columns[0]

    if cal_col is None:
        for c in df.columns:
            if "cal" in c.lower():
                cal_col = c
                break

    if cal_col is None:
        raise ValueError(f"Could not find calories column. Columns: {df.columns.tolist()}")

    out = df[[dish_col, cal_col]].copy()
    out.columns = ["dish_id", "calories"]

    out["dish_id"] = (
        out["dish_id"]
        .astype(str)
        .str.replace("dish_", "", regex=False)
        .str.replace(".0", "", regex=False)
        .str.strip()
    )
    out["calories"] = pd.to_numeric(out["calories"], errors="coerce")
    out = out.dropna(subset=["calories"])

    return out


def extract_dish_id_from_path(p):
    """
    Scans the path for a folder starting with 'dish_' and returns the full name.
    """
    for part in p.parts:
        if part.startswith("dish_"):
            return part
    return None


def collect_image_label_table(
    dataset_path: str,
    use_overhead: bool = True,
    use_side_angles: bool = False,
    max_images_total: Optional[int] = 20000,
    max_images_per_dish: int = 2,
    seed: int = 42,
) -> pd.DataFrame:
    root = Path(dataset_path)
    dish_cal = load_dish_calories(dataset_path)
    dish_to_cal = dict(zip(dish_cal["dish_id"], dish_cal["calories"]))

    rows = []

    if use_overhead:
        overhead_dir = root / "imagery" / "realsense_overhead"
        rgb_paths = list(overhead_dir.glob("*/rgb.png"))
        print(f"Found {len(rgb_paths)} overhead rgb images")

        for p in rgb_paths:
            dish_id = extract_dish_id_from_path(p)
            if dish_id is None:
                continue
            if dish_id not in dish_to_cal:
                continue

            rows.append({
                "image_path": str(p),
                "dish_id": dish_id,
                "calories": dish_to_cal[dish_id],
            })

    if use_side_angles:
        side_dir = root / "imagery" / "side_angles"
        side_paths = list(side_dir.rglob("*.jpg")) + list(side_dir.rglob("*.png"))
        print(f"Found {len(side_paths)} side-angle images before filtering")

        per_dish_counts = {}
        for p in side_paths:
            dish_id = extract_dish_id_from_path(p)
            if dish_id is None or dish_id not in dish_to_cal:
                continue

            count = per_dish_counts.get(dish_id, 0)
            if count >= max_images_per_dish:
                continue

            rows.append({
                "image_path": str(p),
                "dish_id": dish_id,
                "calories": dish_to_cal[dish_id],
            })
            per_dish_counts[dish_id] = count + 1

    df = pd.DataFrame(rows)
    print(f"Matched {len(df)} image/label pairs")

    if df.empty:
        raise RuntimeError("No image/label pairs created.")

    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    if max_images_total is not None and len(df) > max_images_total:
        df = df.iloc[:max_images_total].copy()

    return df