import os

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

# loads the whole set of dish nutrition - basically an extension of the prev. defined load_dish_calories
# w/ a couple of additions to make nan safety easier. honestly not sure if therre even are nans in the database
# but doesnt hurt to be safe 
def load_dish_nutrition(dataset_path: str) -> pd.DataFrame:
    csv_path = Path(dataset_path) / "dish_nutrition_values.csv"
    df = pd.read_csv(csv_path)

    target_mappings = {
        "dish_id": ["dish_id", "id"],
        "calories": ["cal"],
        "mass": ["mass"],
        "fat": ["fat"],
        "protein": ["protein"],
        "carbohydrates": ["carb"]
    }

    found_cols = {}
    for target_name, substrings in target_mappings.items():
        for col in df.columns:
            if any(sub in col.lower() for sub in substrings):
                found_cols[target_name] = col
                break

    if "dish_id" not in found_cols:
        found_cols["dish_id"] = df.columns[0]

    if "calories" not in found_cols:
        raise ValueError(f"Could not find calories column. Columns: {df.columns.tolist()}")

    out = df[list(found_cols.values())].copy()
    out.columns = list(found_cols.keys())

    out["dish_id"] = (
        out["dish_id"]
        .astype(str)
        .str.replace("dish_", "", regex=False)
        .str.replace(".0", "", regex=False)
        .str.strip()
    )
    
    target_cols = ["calories", "mass", "fat", "protein", "carbohydrates"]
    for col in target_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
            
    cols_to_check = [c for c in target_cols if c in out.columns]
    
    original_len = len(out)
    out = out.dropna(subset=cols_to_check)
    print(f"Dropped {original_len - len(out)} rows due to missing nutritional values.")

    return out


def extract_dish_id_from_path(p: Path) -> Optional[str]:
    for part in p.parts:
        if part.startswith("dish_"):
            return part.replace("dish_", "") 
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
    
    # Load the updated dataframe
    dish_nutr = load_dish_nutrition(dataset_path)
    
    # Convert DataFrame to a dictionary of dictionaries for fast lookups
    # Example format: {'dish_1': {'calories': 500, 'mass': 200, ...}, ...}
    dish_to_nutr = dish_nutr.set_index("dish_id").to_dict(orient="index")

    rows = []

    if use_overhead:
        overhead_dir = root / "imagery" / "realsense_overhead"
        rgb_paths = list(overhead_dir.glob("*/rgb.png"))
        print(f"Found {len(rgb_paths)} overhead rgb images")

        for p in rgb_paths:
            dish_id = extract_dish_id_from_path(p)
            if dish_id is None or dish_id not in dish_to_nutr:
                continue

            nutr = dish_to_nutr[dish_id]
            rows.append({
                "image_path": str(p),
                "dish_id": dish_id,
                "calories": nutr.get("calories"),
                "mass": nutr.get("mass"),
                "fat": nutr.get("fat"),
                "protein": nutr.get("protein"),
                "carbohydrates": nutr.get("carbohydrates"),
            })

    if use_side_angles:
        side_dir = root / "imagery" / "side_angles"
        side_paths = list(side_dir.rglob("*.jpg")) + list(side_dir.rglob("*.png"))
        print(f"Found {len(side_paths)} side-angle images before filtering")

        per_dish_counts = {}
        for p in side_paths:
            dish_id = extract_dish_id_from_path(p)
            if dish_id is None or dish_id not in dish_to_nutr:
                continue

            count = per_dish_counts.get(dish_id, 0)
            if count >= max_images_per_dish:
                continue

            nutr = dish_to_nutr[dish_id]
            rows.append({
                "image_path": str(p),
                "dish_id": dish_id,
                "calories": nutr.get("calories"),
                "mass": nutr.get("mass"),
                "fat": nutr.get("fat"),
                "protein": nutr.get("protein"),
                "carbohydrates": nutr.get("carbohydrates"),
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

def ensure_dataset(repo_id: str, expected_subpaths=None) -> str:
    import os
    import kagglehub

    path = kagglehub.dataset_download(repo_id)
    print("Dataset path:", path)

    if expected_subpaths:
        for sp in expected_subpaths:
            full = os.path.join(path, sp)
            if not os.path.exists(full):
                raise RuntimeError(
                    f"Dataset downloaded, but expected path not found: {full}"
                )

    return path

def get_df_from_labels(dataset_path, cfg):
    label_file_name = "full_labels_built.csv"
    labels_csv = os.path.join("data_cache", label_file_name)
    os.makedirs("data_cache", exist_ok=True)

    if os.path.exists(labels_csv):
        print(f"Using existing labels file: {labels_csv}")
        df = pd.read_csv(labels_csv)
    else:
        print(f"{label_file_name} not found. Building it from dataset...")
        df = collect_image_label_table(
            dataset_path,
            use_overhead=cfg.use_overhead,
            use_side_angles=cfg.use_side_angles,
            max_images_total=cfg.max_images_total,
            max_images_per_dish=cfg.max_images_per_dish,
            seed=cfg.seed)
        df.to_csv(labels_csv, index=False)
        print(f"Saved labels to: {labels_csv}")

    # log transform everything - originally didn't do this and was only log transforming the calories but that was 
    # lowkey destroying my model, tried this in hopes that it was somewhat better lol
    target_cols = ["calories", "mass", "fat", "protein", "carbohydrates"]
    for col in target_cols:
        if col in df.columns:
            df[col] = np.log1p(df[col])

    return df