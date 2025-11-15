import os
import shutil
import kagglehub
from pathlib import Path


def download_dataset(dataset_ref: str):
    """
    dataset_ref -> ex: "rupankarmajumdar/crop-pests-dataset"
    """

    # ------------------------------------------------
    # 1. Get the project root folder
    #    (one level above where this script is located)
    # ------------------------------------------------
    project_root = Path(__file__).resolve().parents[1]  # <-- key change!
    print(f"📂 Project root detected as: {project_root}")

    # ------------------------------------------------
    # 2. Create datasets/ folder under project root
    # ------------------------------------------------
    dataset_name = dataset_ref.split("/")[-1]
    target_path = project_root / "AgriPest" / dataset_name
    target_path.mkdir(parents=True, exist_ok=True)
    print(f"📁 Will save dataset to: {target_path}")

    # ------------------------------------------------
    # 3. Download via KaggleHub
    # ------------------------------------------------
    print("⬇ Downloading from KaggleHub...")
    cache_path = Path(kagglehub.dataset_download(dataset_ref))
    print(f"✅ Cached at: {cache_path}")

    # ------------------------------------------------
    # 4. Copy into project datasets folder
    # ------------------------------------------------
    print("📦 Copying into project...")

    for item in cache_path.iterdir():
        dest = target_path / item.name
        if item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dest)

    print(f"✅ Done → {target_path}\n")


if __name__ == "__main__":
    download_dataset("rupankarmajumdar/crop-pests-dataset")