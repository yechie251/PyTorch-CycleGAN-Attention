
# split_data.py
# Prepare CycleGAN dataset structure (train/test and A/B) using command-line paths.

import random
import shutil
import argparse
from pathlib import Path

# Domain mapping: A = Photos, B = Monet
A_SRC_NAME = "photo_jpg"   # Domain A source folder name
B_SRC_NAME = "monet_jpg"   # Domain B source folder name

TRAIN_SPLIT = 0.9          # Train ratio (rest goes to test)
RANDOM_SEED = 42           # For reproducibility
CLEAR_DST = False          # If True, clear existing files in destination folders before copying

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(folder: Path):
    """Return a sorted list of image files under `folder` (recursively)."""
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS])


def copy_files(files, dst: Path):
    """Copy a list of files to `dst` (keeps original filenames)."""
    dst.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copy2(f, dst / f.name)


def clear_dir(p: Path):
    """Delete files (not folders) under `p` recursively."""
    if not p.exists():
        return
    for child in p.rglob("*"):
        try:
            if child.is_file() or child.is_symlink():
                child.unlink(missing_ok=True)
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Split CycleGAN dataset into train/test A/B folders")
    parser.add_argument("--src", type=str, required=True, help="Path to source root (where photo_jpg and monet_jpg are)")
    parser.add_argument("--dst", type=str, required=True, help="Path to destination dataset root")

    args = parser.parse_args()

    SRC_ROOT = Path(args.src)
    DST_ROOT = Path(args.dst)

    # Resolve source folders
    A_src = SRC_ROOT / A_SRC_NAME
    B_src = SRC_ROOT / B_SRC_NAME

    print("SRC_ROOT:", SRC_ROOT, "exists?", SRC_ROOT.exists())
    print("A_src:", A_src, "exists?", A_src.exists())
    print("B_src:", B_src, "exists?", B_src.exists())

    assert A_src.exists(), f"Source folder not found: {A_src}"
    assert B_src.exists(), f"Source folder not found: {B_src}"

    # Collect images
    files_A = collect_images(A_src)
    files_B = collect_images(B_src)
    assert files_A, f"No images found in {A_src}"
    assert files_B, f"No images found in {B_src}"

    # Split into train/test
    random.seed(RANDOM_SEED)
    random.shuffle(files_A)
    random.shuffle(files_B)

    kA = max(1, int(len(files_A) * TRAIN_SPLIT))
    kB = max(1, int(len(files_B) * TRAIN_SPLIT))

    train_A, test_A = files_A[:kA], files_A[kA:]
    train_B, test_B = files_B[:kB], files_B[kB:]

    # Build destination structure
    trainA = DST_ROOT / "train" / "A"
    trainB = DST_ROOT / "train" / "B"
    testA  = DST_ROOT / "test" / "A"
    testB  = DST_ROOT / "test" / "B"
    for p in [trainA, trainB, testA, testB]:
        p.mkdir(parents=True, exist_ok=True)

    # Optional: clear existing files
    if CLEAR_DST:
        for p in [trainA, trainB, testA, testB]:
            clear_dir(p)

    # Copy files
    copy_files(train_A, trainA)
    copy_files(train_B, trainB)
    copy_files(test_A,  testA)
    copy_files(test_B,  testB)

    # Summary
    print("✅ Done")
    print(f"A (photos): {len(train_A)} train, {len(test_A)} test  →  {trainA} | {testA}")
    print(f"B (monet):  {len(train_B)} train, {len(test_B)} test  →  {trainB} | {testB}")


if __name__ == "__main__":
    main()
