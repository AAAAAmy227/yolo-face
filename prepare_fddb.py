"""
Download FDDB dataset from Roboflow and organize in YOLO format for training.

Usage:
    1. Get a free Roboflow API key from https://app.roboflow.com/settings/api
    2. python prepare_fddb.py --api-key YOUR_KEY
    3. python train.py --data-dir datasets/fddb

Output structure:
  datasets/fddb/
    images/train/
    images/val/
    labels/train/
    labels/val/
"""

import argparse
import os
import shutil
from pathlib import Path

try:
    from roboflow import Roboflow
except ImportError:
    print("Please install roboflow: pip install roboflow")
    raise

# Roboflow dataset identifiers (public FDDB dataset)
WORKSPACE = "fddb"
PROJECT = "face-detection-40nq0"
VERSION = 1


def organize_yolo_split(src_split_dir, images_out, labels_out):
    """Copy images + labels from a Roboflow YOLO split dir into our structure."""
    img_dir = os.path.join(src_split_dir, "images")
    lbl_dir = os.path.join(src_split_dir, "labels")

    if not os.path.isdir(img_dir):
        return 0

    os.makedirs(images_out, exist_ok=True)
    os.makedirs(labels_out, exist_ok=True)

    count = 0
    for fname in os.listdir(img_dir):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in (".jpg", ".jpeg", ".png", ".bmp"):
            continue
        src_img = os.path.join(img_dir, fname)
        dst_img = os.path.join(images_out, fname)
        shutil.copy2(src_img, dst_img)

        # Corresponding label file (.txt)
        label_name = os.path.splitext(fname)[0] + ".txt"
        src_lbl = os.path.join(lbl_dir, label_name)
        if os.path.exists(src_lbl):
            shutil.copy2(src_lbl, os.path.join(labels_out, label_name))

        count += 1

    return count


def main():
    parser = argparse.ArgumentParser(
        description="Download FDDB from Roboflow and prepare for YOLO training"
    )
    parser.add_argument("--api-key", required=True, help="Roboflow API key (free from app.roboflow.com/settings/api)")
    parser.add_argument("--output-dir", default="./datasets/fddb", help="Output directory")
    parser.add_argument("--keep-download", action="store_true",
                        help="Keep the raw Roboflow download (for debugging)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    roboflow_dir = output_dir / "_roboflow_download"

    # ── Step 1: Download from Roboflow ──────────────────────────
    print("=" * 60)
    print("Step 1/2: Downloading FDDB from Roboflow")
    print("=" * 60)
    print(f"  Workspace: {WORKSPACE}")
    print(f"  Project:   {PROJECT}")
    print(f"  Version:   {VERSION}")
    print(f"  Format:    YOLOv5/v8")
    print()

    if roboflow_dir.exists():
        print(f"  Already downloaded at {roboflow_dir}")
    else:
        rf = Roboflow(api_key=args.api_key)
        project = rf.workspace(WORKSPACE).project(PROJECT)
        dataset = project.version(VERSION)
        # Downloads YOLO-format data (labels are class_id xc yc w h)
        dataset.download(model_format="yolov8", location=str(roboflow_dir))
        print(f"  Downloaded to {roboflow_dir}")

    # ── Step 2: Organize into train / val ───────────────────────
    print("\n" + "=" * 60)
    print("Step 2/2: Organizing into YOLO training structure")
    print("=" * 60)

    total = 0
    # Roboflow creates subdirs: train/, valid/, test/
    split_map = {
        "train": "train",
        "val": "valid",   # Roboflow calls it "valid"
    }
    # Also check for "test" → merge into val
    test_dir = roboflow_dir / "test"
    has_test = test_dir.is_dir()

    for our_split, rf_split in split_map.items():
        src = roboflow_dir / rf_split
        imgs_out = output_dir / "images" / our_split
        lbls_out = output_dir / "labels" / our_split
        count = organize_yolo_split(str(src), str(imgs_out), str(lbls_out))
        print(f"  [{our_split}] {count} images")
        total += count

    # Merge test → val if present
    if has_test:
        imgs_out = output_dir / "images" / "val"
        lbls_out = output_dir / "labels" / "val"
        count = organize_yolo_split(str(test_dir), str(imgs_out), str(lbls_out))
        print(f"  [val + test] {count} test images merged into val")
        total += count

    # Cleanup
    if not args.keep_download:
        shutil.rmtree(roboflow_dir, ignore_errors=True)
        print(f"  Removed temporary download: {roboflow_dir}")

    print(f"\nDone! Dataset prepared at: {output_dir}")
    print(f"Total images: {total}")
    print("=" * 60)
    print("\nTo train:")
    print(f"  python train.py --data-dir {output_dir}")


if __name__ == "__main__":
    main()
