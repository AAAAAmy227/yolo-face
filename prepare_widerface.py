"""
Download WIDER FACE dataset and convert to YOLO format.

WIDER FACE annotation format:
  Image_name
  num_faces
  x1 y1 w h blur expression illumination occlusion pose ...

YOLO format (per image):
  class_id x_center y_center width height  (all normalized to [0,1])

Output structure:
  datasets/widerface/
    images/train/
    images/val/
    labels/train/
    labels/val/
    data.yaml
"""

import os
import zipfile
import shutil
import argparse
from pathlib import Path
from urllib.request import urlretrieve

try:
    from PIL import Image
except ImportError:
    Image = None


# WIDER FACE URLs (from Hugging Face dataset repo — reliable mirror)
BASE_URL = "https://huggingface.co/datasets/wider_face/resolve/main/data"
FILES = {
    "WIDER_train.zip": f"{BASE_URL}/WIDER_train.zip",
    "WIDER_val.zip": f"{BASE_URL}/WIDER_val.zip",
    "wider_face_split.zip": f"{BASE_URL}/wider_face_split.zip",
}

# Official split: train / val
ANNOTATION_FILES = {
    "train": "wider_face_train_bbx_gt.txt",
    "val": "wider_face_val_bbx_gt.txt",
}


def download_file(url, dest_path):
    """Download a file with a simple progress indicator."""
    if os.path.exists(dest_path):
        print(f"  Already exists: {dest_path}")
        return
    print(f"  Downloading {url} ...")

    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size / (1024 ** 2)
        total = total_size / (1024 ** 2)
        if total > 0:
            print(f"    {downloaded:.1f} / {total:.1f} MB", end="\r")

    urlretrieve(url, dest_path, reporthook)
    print()


def parse_wider_annotation(ann_path):
    """
    Parse WIDER FACE annotation file.
    Returns dict: {image_rel_path: [(x1, y1, w, h, blur, expression, illumination, occlusion, pose, invalid), ...]}
    """
    samples = {}
    with open(ann_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    i = 0
    while i < len(lines):
        image_rel_path = lines[i]
        i += 1
        num_faces = int(lines[i])
        i += 1
        # WIDER FACE quirk: when num_faces == 0, a dummy "0 0 ..." line still exists
        if num_faces == 0:
            i += 1  # skip the dummy line
        faces = []
        for _ in range(num_faces):
            parts = list(map(int, lines[i].split()))
            i += 1
            # parts: x1, y1, w, h, blur, expression, illumination, occlusion, pose (ignore the rest)
            x1, y1, w, h = parts[0], parts[1], parts[2], parts[3]
            # Also parse face validity: if blur==2 and occlusion==2 and illumination==1 → invalid
            # But we keep all and let the YOLO conversion filter
            faces.append(tuple(parts))
        samples[image_rel_path] = faces
    return samples


def get_image_size(img_path):
    """Get (width, height) of an image using PIL."""
    if Image is None:
        return (1024, 1024)  # fallback
    with Image.open(img_path) as img:
        return img.size  # (width, height)


def convert_to_yolo(samples, split, images_src_dir, images_out_dir, labels_out_dir):
    """Convert parsed annotations to YOLO format, filtering invalid/small faces."""
    os.makedirs(images_out_dir, exist_ok=True)
    os.makedirs(labels_out_dir, exist_ok=True)

    copied = 0
    skipped_no_faces = 0
    skipped_no_file = 0

    for rel_path, faces in samples.items():
        src_path = os.path.join(images_src_dir, rel_path)
        if not os.path.exists(src_path):
            skipped_no_file += 1
            continue

        valid_faces = []
        for f in faces:
            x1, y1, w, h = f[0], f[1], f[2], f[3]
            blur, occlusion = f[4], f[7]
            if w <= 0 or h <= 0:
                continue
            if w * h < 20 * 20:  # skip tiny faces
                continue
            if blur == 2 and occlusion == 2:  # both heavily blurred AND occluded
                continue
            valid_faces.append(f)

        if not valid_faces:
            skipped_no_faces += 1
            continue

        # Copy image
        rel_subdir = os.path.dirname(rel_path)
        dest_subdir = os.path.join(images_out_dir, rel_subdir)
        os.makedirs(dest_subdir, exist_ok=True)
        dest_img = os.path.join(dest_subdir, os.path.basename(rel_path))
        shutil.copy2(src_path, dest_img)

        # Get actual image dimensions for normalization
        img_w, img_h = get_image_size(src_path)

        # Write YOLO label file
        label_name = os.path.splitext(os.path.basename(rel_path))[0] + ".txt"
        label_subdir = os.path.join(labels_out_dir, rel_subdir)
        os.makedirs(label_subdir, exist_ok=True)
        label_path = os.path.join(label_subdir, label_name)

        with open(label_path, "w") as f:
            for face in valid_faces:
                x1, y1, w, h = face[0], face[1], face[2], face[3]
                x_center = (x1 + w / 2) / img_w
                y_center = (y1 + h / 2) / img_h
                w_norm = w / img_w
                h_norm = h / img_h
                f.write(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

        copied += 1

    print(
        f"  [{split}] Copied {copied} images, "
        f"{skipped_no_faces} skipped (no valid faces), "
        f"{skipped_no_file} skipped (file not found)"
    )
    return copied


def create_data_yaml(out_dir):
    """Create data.yaml for YOLO training."""
    train_dir = os.path.join(out_dir, "images/train").replace("\\", "/")
    val_dir = os.path.join(out_dir, "images/val").replace("\\", "/")

    yaml_content = f"""# WIDER FACE dataset for YOLO face detection
path: {os.path.abspath(out_dir).replace("\\\\", "/").replace("\\", "/")}
train: images/train
val: images/val

nc: 1
names: ["face"]
"""
    yaml_path = os.path.join(out_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"  Created {yaml_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare WIDER FACE dataset for YOLO training")
    parser.add_argument("--download-dir", default="./downloads", help="Directory to store downloaded ZIPs")
    parser.add_argument("--output-dir", default="./datasets/widerface", help="Output directory for YOLO-format dataset")
    parser.add_argument("--no-download", action="store_true", help="Skip download (use existing files)")
    args = parser.parse_args()

    download_dir = Path(args.download_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    os.makedirs(download_dir, exist_ok=True)

    # ── Step 1: Download ──────────────────────────────────────────
    if not args.no_download:
        print("=" * 60)
        print("Step 1/4: Downloading WIDER FACE dataset (~1.6 GB total)")
        print("=" * 60)
        for fname, url in FILES.items():
            download_file(url, download_dir / fname)
    else:
        print("Skipping download (--no-download)")

    # ── Step 2: Extract annotations ───────────────────────────────
    print("\n" + "=" * 60)
    print("Step 2/4: Extracting annotations")
    print("=" * 60)
    ann_zip = download_dir / "wider_face_split.zip"
    ann_extract_dir = download_dir / "wider_face_split"
    if not os.path.exists(ann_extract_dir):
        with zipfile.ZipFile(ann_zip, "r") as zf:
            zf.extractall(download_dir)
        print(f"  Extracted to {ann_extract_dir}")
    else:
        print(f"  Already extracted: {ann_extract_dir}")

    # ── Step 3: Parse annotations and convert to YOLO format ─────
    print("\n" + "=" * 60)
    print("Step 3/4: Converting annotations to YOLO format")
    print("=" * 60)

    total_images = 0
    for split_name in ["train", "val"]:
        ann_file = os.path.join(ann_extract_dir, ANNOTATION_FILES[split_name])
        print(f"\nProcessing {split_name} split...")
        print(f"  Annotation: {ann_file}")
        samples = parse_wider_annotation(ann_file)
        print(f"  Parsed {len(samples)} image entries")

        # Source images are in WIDER_<split>/images/
        src_dir = download_dir / f"WIDER_{split_name}" / "images"
        if not os.path.exists(src_dir):
            # Extract the images ZIP if needed
            img_zip = download_dir / f"WIDER_{split_name}.zip"
            if os.path.exists(img_zip):
                print(f"  Extracting {img_zip} ...")
                with zipfile.ZipFile(img_zip, "r") as zf:
                    zf.extractall(download_dir)
                print("  Done.")
            else:
                print(f"  WARNING: {img_zip} not found. Skipping.")
                continue

        images_out = output_dir / "images" / split_name
        labels_out = output_dir / "labels" / split_name
        count = convert_to_yolo(samples, split_name, str(src_dir), str(images_out), str(labels_out))
        total_images += count

    # ── Step 4: Create data.yaml ──────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 4/4: Creating data.yaml")
    print("=" * 60)
    create_data_yaml(str(output_dir))

    print("\n" + "=" * 60)
    print(f"Done! Dataset prepared at: {output_dir}")
    print(f"Total images with valid faces: {total_images}")
    print("=" * 60)
    print("\nTo train YOLO:")
    print(f"  yolo train data={output_dir}/data.yaml model=yolov8n.pt epochs=100")


if __name__ == "__main__":
    main()
