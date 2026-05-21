"""Face detection dataset in YOLO format with YOLOv1 target encoding."""

import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF


class FaceDatasetYOLO(Dataset):
    """Pre-loads images + targets for speed.
    Filters dense images (>max_faces) and tiny/dummy boxes.
    RandomHorizontalFlip on-the-fly."""

    def __init__(
        self,
        root,
        split="train",
        img_size=224,
        S=7,
        B=2,
        C=1,
        max_samples=None,
        seed=42,
        augment=False,
        max_faces=5,
    ):
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.img_size = img_size
        self.S, self.B, self.C = S, B, C
        self.augment = augment

        img_dir = self.root / "images" / split
        label_dir = self.root / "labels" / split
        if not img_dir.is_dir():
            raise FileNotFoundError(
                f"{img_dir} not found. Run prepare_fddb.py first."
            )

        all_images = sorted(img_dir.rglob("*"))
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        images = [p for p in all_images if p.suffix.lower() in exts]

        samples = []
        for img_path in images:
            rel = img_path.relative_to(img_dir)
            label_path = label_dir / rel.with_suffix(".txt")
            if label_path.exists():
                samples.append((str(img_path), str(label_path)))

        if not samples:
            raise RuntimeError(
                f"No valid image-label pairs in {img_dir} / {label_dir}"
            )

        # Deterministic subset
        if max_samples is not None and max_samples < len(samples):
            rng = random.Random(seed)
            samples = rng.sample(samples, max_samples)

        # ── Filter: load boxes, skip dense images, skip tiny/dummy boxes ──
        filtered = []
        skipped_dense = 0
        skipped_empty = 0
        for img_path, label_path in samples:
            boxes = self._load_boxes(label_path)
            if not boxes:
                skipped_empty += 1
                continue
            if len(boxes) > max_faces:
                skipped_dense += 1
                continue
            filtered.append((img_path, boxes))

        if not filtered:
            raise RuntimeError(
                f"All samples filtered out. "
                f"{skipped_empty} empty, {skipped_dense} too dense."
            )

        if split == "train":
            n_before = len(samples)
            n_after = len(filtered)
            if n_before != n_after:
                print(f"  [{split}] filtered: {n_before} → {n_after} "
                      f"(empty={skipped_empty}, dense={skipped_dense})")

        # ── Pre-load image tensors + build targets ──
        self.images = []
        self.targets = []
        for img_path, boxes in filtered:
            img = TF.to_tensor(
                TF.resize(
                    Image.open(img_path).convert("RGB"),
                    (self.img_size, self.img_size),
                )
            )
            target = self._boxes_to_target(boxes)
            self.images.append(img)
            self.targets.append(target)

        # ── Pre-build flipped copies (for per-epoch RandomHorizontalFlip) ──
        self.flip_images = []
        self.flip_targets = []
        if augment:
            for img in self.images:
                self.flip_images.append(TF.hflip(img))
            for target in self.targets:
                self.flip_targets.append(self._flip_target(target))

    def _load_boxes(self, label_path):
        """Parse YOLO label → list of (xc, yc, w, h), filtering tiny/dummy."""
        boxes = []
        try:
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    xc, yc, w, h = map(float, parts[1:5])
                    if w <= 1e-6 or h <= 1e-6:
                        continue
                    if w * h < 0.002:   # skip < ~10×10 px at 224×224
                        continue
                    boxes.append((xc, yc, w, h))
        except Exception:
            pass
        return boxes

    def _boxes_to_target(self, boxes):
        """Convert list of (xc, yc, w, h) → (S, S, B*5+C) target tensor."""
        target = torch.zeros(self.S, self.S, self.B * 5 + self.C)
        for xc, yc, w, h in boxes:
            cx = int(xc * self.S)
            cy = int(yc * self.S)
            if cx >= self.S or cy >= self.S:
                continue
            x_rel = xc * self.S - cx
            y_rel = yc * self.S - cy
            w = min(w, 2.0)
            h = min(h, 2.0)
            for b in range(self.B):
                start = b * 5
                target[cy, cx, start] = 1.0
                target[cy, cx, start + 1] = x_rel
                target[cy, cx, start + 2] = y_rel
                target[cy, cx, start + 3] = w
                target[cy, cx, start + 4] = h
            target[cy, cx, self.B * 5] = 1.0
        return target

    def _flip_target(self, target):
        """Horizontally flip a target tensor."""
        S = self.S
        flipped = torch.zeros_like(target)
        for cy in range(S):
            for cx in range(S):
                if target[cy, cx, 0] > 0.5:
                    x_rel = target[cy, cx, 1].item()
                    xc = (cx + x_rel) / S
                    xc_flip = 1.0 - xc
                    cx_flip = min(int(xc_flip * S), S - 1)
                    x_rel_flip = xc_flip * S - cx_flip
                    for b in range(self.B):
                        start = b * 5
                        flipped[cy, cx_flip, start] = target[cy, cx, start]
                        flipped[cy, cx_flip, start + 1] = x_rel_flip
                        flipped[cy, cx_flip, start + 2] = target[cy, cx, start + 2]
                        flipped[cy, cx_flip, start + 3] = target[cy, cx, start + 3]
                        flipped[cy, cx_flip, start + 4] = target[cy, cx, start + 4]
                    flipped[cy, cx_flip, self.B * 5] = target[cy, cx, self.B * 5]
        return flipped

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.augment and random.random() > 0.5:
            return self.flip_images[idx], self.flip_targets[idx]
        return self.images[idx], self.targets[idx]


def create_dataloaders(
    root,
    img_size=224,
    batch_size=32,
    train_samples=None,
    val_samples=None,
    S=7,
    B=2,
    C=1,
):
    """Build train + val DataLoaders. Training gets filtering + flip augmentation."""
    train_ds = FaceDatasetYOLO(
        root, "train", img_size, S, B, C,
        max_samples=train_samples, augment=True, max_faces=5,
    )
    val_ds = FaceDatasetYOLO(
        root, "val", img_size, S, B, C,
        max_samples=val_samples, augment=False, max_faces=50,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0
    )
    return train_loader, val_loader
