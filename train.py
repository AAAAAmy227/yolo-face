"""Train YOLOv1 on FDDB face dataset.

Usage:
    # First prepare FDDB:
    python prepare_fddb.py
    # Then train:
    python train.py --data-dir datasets/fddb
"""

import argparse
import math
import time

import torch

try:
    from torch.utils.tensorboard import SummaryWriter
    _has_tb = True
except ImportError:
    _has_tb = False

from dataset import create_dataloaders
from model import YOLOv1, yolo_loss, decode_boxes, nms, average_precision


@torch.no_grad()
def validate(model, val_loader, device):
    """Mean validation loss and AP@0.5."""
    model.eval()
    total_loss = 0.0
    aps = []
    S, B = model.S, model.B

    for images, targets in val_loader:
        images = images.to(device)
        targets = targets.to(device)

        preds = model(images)
        loss = yolo_loss(preds, targets)
        total_loss += loss.item() * images.size(0)

        for i in range(images.size(0)):
            gt = targets[i].cpu()
            gt_boxes = []
            for cy in range(S):
                for cx in range(S):
                    if gt[cy, cx, 0] > 0.5:
                        x_rel, y_rel, w, h = gt[cy, cx, 1:5].tolist()
                        xc = (cx + x_rel) / S
                        yc = (cy + y_rel) / S
                        gt_boxes.append([
                            max(0, xc - w/2), max(0, yc - h/2),
                            min(1, xc + w/2), min(1, yc + h/2),
                        ])

            pred_boxes = decode_boxes(preds[i].cpu(), conf_thresh=0.3, S=S, B=B)
            pred_boxes = nms(pred_boxes)
            ap = average_precision(pred_boxes, gt_boxes, iou_thresh=0.5)
            if not math.isnan(ap):
                aps.append(ap)

    avg_loss = total_loss / len(val_loader.dataset)
    avg_ap = sum(aps) / len(aps) if aps else 0.0
    return avg_loss, avg_ap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="datasets/fddb")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-samples", type=int, default=None,
                        help="Limit training samples (default: all)")
    parser.add_argument("--val-samples", type=int, default=None,
                        help="Limit validation samples (default: all)")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--checkpoint", default="yolov1_face.pth")
    parser.add_argument("--log-dir", default="logs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    print("Loading data ...")
    train_loader, val_loader = create_dataloaders(
        root=args.data_dir, img_size=args.img_size, batch_size=args.batch_size,
        train_samples=args.train_samples, val_samples=args.val_samples,
    )
    print(f"  Train: {len(train_loader.dataset)} images")
    print(f"  Val:   {len(val_loader.dataset)} images")

    model = YOLOv1().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 50, 100], gamma=0.5)

    writer = SummaryWriter(args.log_dir) if _has_tb else None

    best_val_ap = 0.0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0

        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            # GPU batch color jitter (per-epoch random, ~0.02ms overhead at B=32)
            b = images.size(0)
            bf = 1.0 + (torch.rand(b, 1, 1, 1, device=device) - 0.5) * 0.3
            cf = 1.0 + (torch.rand(b, 1, 1, 1, device=device) - 0.5) * 0.3
            sf = torch.rand(b, 1, 1, 1, device=device) * 0.6 + 0.7
            images = images * bf
            im_mean = images.mean(dim=(2, 3), keepdim=True)
            images = (images - im_mean) * cf + im_mean
            gray = images.mean(dim=1, keepdim=True)
            images = gray + (images - gray) * sf
            images = images.clamp(0, 1)
            preds = model(images)
            loss = yolo_loss(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * images.size(0)

        scheduler.step()
        avg_train_loss = epoch_loss / len(train_loader.dataset)

        if epoch % 20 == 0 or epoch == 1:
            val_loss, val_ap = validate(model, val_loader, device)
            print(
                f"Epoch {epoch:3d}/{args.epochs}  "
                f"train_loss={avg_train_loss:.4f}  "
                f"val_loss={val_loss:.4f}  "
                f"val_AP@0.5={val_ap:.3f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )
            if writer:
                writer.add_scalar("Loss/train", avg_train_loss, epoch)
                writer.add_scalar("Loss/val", val_loss, epoch)
                writer.add_scalar("AP@0.5/val", val_ap, epoch)
            if val_ap > best_val_ap:
                best_val_ap = val_ap
                torch.save(model.state_dict(), args.checkpoint)
                print(f"  Best model saved (AP={val_ap:.3f})")
        else:
            print(f"Epoch {epoch:3d}/{args.epochs}  train_loss={avg_train_loss:.4f}")

        if writer:
            writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed / 60:.1f} min. Best val AP: {best_val_ap:.3f}")
    print(f"Final weights: {args.checkpoint}")
    if writer:
        writer.close()


if __name__ == "__main__":
    main()
