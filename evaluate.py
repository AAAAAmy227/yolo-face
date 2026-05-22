"""Comprehensive evaluation of YOLOv1 face detector.

Computes mAP@.5, mAP@.9, mAP@[.5:.95], and saves PR-curve data at IoU=0.7.

Usage:
    python evaluate.py --data-dir datasets/fddb --weights yolov1_face.pth
"""

import argparse
import json
import math
import os

import torch

from dataset import create_dataloaders
from model import YOLOv1, decode_boxes, nms, collect_pr_data, compute_ap


@torch.no_grad()
def evaluate(model, val_loader, device, iou_thresh, conf_thresh=0.01):
    """Compute AP at a given IoU threshold over the full validation set."""
    model.eval()
    S, B = model.S, model.B
    all_preds, all_gts = [], []

    for images, targets in val_loader:
        images = images.to(device)
        targets_np = targets.cpu()

        preds = model(images).cpu()

        for i in range(images.size(0)):
            gt = targets_np[i]
            gt_boxes = []
            for cy in range(S):
                for cx in range(S):
                    if gt[cy, cx, 0] > 0.5:
                        x_rel, y_rel, w, h = gt[cy, cx, 1:5].tolist()
                        xc = (cx + x_rel) / S
                        yc = (cy + y_rel) / S
                        gt_boxes.append([
                            max(0, xc - w / 2),
                            max(0, yc - h / 2),
                            min(1, xc + w / 2),
                            min(1, yc + h / 2),
                        ])

            pred_boxes = decode_boxes(preds[i], conf_thresh=conf_thresh, S=S, B=B)
            pred_boxes = nms(pred_boxes)

            all_preds.append(pred_boxes)
            all_gts.append(gt_boxes)

    confs, tps, num_gt = collect_pr_data(all_preds, all_gts, iou_thresh=iou_thresh)
    ap, precisions, recalls = compute_ap(confs, tps, num_gt)
    return ap, precisions, recalls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="datasets/fddb")
    parser.add_argument("--weights", default="yolov1_face.pth")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--output", default="eval_results.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model = YOLOv1().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()
    print(f"Loaded weights: {args.weights}")

    # Data
    _, val_loader = create_dataloaders(
        root=args.data_dir, img_size=args.img_size, batch_size=args.batch_size,
    )
    print(f"Validation images: {len(val_loader.dataset)}")

    # ── mAP@[.5:.95] ──
    iou_thresholds = [i / 20.0 + 0.5 for i in range(10)]  # 0.5, 0.55, ..., 0.95
    results = {}
    pr_data_07 = None

    for iou in iou_thresholds:
        ap, precisions, recalls = evaluate(model, val_loader, device, iou)
        results[f"AP@{iou:.2f}"] = round(ap, 4)
        print(f"  AP@{iou:.2f} = {ap:.4f}")

        if abs(iou - 0.7) < 1e-6:
            pr_data_07 = {"precisions": precisions, "recalls": recalls}

    results["mAP@[.5:.95]"] = round(
        sum(results[f"AP@{iou:.2f}"] for iou in iou_thresholds) / len(iou_thresholds), 4
    )

    print(f"\n=== Summary ===")
    print(f"  AP@0.5       = {results['AP@0.50']:.4f}")
    print(f"  AP@0.9       = {results['AP@0.90']:.4f}")
    print(f"  mAP@[.5:.95] = {results['mAP@[.5:.95]']:.4f}")

    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Save PR data for IoU=0.7
    if pr_data_07 is not None:
        pr_path = "pr_data_iou07.json"
        with open(pr_path, "w") as f:
            json.dump(pr_data_07, f)
        print(f"PR curve data (IoU=0.7) saved to {pr_path}")


if __name__ == "__main__":
    main()
