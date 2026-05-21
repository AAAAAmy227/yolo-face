"""YOLOv1 for face detection (S=7, B=2, C=1)."""

import torch
import torch.nn as nn
F = torch.nn.functional


class YOLOv1(nn.Module):
    def __init__(self, S=7, B=2, C=1):
        super().__init__()
        self.S, self.B, self.C = S, B, C

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1), nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 128, 1), nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128 * 7 * 7, 1024),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(1024, S * S * (B * 5 + C)),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="leaky_relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """(batch,3,224,224) → (batch,S,S,B*5+C) with sigmoid/exp activations applied."""
        x = self.features(x)
        x = self.classifier(x)
        x = x.view(-1, self.S, self.S, self.B * 5 + self.C)

        for b in range(self.B):
            start = b * 5
            x[..., start] = torch.sigmoid(x[..., start])          # conf
            x[..., start + 1] = torch.sigmoid(x[..., start + 1])  # x
            x[..., start + 2] = torch.sigmoid(x[..., start + 2])  # y
            x[..., start + 3] = x[..., start + 3].exp()           # w
            x[..., start + 4] = x[..., start + 4].exp()           # h
        x[..., self.B * 5:] = torch.sigmoid(x[..., self.B * 5:])  # cls
        return x


def yolo_loss(pred, target, S=7, B=2, C=1, lambda_coord=5, lambda_noobj=0.5):
    """YOLOv1 MSE loss."""
    obj_mask = target[..., 0] > 0
    has_obj = obj_mask.any()
    loss = 0.0

    for b in range(B):
        p = pred[..., b * 5 : b * 5 + 5]
        t = target[..., b * 5 : b * 5 + 5]
        if has_obj:
            loss += lambda_coord * F.mse_loss(p[obj_mask, 1:3], t[obj_mask, 1:3], reduction="sum")
            loss += lambda_coord * F.mse_loss(
                torch.sqrt(p[obj_mask, 3:5] + 1e-6),
                torch.sqrt(t[obj_mask, 3:5] + 1e-6), reduction="sum")
            loss += F.mse_loss(p[obj_mask, 0], t[obj_mask, 0], reduction="sum")
        loss += lambda_noobj * F.mse_loss(p[~obj_mask, 0], t[~obj_mask, 0], reduction="sum")

    if has_obj:
        loss += F.mse_loss(pred[obj_mask, B * 5:], target[obj_mask, B * 5:], reduction="sum")

    return loss / pred.size(0)


def decode_boxes(pred, conf_thresh=0.3, S=7, B=2):
    """YOLO output → list of [x1,y1,x2,y2,score] in [0,1] coords."""
    boxes = []
    for cy in range(S):
        for cx in range(S):
            for b in range(B):
                start = b * 5
                conf = pred[cy, cx, start].item()
                if conf < conf_thresh:
                    continue
                x_rel = pred[cy, cx, start + 1].item()
                y_rel = pred[cy, cx, start + 2].item()
                w, h = pred[cy, cx, start + 3].item(), pred[cy, cx, start + 4].item()
                xc = (cx + x_rel) / S
                yc = (cy + y_rel) / S
                x1 = max(0, xc - w / 2)
                y1 = max(0, yc - h / 2)
                x2 = min(1, xc + w / 2)
                y2 = min(1, yc + h / 2)
                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2, conf])
    return boxes


def iou(b1, b2):
    """IoU of two boxes [x1,y1,x2,y2]."""
    inter_x1 = max(b1[0], b2[0])
    inter_y1 = max(b1[1], b2[1])
    inter_x2 = min(b1[2], b2[2])
    inter_y2 = min(b1[3], b2[3])
    inter = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (area1 + area2 - inter + 1e-8)


def nms(boxes, iou_thresh=0.5):
    """Non-maximum suppression."""
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    keep = []
    while boxes:
        best = boxes.pop(0)
        keep.append(best)
        boxes = [b for b in boxes if iou(best, b) < iou_thresh]
    return keep


def average_precision(pred_boxes, gt_boxes, iou_thresh=0.5):
    """Precision at given IoU threshold for a single image."""
    if not gt_boxes:
        return float("nan")
    if not pred_boxes:
        return 0.0
    pred_boxes = sorted(pred_boxes, key=lambda x: x[4], reverse=True)
    tp = fp = 0
    gt_matched = [False] * len(gt_boxes)
    for pb in pred_boxes:
        best_iou = best_idx = 0
        for j, gb in enumerate(gt_boxes):
            if not gt_matched[j]:
                d = iou(pb, gb)
                if d > best_iou:
                    best_iou = d
                    best_idx = j
        if best_iou >= iou_thresh and best_idx >= 0:
            tp += 1
            gt_matched[best_idx] = True
        else:
            fp += 1
    return tp / (tp + fp) if tp else 0.0
