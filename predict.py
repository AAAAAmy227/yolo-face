"""Run YOLOv1 detection on an image.

Usage:
    python predict.py --weights yolov1_face.pth --image path/to/image.jpg
    python predict.py --weights yolov1_face.pth --image path/to/image.jpg --save output.jpg
"""

import argparse

import torch
from PIL import Image, ImageDraw
from torchvision.transforms import functional as TF

from model import YOLOv1, decode_boxes, nms


@torch.no_grad()
def predict(model, image_path, img_size=224, conf_thresh=0.3, iou_thresh=0.5):
    """Run inference → (resized_image, list of [x1,y1,x2,y2,score])."""
    img = Image.open(image_path).convert("RGB")
    img_resized = img.resize((img_size, img_size))
    img_tensor = TF.to_tensor(img_resized).unsqueeze(0)

    device = next(model.parameters()).device
    img_tensor = img_tensor.to(device)

    output = model(img_tensor).squeeze(0).cpu()

    pred_boxes = decode_boxes(output, conf_thresh=conf_thresh, S=model.S, B=model.B)
    pred_boxes = nms(pred_boxes, iou_thresh=iou_thresh)

    scale = img_size
    boxes_scaled = [
        [int(x1 * scale), int(y1 * scale), int(x2 * scale), int(y2 * scale), score]
        for x1, y1, x2, y2, score in pred_boxes
    ]
    return img_resized, boxes_scaled


def draw_boxes(img, boxes):
    """Draw bounding boxes on a PIL Image."""
    draw = ImageDraw.Draw(img)
    for x1, y1, x2, y2, score in boxes:
        draw.rectangle([x1, y1, x2, y2], outline="lime", width=2)
        draw.text((x1, max(0, y1 - 12)), f"{score:.2f}", fill="lime")
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="Path to .pth weights")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--save", default=None, help="Path to save output image")
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLOv1().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()
    print(f"Loaded weights: {args.weights}")

    img, boxes = predict(
        model, args.image, conf_thresh=args.conf, iou_thresh=args.iou
    )
    print(f"Detected {len(boxes)} faces:")
    for x1, y1, x2, y2, score in boxes:
        print(f"  ({x1},{y1}) → ({x2},{y2})  conf={score:.3f}")

    result = draw_boxes(img.copy(), boxes)
    result.show()

    if args.save:
        result.save(args.save)
        print(f"Saved to {args.save}")


if __name__ == "__main__":
    main()
