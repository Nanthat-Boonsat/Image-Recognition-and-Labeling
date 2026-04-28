"""Run object detection inference on a folder of images.

This script loads a trained checkpoint, runs prediction on each image in an
input directory, and saves visualized detections to an output directory.
"""

import argparse
import csv
import json
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import torch
from torchvision.transforms.functional import to_tensor

from model import build_model


VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on a folder of images")
    base_dir = Path(__file__).resolve().parent

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(base_dir / "output" / "best_model.pth"),
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Folder containing test images",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(base_dir / "output" / "inference"),
        help="Folder to save predicted images",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="fasterrcnn_resnet50_fpn",
        choices=["fasterrcnn_resnet50_fpn", "fasterrcnn_mobilenet_v3"],
        help="Backbone used during training",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=2,
        help="Number of classes including background",
    )
    parser.add_argument(
        "--score_thresh",
        type=float,
        default=0.5,
        help="Minimum score threshold for drawing detections",
    )
    parser.add_argument(
        "--max_detections",
        type=int,
        default=50,
        help="Maximum detections to keep per image",
    )
    parser.add_argument(
        "--label_map_json",
        type=str,
        default="",
        help='Optional JSON mapping from class id to class name, e.g. {"1": "mug"}',
    )

    return parser.parse_args()


def load_state_dict(checkpoint_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            cleaned[key[len("module.") :]] = value
        else:
            cleaned[key] = value
    return cleaned


def load_label_map(path: str):
    if not path:
        return {}

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    label_map = {}
    for key, value in raw.items():
        label_map[int(key)] = str(value)
    return label_map


def find_images(input_dir: Path):
    files = [p for p in sorted(input_dir.iterdir()) if p.suffix.lower() in VALID_IMAGE_EXTS]
    return files


def draw_predictions(image: Image.Image, pred: dict, label_map: dict, score_thresh: float, max_detections: int):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    boxes = pred["boxes"].detach().cpu()
    labels = pred["labels"].detach().cpu()
    scores = pred["scores"].detach().cpu()

    kept = 0
    for box, label, score in zip(boxes, labels, scores):
        score_val = float(score.item())
        if score_val < score_thresh:
            continue
        if kept >= max_detections:
            break

        x1, y1, x2, y2 = [float(v) for v in box.tolist()]
        class_id = int(label.item())
        class_name = label_map.get(class_id, f"class_{class_id}")
        caption = f"{class_name}: {score_val:.2f}"

        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1 + 2, max(0, y1 - 14)), caption, fill="yellow", font=font)
        kept += 1

    return kept


def main():
    args = parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()

    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = find_images(input_dir)
    if not image_paths:
        raise RuntimeError(f"No supported image files found in {input_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.backbone, num_classes=args.num_classes)
    model.load_state_dict(load_state_dict(checkpoint_path))
    model.to(device)
    model.eval()

    label_map = load_label_map(args.label_map_json)

    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Images found: {len(image_paths)}")
    print(f"Saving predictions to: {output_dir}")

    rows = []
    with torch.no_grad():
        for img_path in image_paths:
            image = Image.open(img_path).convert("RGB")
            image_tensor = to_tensor(image).to(device)

            prediction = model([image_tensor])[0]
            kept = draw_predictions(
                image,
                prediction,
                label_map=label_map,
                score_thresh=args.score_thresh,
                max_detections=args.max_detections,
            )

            out_path = output_dir / img_path.name
            image.save(out_path)

            rows.append([img_path.name, kept])
            print(f"{img_path.name}: kept {kept} detections")

    csv_path = output_dir / "predictions_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "detections_kept"])
        writer.writerows(rows)

    print(f"Saved summary CSV: {csv_path}")
    print("Inference complete.")


if __name__ == "__main__":
    main()
