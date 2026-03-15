import csv
import os
import random


images_dir = "/home/nongbhor/pytorch-object-detection/project/data/images"
labels_dir = "/home/nongbhor/pytorch-object-detection/project/data/labels"
output_csv_dir = "/home/nongbhor/pytorch-object-detection/project/data/CSVs"

os.makedirs(output_csv_dir, exist_ok=True)


def collect_pairs() -> list[list[str]]:
    """Collect [image_path, label_path] rows where both files exist."""
    rows: list[list[str]] = []
    image_exts = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")

    for image_name in sorted(os.listdir(images_dir)):
        _, ext = os.path.splitext(image_name)
        if ext not in image_exts:
            continue

        stem, _ = os.path.splitext(image_name)
        label_name = f"{stem}.txt"
        label_abs = os.path.join(labels_dir, label_name)
        if not os.path.exists(label_abs):
            continue

        image_rel = os.path.join("data", "images", image_name)
        label_rel = os.path.join("data", "labels", label_name)
        rows.append([image_rel, label_rel])

    return rows


def write_csv(path: str, rows: list[list[str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["images files", "labels files"])
        writer.writerows(rows)


pairs = collect_pairs()
if not pairs:
    raise ValueError("No image-label pairs found. Check filenames and paths in data/images and data/labels.")

random.seed(42)
random.shuffle(pairs)

split_idx = int(0.8 * len(pairs))
train_rows = pairs[:split_idx]
val_rows = pairs[split_idx:]

train_csv = os.path.join(output_csv_dir, "train_df.csv")
val_csv = os.path.join(output_csv_dir, "val_df.csv")

write_csv(train_csv, train_rows)
write_csv(val_csv, val_rows)

print(f"Train CSV: {train_csv}, {len(train_rows)} rows")
print(f"Val CSV: {val_csv}, {len(val_rows)} rows")