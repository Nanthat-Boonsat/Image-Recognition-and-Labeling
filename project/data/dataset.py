import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
import os
from utils1 import show_batch



class ObjDetectionDataset(torch.utils.data.Dataset):
    """Dataset that reads image paths and YOLO label files from a DataFrame."""

    def __init__(self, df, base_dir=None):
        self.df = df.reset_index(drop=True)
        self.base_dir = base_dir or os.getcwd()

    def _resolve_path(self, path_value):
        """Resolve absolute or project-relative file paths from CSV values."""
        path_value = str(path_value)
        if os.path.isabs(path_value):
            return path_value

        candidates = [
            path_value,
            os.path.join(self.base_dir, path_value),
        ]

        if path_value.startswith("data/"):
            trimmed = path_value[len("data/"):]
            candidates.append(os.path.join(self.base_dir, trimmed))

        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate

        return os.path.join(self.base_dir, path_value)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """Return one training sample as (image_tensor, target_dict)."""
        row = self.df.iloc[idx]

        img_path = self._resolve_path(row["images files"])
        label_path = self._resolve_path(row["labels files"])

        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        image = to_tensor(img)

        boxes, labels = [], []
        with open(label_path) as f:
            for line in f:
                cls, xc, yc, bw, bh = map(float, line.split())
                x1 = (xc - bw/2) * w
                y1 = (yc - bh/2) * h
                x2 = (xc + bw/2) * w
                y2 = (yc + bh/2) * h
                boxes.append([x1, y1, x2, y2])
                labels.append(int(cls) + 1)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx]),
        }
        return image, target


def preview_train_batch(train_loader):
    images, targets = next(iter(train_loader))
    show_batch(images, targets)