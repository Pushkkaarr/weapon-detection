import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

# Mapping classes to integers
CLASS_MAP = {"no_weapon": 0, "weapon": 1}

class JsonDetectionDataset(Dataset):
    def __init__(self, img_dir, ann_dir, transforms=None):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.transforms = transforms

        # List all JSON annotation files
        self.files = [f for f in os.listdir(ann_dir) if f.endswith(".json")]
        if len(self.files) == 0:
            raise RuntimeError(f"No JSON files found in {ann_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        ann_file = self.files[idx]
        ann_path = os.path.join(self.ann_dir, ann_file)

        # Load JSON annotation safely
        try:
            with open(ann_path) as f:
                ann = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to read annotation {ann_file}: {e}")

        # Image filename (remove only '.json')
        img_name = ann_file[:-5]  # strips '.json'
        img_path = os.path.join(self.img_dir, img_name)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found for annotation: {ann_file}")

        img = Image.open(img_path).convert("RGB")
        width, height = img.size

        boxes, labels = [], []

        for obj in ann.get("objects", []):
            cls = obj.get("classTitle")
            if cls == "weapon":
                label = CLASS_MAP["weapon"]
            else:
                label = CLASS_MAP["no_weapon"]

            # Get bounding box, skip if invalid
            exterior = obj.get("points", {}).get("exterior", [])
            if len(exterior) != 2:
                continue

            (x_min, y_min), (x_max, y_max) = exterior

            # Skip boxes with zero width or height
            if x_max <= x_min or y_max <= y_min:
                continue

            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(label)

        # If no valid boxes, assign dummy box covering whole image
        if len(boxes) == 0:
            boxes = [[0, 0, width, height]]
            labels = [CLASS_MAP["no_weapon"]]

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels}

        if self.transforms:
            img = self.transforms(img)

        return img, target


# Quick test
if __name__ == "__main__":
    dataset = JsonDetectionDataset(
        img_dir="data/train/images",
        ann_dir="data/train/labels",
        transforms=T.ToTensor()
    )
    img, target = dataset[0]
    print("Image shape:", img.shape)
    print("Target:", target)
