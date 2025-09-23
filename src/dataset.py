import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

CLASS_MAP = {"weapon": 1}  # 'no_weapon' can be 0 if needed

class JsonDetectionDataset(Dataset):
    def __init__(self, img_dir, ann_dir, transforms=None):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.transforms = transforms

        # list all json files
        self.files = [f for f in os.listdir(ann_dir) if f.endswith(".json")]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        ann_file = self.files[idx]
        ann_path = os.path.join(self.ann_dir, ann_file)
        with open(ann_path) as f:
            ann = json.load(f)

        # Remove only the '.json' to get correct image filename
        img_name = ann_file[:-5]  # strips '.json'
        img_path = os.path.join(self.img_dir, img_name)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found for annotation: {ann_file}")

        img = Image.open(img_path).convert("RGB")

        # Extract boxes and labels
        boxes = []
        labels = []
        for obj in ann.get("objects", []):
            cls = obj.get("classTitle")
            if cls not in CLASS_MAP:
                continue

            exterior = obj.get("points", {}).get("exterior", [])
            if len(exterior) != 2:
                continue  # skip invalid bboxes

            (x_min, y_min), (x_max, y_max) = exterior
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(CLASS_MAP[cls])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels}

        if self.transforms:
            img = self.transforms(img)

        return img, target


# quick test
if __name__ == "__main__":
    dataset = JsonDetectionDataset(
        img_dir="data/train/images",
        ann_dir="data/train/labels",
        transforms=T.ToTensor()
    )
    img, target = dataset[0]
    print("Image shape:", img.shape)
    print("Target:", target)
