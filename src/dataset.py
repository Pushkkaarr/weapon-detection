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
        with open(os.path.join(self.ann_dir, ann_file)) as f:
            ann = json.load(f)

        # Load image
        img_name = os.path.splitext(ann_file)[0] + ".jpg"  # adjust if png
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert("RGB")

        # Extract boxes and labels
        boxes = []
        labels = []
        for obj in ann["objects"]:
            cls = obj["classTitle"]
            if cls not in CLASS_MAP:
                continue
            (x_min, y_min), (x_max, y_max) = obj["points"]["exterior"]
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
