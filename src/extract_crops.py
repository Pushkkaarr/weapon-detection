import os
import json
from PIL import Image

# Paths
img_dir = "data/train/images"
ann_dir = "data/train/labels"
crop_dir = "data/classifier/train/weapon"

os.makedirs(crop_dir, exist_ok=True)

# Supported image extensions
IMG_EXTS = [".jpg", ".jpeg", ".png"]

for ann_file in os.listdir(ann_dir):
    if not ann_file.endswith(".json"):
        continue

    # Load annotation
    ann_path = os.path.join(ann_dir, ann_file)
    with open(ann_path, "r") as f:
        ann = json.load(f)

    # base_name already includes .jpg from JSON filename
    base_name = os.path.splitext(ann_file)[0]  # e.g., '000b9a97776b3634.jpg'
    img_path = os.path.join(img_dir, base_name)

    # Check if image exists
    if not os.path.exists(img_path):
        print(f"[WARNING] Image not found for annotation: {ann_file}")
        continue

    # Open image
    img = Image.open(img_path).convert("RGB")

    # Crop weapon objects
    for i, obj in enumerate(ann.get("objects", [])):
        cls = obj.get("classTitle")
        print(f"Cropping {base_name} object {i} class={cls}")  # debug print

        if cls != "weapon":
            continue

        exterior = obj.get("points", {}).get("exterior", [])
        if len(exterior) != 2:
            print(f"[WARNING] Invalid bbox in {ann_file}")
            continue

        (x_min, y_min), (x_max, y_max) = exterior
        # Ensure coordinates are integers
        x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])

        # Clip coordinates to image boundaries
        x_min = max(0, min(x_min, img.width - 1))
        x_max = max(0, min(x_max, img.width))
        y_min = max(0, min(y_min, img.height - 1))
        y_max = max(0, min(y_max, img.height))

        # Skip if width or height is zero or negative
        if x_max <= x_min or y_max <= y_min:
            print(f"[WARNING] Skipping invalid bbox in {ann_file}: {exterior}")
            continue

        # Crop and save
        crop = img.crop((x_min, y_min, x_max, y_max))
        crop_filename = f"{base_name}_{i}.jpg"
        crop.save(os.path.join(crop_dir, crop_filename))

print("Cropping completed successfully!")
