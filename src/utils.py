import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np

def augment_and_save(image_path, save_dir, n_aug=5):
    img = np.array(Image.open(image_path).convert("RGB"))

    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5)
    ])

    os.makedirs(save_dir, exist_ok=True)

    for i in range(n_aug):
        augmented = transform(image=img)
        aug_img = Image.fromarray(augmented["image"])
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        aug_img.save(os.path.join(save_dir, f"{base_name}_aug{i}.jpg"))
