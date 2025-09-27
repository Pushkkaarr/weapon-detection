import torch
from torch.utils.data import DataLoader
from dataset import JsonDetectionDataset
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms as T

# --------------------------
# Collate function
# --------------------------
def collate_fn(batch):
    return tuple(zip(*batch))

# --------------------------
# Load model
# --------------------------
def get_model(num_classes=2):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def load_model(checkpoint_path, device="cuda"):
    model = get_model(num_classes=2)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# --------------------------
# Validation dataset
# --------------------------
val_dataset = JsonDetectionDataset(
    img_dir="data/val/images",
    ann_dir="data/val/labels",
    transforms=T.ToTensor()
)

val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

# --------------------------
# Metrics computation (simple detection rate)
# --------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model("models/detector_epoch3.pth", device)

score_threshold = 0.5
total_images = 0
images_with_detection = 0

with torch.no_grad():
    for images, _ in val_loader:  # ignore targets, measure detection rate only
        images = list(img.to(device) for img in images)
        outputs = model(images)
        
        for pred in outputs:
            scores = pred["scores"].cpu()
            labels = pred["labels"].cpu()
            
            # Count image as detection if any weapon score > threshold
            if any((labels == 1) & (scores >= score_threshold)):
                images_with_detection += 1
            total_images += 1

detection_rate = images_with_detection / total_images if total_images > 0 else 0.0

print(f"✅ Simple Detection Metrics:")
print(f"  Total Images: {total_images}")
print(f"  Images with Weapon Detection: {images_with_detection}")
print(f"  Detection Rate: {detection_rate:.4f} ({detection_rate*100:.2f}%)")
