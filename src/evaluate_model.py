import torch
from torch.utils.data import DataLoader
from dataset import JsonDetectionDataset
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms as T
from torchvision.ops import box_iou

# --------------------------
# Collate function
# --------------------------
def collate_fn(batch):
    return tuple(zip(*batch))

# --------------------------
# Model
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
# Dataset
# --------------------------
val_dataset = JsonDetectionDataset(
    img_dir="data/val/images",
    ann_dir="data/val/labels",
    transforms=T.ToTensor()
)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

# --------------------------
# Fast accuracy computation
# --------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model("models/detector_epoch3.pth", device)

iou_threshold = 0.5
correct = 0
total = 0

with torch.no_grad():
    for images, targets in val_loader:
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(images)

        for pred, target in zip(outputs, targets):
            pred_boxes = pred['boxes']
            pred_labels = pred['labels']
            gt_boxes = target['boxes']
            gt_labels = target['labels']

            for gt_box, gt_label in zip(gt_boxes, gt_labels):
                total += 1
                if len(pred_boxes) == 0:
                    continue
                ious = box_iou(gt_box.unsqueeze(0), pred_boxes)  # [1, num_pred]
                # Check if any predicted box matches IoU & class
                if any((ious[0] >= iou_threshold) & (pred_labels == gt_label)):
                    correct += 1

accuracy = correct / total
print(f"Validation Accuracy: {accuracy*100:.2f}%")
