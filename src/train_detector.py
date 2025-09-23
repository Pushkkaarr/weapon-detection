import os
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from src.dataset import JsonDetectionDataset
import torchvision.transforms as T
import argparse

def collate_fn(batch):
    return tuple(zip(*batch))

def get_model(num_classes=2):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def train_detector(data_dir, epochs=2, batch_size=2, lr=0.005, device='cpu'):
    # Updated paths pointing to classifier folders
    train_img_dir = os.path.join(data_dir, "classifier/train")
    val_img_dir   = os.path.join(data_dir, "classifier/val")

    # Use transforms to convert images to tensors
    transform = T.Compose([T.ToTensor()])

    # Datasets
    train_dataset = JsonDetectionDataset(train_img_dir, ann_dir=None, transforms=transform)
    val_dataset   = JsonDetectionDataset(val_img_dir, ann_dir=None, transforms=transform)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # Model
    model = get_model(num_classes=2)
    model.to(device)

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for images, targets in train_loader:
            # Convert PIL Images to tensors and move to device
            images = [img.to(device) for img in images]

            # If targets exist, move to device; else use empty targets (demo-level)
            if targets[0] is not None:
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            else:
                targets = [{} for _ in images]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            total_loss += losses.item()

        print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss:.4f}")

        # Save checkpoint
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), f"models/detector_epoch{epoch+1}.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--device", type=str, default="cpu")  # Change to 'cuda' if GPU available
    args = parser.parse_args()

    train_detector(args.data_dir, args.epochs, args.batch_size, args.lr, args.device)
