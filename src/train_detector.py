import os
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms as T
from tqdm import tqdm
import argparse
from dataset import JsonDetectionDataset  # assuming dataset.py is in same folder

# Collate function
def collate_fn(batch):
    return tuple(zip(*batch))

# Model with updated head
def get_model(num_classes=2):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# Training
def train_detector(data_dir, epochs=2, batch_size=2, lr=0.005, device="cuda", resume_checkpoint=None):
    train_img_dir = os.path.join(data_dir, "train/images")
    train_ann_dir = os.path.join(data_dir, "train/labels")
    val_img_dir   = os.path.join(data_dir, "val/images")
    val_ann_dir   = os.path.join(data_dir, "val/labels")

    # Transform
    transform = T.ToTensor()

    # Datasets
    train_dataset = JsonDetectionDataset(train_img_dir, train_ann_dir, transforms=transform)
    val_dataset   = JsonDetectionDataset(val_img_dir, val_ann_dir, transforms=transform)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # Model
    model = get_model(num_classes=2)
    model.to(device)

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)

    # Resume from checkpoint if provided
    start_epoch = 0
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"Loading checkpoint: {resume_checkpoint}")
        model.load_state_dict(torch.load(resume_checkpoint, map_location=device))
        # Extract epoch number from filename if saved like detector_epochX.pth
        try:
            start_epoch = int(os.path.basename(resume_checkpoint).split("epoch")[1].split(".")[0])
        except:
            start_epoch = 0
        print(f"Resuming from epoch {start_epoch}")

    total_batches = len(train_loader)

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")
        for batch_idx, (images, targets) in enumerate(pbar):
            # Move to device
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward + loss
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            total_loss += losses.item()
            pbar.set_postfix(loss=losses.item())

            # Save checkpoint every 10% of epoch
            if (batch_idx + 1) % max(1, total_batches // 10) == 0:
                os.makedirs("models", exist_ok=True)
                checkpoint_path = f"models/checkpoint_epoch{epoch+1}_batch{batch_idx+1}.pth"
                torch.save(model.state_dict(), checkpoint_path)
                print(f"\nCheckpoint saved at {checkpoint_path}")

        print(f"Epoch [{epoch+1}/{epochs}] Average Loss: {total_loss/len(train_loader):.4f}")

        # Save final checkpoint at end of epoch
        os.makedirs("models", exist_ok=True)
        final_checkpoint = f"models/detector_epoch{epoch+1}.pth"
        torch.save(model.state_dict(), final_checkpoint)
        print(f"Epoch {epoch+1} finished, saved model to {final_checkpoint}")

# Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--device", type=str, default="cpu")  # change to "cuda" if GPU available
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    # Detect GPU automatically if available
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU instead.")
        args.device = "cpu"

    train_detector(args.data_dir, args.epochs, args.batch_size, args.lr, args.device, resume_checkpoint=args.resume)
