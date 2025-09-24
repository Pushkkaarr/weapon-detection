import os
import cv2
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T

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
# Extract frames with weapons
# --------------------------
def extract_weapon_frames(video_path, output_dir, model, device="cuda", score_threshold=0.5):
    os.makedirs(output_dir, exist_ok=True)

    transform = T.ToTensor()
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise FileNotFoundError(f"❌ Video file not found: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_num = 0

    print(f"Video info → FPS: {fps}, Total Frames: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = transform(img).to(device)

        with torch.no_grad():
            outputs = model([img_tensor])[0]

        boxes = outputs["boxes"]
        labels = outputs["labels"]
        scores = outputs["scores"]

        weapon_detected = False
        for box, label, score in zip(boxes, labels, scores):
            if score >= score_threshold and label.item() == 1:  # weapon
                weapon_detected = True
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"weapon {score:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if weapon_detected:
            timestamp_sec = frame_num / fps
            output_path = os.path.join(output_dir, f"weapon_frame_{frame_num}_at_{timestamp_sec:.2f}s.jpg")
            cv2.imwrite(output_path, frame)

        frame_num += 1
        if frame_num % 50 == 0:
            print(f"Processed {frame_num} frames...")

    cap.release()
    print(f"✅ Frames with weapons saved in '{output_dir}'")


# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model("models/detector_epoch3.pth", device)

    extract_weapon_frames(
        video_path="E:\Softwares\VScode\GitHub\AISOLO_DL_Task\weapon-detection\Extraction-One-Shot-Gun-Figh-Scene.mp4",
        output_dir="weapon_frames",
        model=model,
        device=device,
        score_threshold=0.5
    )
