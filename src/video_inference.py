import os
import cv2
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T

# ---------------------------
# Load model
# ---------------------------
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


# ---------------------------
# Run inference on video
# ---------------------------
def process_video(video_path, output_path, model, device="cuda", score_threshold=0.5):
    # Check absolute path + file existence
    abs_path = os.path.abspath(video_path)
    print(f"Looking for video at: {abs_path}")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"❌ Video file not found: {abs_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"❌ OpenCV failed to open video: {abs_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"✅ Video opened successfully → FPS: {fps}, Width: {width}, Height: {height}")

    # Use 'XVID' codec for better compatibility
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR → RGB
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = T.ToTensor()(img).to(device)

        with torch.no_grad():
            outputs = model([img_tensor])

        outputs = outputs[0]
        boxes = outputs["boxes"].cpu()
        labels = outputs["labels"].cpu()
        scores = outputs["scores"].cpu()

        for box, label, score in zip(boxes, labels, scores):
            if score >= score_threshold:
                x1, y1, x2, y2 = map(int, box)
                color = (0, 0, 255) if label.item() == 1 else (0, 255, 0)  # red for weapon
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"{'weapon' if label.item()==1 else 'no_weapon'} {score:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

        out.write(frame)
        frame_num += 1
        if frame_num % 50 == 0:
            print(f"Processed {frame_num} frames...")

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"🎥 Video processing complete → saved at {os.path.abspath(output_path)}")


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model("models/detector_epoch3.pth", device)

    process_video("E:\Softwares\VScode\GitHub\AISOLO_DL_Task\weapon-detection\Extraction-One-Shot-Gun-Figh-Scene.mp4", "output.avi", model, device)
