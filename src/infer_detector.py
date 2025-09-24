import os
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T

# Load model
def get_model(num_classes=2):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)  # no pretraining here
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def load_model(checkpoint_path, device="cuda"):
    model = get_model(num_classes=2)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Run inference
def run_inference(model, image_path, device="cuda", score_threshold=0.5):
    transform = T.ToTensor()
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).to(device)

    with torch.no_grad():
        outputs = model([img_tensor])

    outputs = outputs[0]
    boxes = outputs["boxes"].cpu()
    labels = outputs["labels"].cpu()
    scores = outputs["scores"].cpu()

    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    for box, label, score in zip(boxes, labels, scores):
        if score >= score_threshold:
            x1, y1, x2, y2 = box
            label_name = "weapon" if label.item() == 1 else "no_weapon"
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1, y1), f"{label_name} {score:.2f}", fill="red", font=font)

    return img

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model("models/detector_epoch3.pth", device)

    test_image = "/21f3e14b55749e4e1aeb45e2a560777d-104484294.jpg"  # replace with your test image path
    result = run_inference(model, test_image, device)

    result.save("result.jpg")
    print("Saved result as result.jpg")
