import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torchvision.transforms as T

class Detector:
    def __init__(self, device='cuda'):
        self.device = device
        self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        # Replace the head: num_classes = 2 (background + weapon)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
        self.model.to(self.device)
        self.model.eval()

        # Transform for input images
        self.transform = T.Compose([
            T.ToTensor(),
        ])

    def load(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, image):
        """
        image: PIL Image
        returns: list of predictions (boxes, labels, scores)
        """
        img_tensor = self.transform(image).to(self.device)
        preds = self.model([img_tensor])[0]
        return preds
