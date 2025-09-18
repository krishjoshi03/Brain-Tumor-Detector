import torch
from torchvision import transforms, models
from PIL import Image

# Classes
classes = ['glioma', 'meningioma', 'notumor', 'pituitary']

device = torch.device("cpu")

# Model definition
class TumorClassifierRES(torch.nn.Module):
    def __init__(self, num_classes, dropout_rate=0.5):
        super().__init__()
        self.model = models.resnet50(weights="DEFAULT")
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.layer4.parameters():
            param.requires_grad = True
        self.model.fc = torch.nn.Sequential(
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(self.model.fc.in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# Load model
model = TumorClassifierRES(num_classes=len(classes))
model.load_state_dict(torch.load("Tumor_Prediction.pth", map_location=device))
model.eval()

# Transforms
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

def predict_image(image):
    """Take a PIL image and return prediction + confidence"""
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        prob = torch.nn.functional.softmax(outputs, dim=1)
        confidence, pred_idx = torch.max(prob, 1)
    return classes[pred_idx.item()], confidence.item()
