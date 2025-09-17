import torch
import torch.nn as nn
from torchvision import models, transforms
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io

# Define the same architecture you used in training
class TumorClassifierRES(nn.Module):
    def __init__(self, num_classes=4, dropout_rate=0.5):
        super().__init__()
        self.model = models.resnet50(weights="DEFAULT")

        # Freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze layer4
        for param in self.model.layer4.parameters():
            param.requires_grad = True

        # Custom classifier head
        self.model.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.model.fc.in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TumorClassifierRES(num_classes=4).to(device)
model.load_state_dict(torch.load("Tumor_Prediction.pth", map_location=device))
model.eval()

# Define transforms (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Class labels
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']

# FastAPI app
app = FastAPI()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    return {
        "class": CLASSES[predicted.item()],
        "confidence": confidence.item()
    }


