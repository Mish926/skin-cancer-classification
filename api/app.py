import io
import base64
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import cv2
from PIL import Image
from torchvision import transforms, models
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
import uvicorn

sys.path.insert(0, str(Path(__file__).parent.parent))


class SkinLesionClassifier(nn.Module):
    def __init__(self, num_classes=7, dropout_rate=0.5):
        super().__init__()
        backbone = models.resnet50(weights=None)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout_rate * 0.6),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        f = self.feature_extractor(x)
        f = torch.flatten(f, 1)
        return self.classifier(f)


CLASSES = [
    "Melanocytic nevi",
    "Melanoma",
    "Benign keratosis",
    "Basal cell carcinoma",
    "Actinic keratoses",
    "Vascular lesions",
    "Dermatofibroma",
]

RISK = {
    "Melanocytic nevi":     ("LOW",    "Benign mole. Monitor for changes over time."),
    "Melanoma":             ("HIGH",   "Malignant melanoma. Immediate dermatologist referral recommended."),
    "Benign keratosis":     ("LOW",    "Non-cancerous growth. Routine monitoring advised."),
    "Basal cell carcinoma": ("MEDIUM", "Common skin cancer. Dermatologist evaluation recommended."),
    "Actinic keratoses":    ("MEDIUM", "Pre-cancerous lesion. Treatment to prevent progression recommended."),
    "Vascular lesions":     ("LOW",    "Usually benign vascular lesion. Monitor for changes."),
    "Dermatofibroma":       ("LOW",    "Benign fibrous nodule. No immediate action required."),
}

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

device = torch.device("cpu")
model = SkinLesionClassifier()
model_path = Path(__file__).parent.parent / "results" / "best_model.pth"
state = torch.load(model_path, map_location=device)
if "model_state" in state:
    state = state["model_state"]
model.load_state_dict(state)
model.eval()
print(f"Model loaded from {model_path}")


class GradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        target = list(model.feature_extractor.children())[7][-1].conv3
        target.register_forward_hook(lambda m, i, o: setattr(self, "activations", o.detach()))
        target.register_backward_hook(lambda m, gi, go: setattr(self, "gradients", go[0].detach()))

    def generate(self, tensor, class_idx):
        self.model.zero_grad()
        out = self.model(tensor)
        out[0, class_idx].backward()
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1)).squeeze().numpy()
        cam = cv2.resize(cam, (224, 224))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def make_heatmap(original, cam):
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    orig = cv2.resize(np.array(original), (224, 224))
    return (0.5 * orig + 0.5 * heatmap).astype(np.uint8)


def to_b64(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


app = FastAPI(title="SkinSight — Skin Cancer Classification API")


@app.get("/health")
def health():
    return {"status": "ok", "model": "ResNet50", "classes": 7, "macro_auc": 0.911}


@app.get("/classes")
def get_classes():
    return {"classes": CLASSES}


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    contents = await image.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    tensor = TRANSFORM(img).unsqueeze(0)
    tensor.requires_grad_(True)

    with torch.set_grad_enabled(True):
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    top3_idx = torch.argsort(probs, descending=True)[:3].tolist()
    pred_idx = top3_idx[0]
    pred_class = CLASSES[pred_idx]
    confidence = round(probs[pred_idx].item() * 100, 2)
    risk, recommendation = RISK[pred_class]

    gradcam = GradCAM(model)
    cam = gradcam.generate(tensor, pred_idx)
    heatmap = make_heatmap(img, cam)

    return {
        "prediction": pred_class,
        "confidence": confidence,
        "risk_level": risk,
        "recommendation": recommendation,
        "top_3": [
            {"class": CLASSES[i], "confidence": round(probs[i].item() * 100, 2)}
            for i in top3_idx
        ],
        "gradcam_heatmap": to_b64(heatmap),
        "original_image": to_b64(np.array(img.resize((224, 224)))),
        "model_info": {
            "architecture": "ResNet50",
            "dataset": "HAM10000",
            "macro_auc": 0.911
        }
    }


@app.get("/", response_class=HTMLResponse)
def dashboard():
    with open(Path(__file__).parent / "templates" / "index.html") as f:
        return f.read()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
