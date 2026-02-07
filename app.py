import torch
import torch.nn as nn
import numpy as np
import os
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["CN", "MCI", "AD"]

# ================= MODEL =================
class Alzheimer3DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(1, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(16, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 3)
        )

    def forward(self, x):
        return self.fc(self.conv(x))


model = Alzheimer3DCNN().to(DEVICE)
model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
model.eval()

# ================= GRAD-CAM =================
def generate_gradcam(model, input_tensor, target_class):
    gradients = []
    activations = []

    def forward_hook(module, inp, out):
        activations.append(out)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    # Hook LAST CONV LAYER
    target_layer = model.conv[8]  # Conv3d(32 → 64)

    fhook = target_layer.register_forward_hook(forward_hook)
    bhook = target_layer.register_backward_hook(backward_hook)

    output = model(input_tensor)
    loss = output[0, target_class]

    model.zero_grad()
    loss.backward()

    grads = gradients[0]
    fmap = activations[0]

    weights = torch.mean(grads, dim=(2, 3, 4), keepdim=True)
    cam = torch.sum(weights * fmap, dim=1).squeeze()

    cam = torch.relu(cam)
    cam = cam.detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    fhook.remove()
    bhook.remove()

    return cam


# ================= ROUTES =================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    filename = file.filename
    temp_path = "temp.npy"
    file.save(temp_path)

    # ==========================================
    # CASE 1: MRI Scan file → AI Diagnosis
    # ==========================================
    if "_X.npy" in filename:
        X = np.load(temp_path)

        X = (X - X.min()) / (X.max() - X.min() + 1e-8)
        X_tensor = torch.tensor(X).unsqueeze(0).unsqueeze(0).float().to(DEVICE)

        with torch.no_grad():
            output = model(X_tensor)
            probs = torch.softmax(output, dim=1).cpu().numpy()[0]

        best_idx = np.argmax(probs)
        diagnosis = CLASS_NAMES[best_idx]
        confidence = float(probs[best_idx]) * 100

        os.remove(temp_path)

        return jsonify({
            "type": "AI Prediction",
            "diagnosis": diagnosis,
            "confidence": round(confidence, 2),
            "probabilities": {
                "CN": float(probs[0]),
                "MCI": float(probs[1]),
                "AD": float(probs[2])
            }
        })

    # ==========================================
    # CASE 2: Label file → Show Ground Truth
    # ==========================================
    elif "_y.npy" in filename:
        label_data = np.load(temp_path, allow_pickle=True)
        os.remove(temp_path)

        if isinstance(label_data, np.ndarray):
            label_data = label_data.item() if label_data.size == 1 else label_data[0]

        if isinstance(label_data, str):
            diagnosis = label_data
        else:
            diagnosis = CLASS_NAMES[int(label_data)]

        return jsonify({
            "type": "Ground Truth Label",
            "diagnosis": diagnosis,
            "confidence": "100% (dataset label)"
        })

    # ==========================================
    # UNKNOWN FILE
    # ==========================================
    else:
        os.remove(temp_path)
        return jsonify({"error": "Unknown file type"})



    X = np.load(temp_path)
    X = (X - X.min()) / (X.max() - X.min() + 1e-8)
    X_tensor = torch.tensor(X).unsqueeze(0).unsqueeze(0).float().to(DEVICE)

    with torch.no_grad():
        output = model(X_tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]

    best_idx = np.argmax(probs)
    diagnosis = CLASS_NAMES[best_idx]
    confidence = float(probs[best_idx]) * 100

    # Generate heatmap
    cam = generate_gradcam(model, X_tensor, best_idx)
    np.save("static/heatmap.npy", cam)

    os.remove(temp_path)

    return jsonify({
        "diagnosis": diagnosis,
        "confidence": round(confidence, 2),
        "probabilities": {
            "CN": float(probs[0]),
            "MCI": float(probs[1]),
            "AD": float(probs[2])
        }
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
