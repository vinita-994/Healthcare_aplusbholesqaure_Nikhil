# =========================================================
# HIGH-PERFORMANCE MRI CLASSIFIER (RESNET + FOCAL LOSS)
# =========================================================

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from scipy.ndimage import rotate

# ================= CONFIGURATION =================
CONFIG = {
    "lr": 1e-4,
    "batch_size": 2,          # Keep low for VRAM
    "accum_steps": 16,        # Simulates batch_size = 32
    "epochs": 50,
    "patience": 10,
    "path": "dataset/processed",
    "save_path": "best_mri_model.pth"
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= UTILS: FOCAL LOSS =================
# Helps model focus on "hard" examples rather than easy background
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

# ================= ROBUST DATASET =================
class AdvancedMRIDataset(Dataset):
    def __init__(self, folder, train=True):
        self.folder = folder
        self.samples = sorted([f.replace("_X.npy","") for f in os.listdir(folder) if f.endswith("_X.npy")])
        self.train = train

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sid = self.samples[idx]
        X = np.load(os.path.join(self.folder, sid+"_X.npy")).astype(np.float32)
        y = np.load(os.path.join(self.folder, sid+"_y.npy"))

        # --- 1. ROBUST NORMALIZATION ---
        # Clip top/bottom 1% intensities to remove outliers (MRI artifacts)
        p1 = np.percentile(X, 1)
        p99 = np.percentile(X, 99)
        X = np.clip(X, p1, p99)
        # Min-Max scale to 0-1 range
        X = (X - X.min()) / (X.max() - X.min() + 1e-8)

        # --- 2. ADVANCED AUGMENTATION ---
        if self.train:
            # Random Rotation (-10 to 10 degrees)
            if np.random.rand() > 0.5:
                angle = np.random.uniform(-10, 10)
                # Rotate around Z-axis (axial plane)
                X = rotate(X, angle, axes=(1, 2), reshape=False, mode='nearest')
            
            # Random Intensity Shift
            if np.random.rand() > 0.5:
                X = X * np.random.uniform(0.9, 1.1)

            # Elastic-like Flips
            if np.random.rand() > 0.5: X = np.flip(X, axis=0) # Sagittal
            if np.random.rand() > 0.5: X = np.flip(X, axis=2) # Axial

        X = torch.tensor(X.copy()).unsqueeze(0) # Add Channel Dim
        y = torch.tensor(0 if y == "CN" else 1, dtype=torch.long)
        return X, y

# ================= MODEL: RESIDUAL 3D CNN =================
class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_c)
        self.conv2 = nn.Conv3d(out_c, out_c, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_c)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_c, out_c, 1, stride=stride, bias=False),
                nn.BatchNorm3d(out_c)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class ResNet3D_Light(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU()
        )
        
        # Deep Residual Layers
        self.layer1 = ResidualBlock(32, 32)
        self.layer2 = ResidualBlock(32, 64, stride=2) # Downsample
        self.layer3 = ResidualBlock(64, 128, stride=2) # Downsample
        self.pool = nn.AdaptiveAvgPool3d(1)
        
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.pre(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = x.flatten(1)
        return self.fc(x)

# ================= SETUP =================
full_dataset = AdvancedMRIDataset(CONFIG["path"], train=True)

# Stratified Split
labels = [0 if np.load(os.path.join(CONFIG["path"], s+"_y.npy"))=="CN" else 1 for s in full_dataset.samples]
train_idx, val_idx = train_test_split(list(range(len(full_dataset))), test_size=0.2, stratify=labels, random_state=42)

train_loader = DataLoader(Subset(full_dataset, train_idx), batch_size=CONFIG["batch_size"], shuffle=True, num_workers=2)
val_loader = DataLoader(Subset(AdvancedMRIDataset(CONFIG["path"], train=False), val_idx), batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2)

model = ResNet3D_Light().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=1e-3)
criterion = FocalLoss(alpha=1, gamma=2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])

# ================= TRAINING LOOP =================
print("🚀 Starting High-Performance Training...")
best_auc = 0
trigger_times = 0

for epoch in range(CONFIG["epochs"]):
    model.train()
    train_loss = 0
    optimizer.zero_grad() 

    for i, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)
        
        out = model(X)
        loss = criterion(out, y)
        
        # GRADIENT ACCUMULATION (Simulates larger batch size)
        loss = loss / CONFIG["accum_steps"]
        loss.backward()

        if (i + 1) % CONFIG["accum_steps"] == 0:
            optimizer.step()
            optimizer.zero_grad()
            
        train_loss += loss.item() * CONFIG["accum_steps"]

    # ===== VALIDATION =====
    model.eval()
    val_loss = 0
    y_true, y_scores = [], []
    
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            val_loss += criterion(out, y).item()
            
            probs = torch.softmax(out, 1)[:, 1].cpu().numpy()
            y_true.extend(y.cpu().numpy())
            y_scores.extend(probs)

    try:
        current_auc = roc_auc_score(y_true, y_scores)
    except:
        current_auc = 0.5 # Handle single class edge case

    scheduler.step()
    
    print(f"Epoch {epoch+1}/{CONFIG['epochs']} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f} | AUC: {current_auc:.4f}")

    # ===== EARLY STOPPING & SAVING =====
    if current_auc > best_auc:
        best_auc = current_auc
        torch.save(model.state_dict(), CONFIG["save_path"])
        print(f"    >>> New Best Model Saved! (AUC: {best_auc:.4f})")
        trigger_times = 0
    else:
        trigger_times += 1
        if trigger_times >= CONFIG["patience"]:
            print("Early stopping!")
            break

# ================= FINAL REPORT =================
print("\n📊 FINAL EVALUATION")
model.load_state_dict(torch.load(CONFIG["save_path"]))
model.eval()
y_pred = []
y_true = []
y_scores = []

with torch.no_grad():
    for X, y in val_loader:
        X = X.to(device)
        out = model(X)
        probs = torch.softmax(out, 1)[:, 1].cpu().numpy()
        preds = (probs > 0.4).astype(int) # Optimized threshold
        
        y_true.extend(y.numpy())
        y_pred.extend(preds)
        y_scores.extend(probs)

print(classification_report(y_true, y_pred))
print(f"Final AUC: {roc_auc_score(y_true, y_scores):.4f}")
print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
