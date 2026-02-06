# =========================================================
# MRI Alzheimer (AD) vs CN Classifier — Stable Version
# =========================================================

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

# ================= PATH =================
DATA_DIR = "dataset/processed"

# ================= DATASET =================
class MRIDataset(Dataset):
    def __init__(self, folder, train=True):
        self.folder = folder
        self.samples = [f.replace("_X.npy","") for f in os.listdir(folder) if f.endswith("_X.npy")]
        self.train = train

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sid = self.samples[idx]
        X = np.load(os.path.join(self.folder, sid+"_X.npy"))
        y = np.load(os.path.join(self.folder, sid+"_y.npy"))

        # Safe augmentations (MRI is 3D → axes 0,1,2 only)
        if self.train:
            if np.random.rand() > 0.5:
                X = np.flip(X, axis=1).copy()
            if np.random.rand() > 0.5:
                X = np.flip(X, axis=2).copy()

        X = torch.tensor(X).unsqueeze(0).float()
        y = torch.tensor(0 if y=="CN" else 1).long()
        return X, y

# ================= LOAD DATA =================
full_dataset = MRIDataset(DATA_DIR, train=True)

labels = [0 if np.load(os.path.join(DATA_DIR, s+"_y.npy"))=="CN" else 1
          for s in full_dataset.samples]

train_idx, val_idx = train_test_split(
    list(range(len(full_dataset))),
    test_size=0.3,
    stratify=labels,
    random_state=42
)

train_ds = Subset(full_dataset, train_idx)
val_ds   = Subset(MRIDataset(DATA_DIR, train=False), val_idx)

train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=2)

# ================= DEVICE =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= CLASS WEIGHTS =================
weights = compute_class_weight(class_weight='balanced',
                               classes=np.array([0,1]),
                               y=np.array(labels))
weights = torch.tensor(weights, dtype=torch.float32).to(device)

# ================= MODEL =================
class Alzheimer3DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(1,32,3,padding=1), nn.BatchNorm3d(32), nn.ReLU(),
            nn.Conv3d(32,32,3,padding=1), nn.BatchNorm3d(32), nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(32,64,3,padding=1), nn.BatchNorm3d(64), nn.ReLU(),
            nn.Conv3d(64,64,3,padding=1), nn.BatchNorm3d(64), nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(64,128,3,padding=1), nn.BatchNorm3d(128), nn.ReLU(),
            nn.AdaptiveAvgPool3d(1)
        )

        self.fc = nn.Sequential(
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64,2)
        )

    def forward(self,x):
        x=self.conv(x)
        x=x.view(x.size(0),-1)
        return self.fc(x)

model = Alzheimer3DCNN().to(device)

criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ================= TRAIN =================
print("🚀 Training Model...")
best_auc = 0

for epoch in range(25):
    model.train()
    train_loss = 0

    for X,y in train_loader:
        X,y=X.to(device),y.to(device)
        optimizer.zero_grad()
        out=model(X)
        loss=criterion(out,y)
        loss.backward()
        optimizer.step()
        train_loss+=loss.item()

    # Validation
    model.eval()
    y_true,y_scores=[],[]
    with torch.no_grad():
        for X,y in val_loader:
            X=X.to(device)
            out=model(X)
            probs=torch.softmax(out,dim=1)[:,1].cpu().numpy()
            y_scores.extend(probs)
            y_true.extend(y.numpy())

    auc = roc_auc_score(y_true,y_scores)
    print(f"Epoch {epoch+1} | Loss {train_loss:.3f} | AUC {auc:.3f}")

    if auc > best_auc:
        best_auc = auc
        torch.save(model.state_dict(),"best_model.pth")

# ================= EVALUATION =================
print("\n📊 FINAL EVALUATION")
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

y_true,y_pred,y_scores=[],[],[]

with torch.no_grad():
    for X,y in val_loader:
        X=X.to(device)
        out=model(X)
        probs=torch.softmax(out,dim=1)[:,1].cpu().numpy()
        preds=torch.argmax(out,1).cpu().numpy()

        y_true.extend(y.numpy())
        y_pred.extend(preds)
        y_scores.extend(probs)

print(classification_report(y_true,y_pred,target_names=["CN","AD"]))
print("AUC:", roc_auc_score(y_true,y_scores))

# ===== MANUAL CONFUSION MATRIX =====
y_true=np.array(y_true)
y_pred=np.array(y_pred)

tn = np.sum((y_true==0)&(y_pred==0))
fp = np.sum((y_true==0)&(y_pred==1))
fn = np.sum((y_true==1)&(y_pred==0))
tp = np.sum((y_true==1)&(y_pred==1))

cm = np.array([[tn,fp],
               [fn,tp]])

print("\nConfusion Matrix:")
print(cm)
