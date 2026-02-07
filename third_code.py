# ============================================================
# TASK 3 — MULTI-CLASS MRI CLASSIFICATION (CN vs MCI vs AD)
# Robust • Safe • Works on corrupted datasets
# ============================================================

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, precision_score,
    recall_score, confusion_matrix, roc_auc_score
)

# ================= SETTINGS =================
DATA_DIR = "dataset/processed"
BATCH_SIZE = 4
EPOCHS = 15
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_MAP = {"CN": 0, "MCI": 1, "AD": 2}

# ============================================================
# SAFE DATASET (SKIPS BROKEN FILES)
# ============================================================
class SafeMRIDataset(Dataset):
    def __init__(self, folder, indices=None, train=True):
        self.folder = folder
        self.train = train
        self.samples = []

        all_ids = [f.replace("_X.npy", "") for f in os.listdir(folder) if f.endswith("_X.npy")]

        for sid in all_ids:
            x_path = os.path.join(folder, sid+"_X.npy")
            y_path = os.path.join(folder, sid+"_y.npy")

            if not os.path.exists(y_path):
                continue

            try:
                X = np.load(x_path)
                if X.size == 0:
                    print(f"⚠️ Empty MRI skipped: {sid}")
                    continue

                y = str(np.load(y_path))
                if y not in CLASS_MAP:
                    continue

                self.samples.append(sid)

            except:
                print(f"⚠️ Corrupted file skipped: {sid}")

        if indices is not None:
            self.samples = [self.samples[i] for i in indices if i < len(self.samples)]

        print(f"✅ Dataset contains {len(self.samples)} valid scans")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sid = self.samples[idx]

        X = np.load(os.path.join(self.folder, sid+"_X.npy"))
        y = CLASS_MAP[str(np.load(os.path.join(self.folder, sid+"_y.npy")))]

        # -------- AUGMENTATION --------
        if self.train:
            if np.random.rand() > 0.5:
                X = np.flip(X, axis=1).copy()
            if np.random.rand() > 0.5:
                X = np.flip(X, axis=2).copy()

        X = torch.tensor(X).unsqueeze(0).float()
        return X, torch.tensor(y)

# ============================================================
# 3D CNN MODEL
# ============================================================
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

# ============================================================
# LOAD LABELS SAFELY
# ============================================================
full_dataset = SafeMRIDataset(DATA_DIR)
labels = [CLASS_MAP[str(np.load(os.path.join(DATA_DIR, s+"_y.npy")))]
          for s in full_dataset.samples]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ============================================================
# TRAINING LOOP
# ============================================================
for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
    print(f"\n================ FOLD {fold+1} ================")

    train_ds = SafeMRIDataset(DATA_DIR, train_idx, train=True)
    val_ds   = SafeMRIDataset(DATA_DIR, val_idx, train=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = Alzheimer3DCNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # ================= TRAIN =================
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0

        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss {train_loss:.3f}")

    # ================= VALIDATION =================
    model.eval()
    preds, trues, probs = [], [], []

    with torch.no_grad():
        for X, y in val_loader:
            X = X.to(DEVICE)
            out = model(X)
            preds.extend(out.argmax(1).cpu().numpy())
            trues.extend(y.numpy())
            probs.extend(torch.softmax(out, 1).cpu().numpy())

    print("\n📊 EVALUATION METRICS")
    print("Balanced Accuracy:", balanced_accuracy_score(trues, preds))
    print("Macro F1:", f1_score(trues, preds, average="macro"))
    print("Precision per class:", precision_score(trues, preds, average=None))
    print("Recall per class:", recall_score(trues, preds, average=None))
    print("AUC (OvR):", roc_auc_score(trues, probs, multi_class="ovr"))
    print("Confusion Matrix:\n", confusion_matrix(trues, preds))

    break  # remove this break to run all 5 folds
