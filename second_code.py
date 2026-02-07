# =========================================================
# HIGH-PERFORMANCE BUT GENTLE 2D MRI CLASSIFIER
# Target Balanced Accuracy > 91%
# =========================================================

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, confusion_matrix
from scipy.ndimage import rotate

DATA_PATH = "dataset/processed"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 6
EPOCHS = 40
LR = 1e-4

# ================= SAFE DATA LOADING =================
samples, labels = [], []

for f in os.listdir(DATA_PATH):
    if f.endswith("_X.npy"):
        sid = f.replace("_X.npy","")
        try:
            vol = np.load(os.path.join(DATA_PATH, sid+"_X.npy"))
            lab = str(np.load(os.path.join(DATA_PATH, sid+"_y.npy")))
            if vol.size == 0: continue
            samples.append(sid)
            labels.append(0 if lab=="CN" else 1)
        except:
            continue

samples = np.array(samples)
labels = np.array(labels)

# =====================================================
class MRIDataset(Dataset):
    def __init__(self, indices, train=True):
        self.indices = indices
        self.train = train

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224,224)),
            transforms.Normalize([0.5],[0.5])
        ])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sid = samples[self.indices[idx]]
        vol = np.load(os.path.join(DATA_PATH, sid+"_X.npy")).astype(np.float32)
        label = labels[self.indices[idx]]

        # Soft normalization
        p1,p99 = np.percentile(vol,(1,99))
        vol = np.clip(vol,p1,p99)
        vol = (vol-vol.min())/(vol.max()-vol.min()+1e-8)

        # 🔹 Use 3 central slices (better signal)
        mid = vol.shape[0]//2
        slices = [vol[mid-2], vol[mid], vol[mid+2]]

        imgs = []
        for s in slices:
            if self.train and np.random.rand()>0.6:
                s = rotate(s, np.random.uniform(-8,8), reshape=False)

            img = np.stack([s,s,s], axis=-1)
            img = self.transform(img)
            imgs.append(img)

        imgs = torch.stack(imgs)  # [3,3,224,224]
        return imgs.float(), torch.tensor(label).long()

# =====================================================
class MRIModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(weights="IMAGENET1K_V1")
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(512,2)
        )

    def forward(self,x):  # x [B,S,3,224,224]
        B,S,C,H,W = x.shape
        x = x.view(B*S,C,H,W)
        feats = self.backbone(x)
        feats = feats.view(B,S,512).mean(dim=1)  # gentle slice fusion
        return self.classifier(feats)

# =====================================================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fold_accs, fold_aucs = [], []

for fold,(train_idx,val_idx) in enumerate(skf.split(samples,labels)):
    print(f"\n=========== FOLD {fold+1} ===========")

    train_ds = MRIDataset(train_idx,train=True)
    val_ds   = MRIDataset(val_idx,train=False)

    train_loader = DataLoader(train_ds,batch_size=BATCH_SIZE,shuffle=True)
    val_loader   = DataLoader(val_ds,batch_size=BATCH_SIZE)

    model = MRIModel().to(DEVICE)

    # Class weights (helps balanced accuracy)
    class_counts = np.bincount(labels[train_idx])
    weights = torch.tensor([1/c for c in class_counts], dtype=torch.float32).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,EPOCHS)

    best_auc, patience, counter = 0, 7, 0

    for epoch in range(EPOCHS):
        model.train()
        for imgs,y in train_loader:
            imgs,y = imgs.to(DEVICE),y.to(DEVICE)
            loss = criterion(model(imgs),y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        model.eval()
        y_true,y_scores=[],[]

        with torch.no_grad():
            for imgs,y in val_loader:
                imgs = imgs.to(DEVICE)
                probs = torch.softmax(model(imgs),1)[:,1].cpu().numpy()
                y_scores.extend(probs); y_true.extend(y.numpy())

        auc = roc_auc_score(y_true,y_scores)
        scheduler.step()
        print(f"Epoch {epoch+1} AUC: {auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(),f"best_fold_{fold}.pth")
            counter = 0
        else:
            counter += 1
            if counter>=patience:
                print("Early stopping"); break

    model.load_state_dict(torch.load(f"best_fold_{fold}.pth",weights_only=True))
    model.eval()

    y_true,y_pred,y_scores=[],[],[]
    with torch.no_grad():
        for imgs,y in val_loader:
            imgs = imgs.to(DEVICE)
            probs = torch.softmax(model(imgs),1)[:,1].cpu().numpy()
            preds = (probs>0.5).astype(int)
            y_true.extend(y.numpy()); y_pred.extend(preds); y_scores.extend(probs)

    bal_acc = balanced_accuracy_score(y_true,y_pred)
    auc = roc_auc_score(y_true,y_scores)

    print("Balanced Accuracy:",bal_acc)
    print("AUC:",auc)
    print(confusion_matrix(y_true,y_pred))

    fold_accs.append(bal_acc)
    fold_aucs.append(auc)

print("\n===== FINAL RESULTS =====")
print("Mean Balanced Accuracy:",np.mean(fold_accs))
print("Mean AUC:",np.mean(fold_aucs))
