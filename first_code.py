import os
import numpy as np
import pydicom
from skimage.transform import resize

# 🔒 Raw DICOM folder
DICOM_DIR = "MRI"

# ✅ Processed output
PROCESSED_DIR = "dataset/processed"

TARGET_SHAPE = (128, 128, 128)

os.makedirs(PROCESSED_DIR, exist_ok=True)

print("🔄 Reading DICOM scans without modifying originals...")

def load_dicom_series(folder):
    slices = []
    
    for f in os.listdir(folder):
        if f.endswith(".dcm"):
            path = os.path.join(folder, f)
            ds = pydicom.dcmread(path)
            slices.append(ds)

    # Sort by slice position
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

    volume = np.stack([s.pixel_array for s in slices])
    return volume.astype(np.float32)

def preprocess_volume(volume):
    volume = resize(volume, TARGET_SHAPE, mode='constant', preserve_range=True)
    volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume) + 1e-8)
    return volume

series_count = 0

# Find folders that contain DICOM slices
for root, dirs, files in os.walk(DICOM_DIR):
    if any(f.endswith(".dcm") for f in files):
        try:
            volume = load_dicom_series(root)
            volume = preprocess_volume(volume)

            sid = os.path.basename(root)
            np.save(os.path.join(PROCESSED_DIR, sid + ".npy"), volume)

            print("✅ Processed series:", sid)
            series_count += 1

        except Exception as e:
            print("❌ Skipped folder:", root, "Error:", e)

print(f"🎉 Done! {series_count} MRI volumes processed.")
print("📂 Raw DICOM data remains untouched.")
