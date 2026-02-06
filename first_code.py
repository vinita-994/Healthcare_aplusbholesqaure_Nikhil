import os
import numpy as np
import pandas as pd
import pydicom
import nibabel as nib
import ants
from skimage.transform import resize

# ================= PATHS =================
DICOM_DIR = "MRI"
CSV_LABELS = "MRI_metadata.csv"
OUT_DIR = "dataset/processed"
TARGET_SHAPE = (128,128,128)

os.makedirs(OUT_DIR, exist_ok=True)

# ================= LOAD CSV =================
labels_df = pd.read_csv(CSV_LABELS)

# Keep only needed columns
labels_df = labels_df[["Subject", "Group"]]
labels_df.columns = ["Subject", "label"]

print("🔄 Starting FULL MRI preprocessing pipeline...")

# ================= DICOM → NIFTI =================
def dicom_to_nifti(folder):
    slices = []
    for f in os.listdir(folder):
        if f.endswith(".dcm"):
            ds = pydicom.dcmread(os.path.join(folder, f))
            slices.append(ds)

    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    volume = np.stack([s.pixel_array for s in slices]).astype(np.float32)
    return nib.Nifti1Image(volume, affine=np.eye(4))

# ================= SKULL STRIP =================
def skull_strip(nifti_img):
    ants_img = ants.from_numpy(nifti_img.get_fdata())
    brain_mask = ants.get_mask(ants_img)
    return ants_img * brain_mask

# ================= MNI REGISTRATION =================
def register_to_mni(brain_img):
    mni = ants.image_read(ants.get_ants_data('mni'))
    reg = ants.registration(fixed=mni, moving=brain_img, type_of_transform='Affine')
    return reg['warpedmovout']

# ================= GM SEGMENTATION =================
def segment_gm(img):
    seg = ants.kmeans_segmentation(img, k=3)['segmentation']
    gm = (seg == 2) * img
    return gm

# ================= FINAL PREPROCESS =================
def final_preprocess(img):
    vol = img.numpy()
    vol = resize(vol, TARGET_SHAPE, mode='constant', preserve_range=True)
    vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)
    return vol.astype(np.float32)

# ================= MAIN LOOP =================
count = 0

for root, dirs, files in os.walk(DICOM_DIR):
    if any(f.endswith(".dcm") for f in files):
        try:
            # Extract subject ID from folder path
            parts = root.split(os.sep)
            Subject = next((p for p in parts if "_S_" in p), None)

            if Subject is None:
                print("⚠ Subject ID not found:", root)
                continue

            label_row = labels_df[labels_df["Subject"] == Subject]
            if label_row.empty:
                print("⚠ Label missing for", Subject)
                continue

            label = label_row["label"].values[0]

            # ---- PROCESS PIPELINE ----
            nifti = dicom_to_nifti(root)
            brain = skull_strip(nifti)
            registered = register_to_mni(brain)
            gm = segment_gm(registered)
            processed = final_preprocess(gm)

            np.save(os.path.join(OUT_DIR, f"{Subject}_X.npy"), processed)
            np.save(os.path.join(OUT_DIR, f"{Subject}_y.npy"), label)

            print("✅ Finished:", Subject)
            count += 1

        except Exception as e:
            print("❌ Error with", root, e)

print(f"\n🎉 TASK-1 COMPLETE → {count} MRI volumes processed")
print("📂 Raw DICOM data remains untouched.")
