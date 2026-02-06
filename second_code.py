import os
import numpy as np
import pydicom
import warnings
from skimage.transform import resize

# --- CONFIGURATION ---
# Path where your raw MRI folders are located
# Based on your screenshots, this is likely "dataset/MRI" or just "MRI"
DICOM_ROOT = "dataset/MRI" 

# Path to save the clean files
OUTPUT_DIR = "dataset/processed"
TARGET_SHAPE = (128, 128, 128)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- FUNCTIONS ---
def load_dicom_volume(folder_path):
    """Reads all .dcm files in a folder and stacks them into a 3D block."""
    slices = []
    files = [f for f in os.listdir(folder_path) if f.endswith(".dcm")]
    
    if len(files) < 10: # Skip empty or corrupted folders
        return None

    for f in files:
        try:
            ds = pydicom.dcmread(os.path.join(folder_path, f))
            slices.append(ds)
        except:
            continue
            
    # Sort slices by position (z-axis) to ensure correct brain structure
    try:
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except:
        slices.sort(key=lambda x: x.filename) # Fallback sort

    # Stack into 3D array
    volume = np.stack([s.pixel_array for s in slices])
    return volume.astype(np.float32)

def preprocess_volume(volume):
    """Resizes to 128x128x128 and normalizes to [0,1]."""
    # Resize
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        volume = resize(volume, TARGET_SHAPE, mode='constant', preserve_range=True)
    
    # Normalize
    vol_min, vol_max = np.min(volume), np.max(volume)
    if vol_max > vol_min:
        volume = (volume - vol_min) / (vol_max - vol_min)
        
    return volume

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print(f"🔄 Scanning '{DICOM_ROOT}' for MRI scans...")
    processed_count = 0
    
    # Walk through every folder to find DICOMs
    for root, dirs, files in os.walk(DICOM_ROOT):
        if any(f.endswith(".dcm") for f in files):
            
            # Use the folder name as the ID (e.g., I37402)
            image_id = os.path.basename(root)
            save_path = os.path.join(OUTPUT_DIR, f"{image_id}.npy")
            
            if os.path.exists(save_path):
                print(f"⚠️  Skipping {image_id} (Already exists)")
                continue
                
            try:
                # 1. Load
                vol = load_dicom_volume(root)
                if vol is None: continue
                
                # 2. Process
                clean_vol = preprocess_volume(vol)
                
                # 3. Save
                np.save(save_path, clean_vol)
                print(f"✅ Processed: {image_id} | Shape: {clean_vol.shape}")
                processed_count += 1
                
            except Exception as e:
                print(f"❌ Error on {image_id}: {e}")

    print(f"\n🎉 DONE! Processed {processed_count} volumes.")
    print(f"📂 Files saved in: {OUTPUT_DIR}")