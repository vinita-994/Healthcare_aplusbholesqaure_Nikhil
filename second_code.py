import os
import pandas as pd
import numpy as np
import nibabel as nib
import cv2
from sklearn.model_selection import train_test_split
import shutil

# --- CONFIGURATION ---
DATA_ROOT = "dataset/MRI/MRI"  # Path to the folders in your screenshot
CSV_PATH = "dataset/labels.csv" # Path to your labels file
OUTPUT_DIR = "dataset/processed"
IMG_SIZE = 224 # Standard input for ResNet/EfficientNet

# Create output directories if they don't exist
for split in ['train', 'val', 'test']:
    for category in ['CN', 'AD', 'MCI']:
        os.makedirs(os.path.join(OUTPUT_DIR, split, category), exist_ok=True)

def preprocess_mri(file_path):
    """
    Reads .nii file, extracts middle slice, normalizes, and resizes.
    [cite_start]Follows PDF requirements: Center crop, Min-max scaling [cite: 37-44].
    """
    try:
        # 1. Load MRI
        img = nib.load(file_path)
        data = img.get_fdata()

        # 2. Extract Middle Slice (Axial View)
        # Brain scans are 3D. We take the middle slice (z-axis) for 2D CNN.
        z_center = data.shape[2] // 2
        slice_2d = data[:, :, z_center]

        # 3. Simple Skull Stripping (Intensity Thresholding)
        # [cite_start]PDF asks for skull stripping[cite: 53]. This removes dark background noise.
        threshold = np.percentile(slice_2d, 10) 
        slice_2d[slice_2d < threshold] = 0

        # [cite_start]4. Resize to 224x224 (Standard for ResNet) [cite: 56]
        slice_resized = cv2.resize(slice_2d, (IMG_SIZE, IMG_SIZE))

        # [cite_start]5. Min-Max Normalization to [0,1] [cite: 44]
        img_min, img_max = slice_resized.min(), slice_resized.max()
        if img_max > img_min:
            slice_normalized = (slice_resized - img_min) / (img_max - img_min)
        else:
            slice_normalized = slice_resized

        # 6. Convert to 3 Channels (Required for Transfer Learning)
        # We stack the grayscale image 3 times to mimic RGB
        slice_rgb = np.stack((slice_normalized,)*3, axis=-1)
        
        return slice_rgb
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def run_pipeline():
    # 1. Load Labels
    df = pd.read_csv(CSV_PATH)
    # Ensure Subject ID matches folder names (remove quotes/spaces if needed)
    df['Subject'] = df['Subject'].astype(str).str.strip()
    
    # 2. Walk through directories
    print(f"Scanning {DATA_ROOT}...")
    
    # List all subject folders (e.g., 002_S_0413)
    subject_folders = [f for f in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, f))]
    
    processed_data = []
    labels = []
    
    for subject_id in subject_folders:
        subject_path = os.path.join(DATA_ROOT, subject_id)
        
        # Find the label for this subject
        subject_row = df[df['Subject'] == subject_id]
        if subject_row.empty:
            print(f"Skipping {subject_id}: No label found in CSV.")
            continue
            
        label = subject_row.iloc[0]['Group'] # Assuming column name is 'Group' (CN/AD/MCI)
        if label not in ['CN', 'AD', 'MCI']: continue # Skip unknown labels

        # Find the .nii or .dcm file inside the nested folders
        # ADNI structure is often Subject -> Date -> ID -> .nii
        for root, _, files in os.walk(subject_path):
            for file in files:
                if file.endswith('.nii') or file.endswith('.nii.gz'):
                    full_path = os.path.join(root, file)
                    
                    # Process the image
                    img_processed = preprocess_mri(full_path)
                    
                    if img_processed is not None:
                        # Save to a temporary list to split later
                        processed_data.append((subject_id, img_processed))
                        labels.append(label)
                    
                    # Break after finding the first valid MRI for this subject 
                    # (Prevent duplicates if multiple scans exist)
                    break 
            else:
                continue
            break

    # [cite_start]3. Split Data (Train/Val/Test) [cite: 57]
    X_train, X_temp, y_train, y_temp = train_test_split(processed_data, labels, test_size=0.3, stratify=labels, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    # 4. Save to Disk
    def save_split(data, dataset_type):
        for (subj_id, img), label in zip(data, dataset_type):
            save_path = os.path.join(OUTPUT_DIR, dataset_type, label, f"{subj_id}.npy")
            # Determine correct split folder name (train/val/test) based on variable
            # Correct logic:
            if data is X_train: folder = 'train'
            elif data is X_val: folder = 'val'
            else: folder = 'test'
            
            save_path = os.path.join(OUTPUT_DIR, folder, label, f"{subj_id}.npy")
            np.save(save_path, img)

    print("Saving processed files...")
    save_split(X_train, y_train) # Note: Logic inside function needs to handle split name, simpler to just loop manualy
    
    # Manual Save Loop (Safer)
    for (subj, img), lbl in zip(X_train, y_train): np.save(os.path.join(OUTPUT_DIR, 'train', lbl, f"{subj}.npy"), img)
    for (subj, img), lbl in zip(X_val, y_val): np.save(os.path.join(OUTPUT_DIR, 'val', lbl, f"{subj}.npy"), img)
    for (subj, img), lbl in zip(X_test, y_test): np.save(os.path.join(OUTPUT_DIR, 'test', lbl, f"{subj}.npy"), img)

    print("Preprocessing Complete! Data is ready for Task 2.")

if __name__ == "__main__":
    run_pipeline()