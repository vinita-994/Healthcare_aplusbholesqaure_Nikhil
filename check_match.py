import os
import pandas as pd

# 1. Setup Paths
PROCESSED_DIR = "dataset/processed"
CSV_PATH = "dataset/labels.csv"

# 2. Check if folder exists
if not os.path.exists(PROCESSED_DIR):
    print(f"❌ Error: Folder '{PROCESSED_DIR}' not found!")
    exit()

# 3. List the first 5 files found
files = [f for f in os.listdir(PROCESSED_DIR) if f.endswith(".npy")]
print(f"📂 Found {len(files)} .npy files in processed folder.")
if len(files) > 0:
    print("   First 5 filenames:", files[:5])
    sample_id_from_file = files[0].replace(".npy", "")
else:
    print("   ❌ No .npy files found! Did the preprocessing script finish?")
    exit()

# 4. Load CSV and check first 5 rows
if not os.path.exists(CSV_PATH):
    print(f"❌ Error: CSV file '{CSV_PATH}' not found!")
    exit()

try:
    df = pd.read_csv(CSV_PATH)
    print("\n📊 CSV Loaded Successfully.")
    print("   Columns found:", df.columns.tolist())
    print("   First 5 rows of Subject column:")
    
    # Try to find the 'Subject' column automatically
    if 'Subject' in df.columns:
        print(df['Subject'].head().tolist())
        
        # 5. TEST THE MATCH
        print("\n🔍 MATCH TEST:")
        print(f"   Checking if file ID '{sample_id_from_file}' is in CSV...")
        
        # Clean both to be sure
        match = df[df['Subject'].astype(str).str.strip() == sample_id_from_file]
        
        if not match.empty:
            print("   ✅ SUCCESS! Found a match.")
            print("   Label:", match.iloc[0]['Group'])
        else:
            print("   ❌ FAILURE! No match found.")
            print("   (This means your filenames do not match the IDs in the CSV exactly.)")
            
    else:
        print("   ❌ ERROR: Column 'Subject' not found in CSV. Please rename the correct column.")

except Exception as e:
    print(f"❌ CSV Error: {e}")