import pandas as pd
import os

# Configuration
root_dir = 'Data/Preprocessed'
output_root = 'Data/Cleaned'
subfolders = ['3_Changepoint', '4_Changepoint', '5_Changepoint', '6_Changepoint', 'Targeted']

# This set will store source_video_paths that must be deleted globally
invalid_video_paths = set()

print("Step 1: Identifying invalid videos across all files...")

# --- PASS 1: Identify "Bad" Videos ---
for folder in subfolders:
    input_path = os.path.join(root_dir, folder)
    if not os.path.exists(input_path):
        continue
        
    for filename in os.listdir(input_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(input_path, filename)
            df = pd.read_csv(file_path, usecols=['source_video_path', 'face_id', 'success'])
            
            # Find video paths where face_id is not 0 OR success is not 1
            bad_rows = df[(df['face_id'] != 0) | (df['success'] != 1)]
            bad_paths = bad_rows['source_video_path'].unique()
            
            # Add these to our global "blacklist"
            invalid_video_paths.update(bad_paths)

print(f"Total unique videos to be removed: {len(invalid_video_paths)}")

# --- PASS 2: Clean and Save ---
print("\nStep 2: Cleaning files and removing associated rows...")

for folder in subfolders:
    input_path = os.path.join(root_dir, folder)
    output_path = os.path.join(output_root, folder)
    
    if not os.path.exists(input_path):
        continue
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    for filename in os.listdir(input_path):
        if filename.endswith('.csv'):
            print(f"Processing: {folder}/{filename}")
            
            file_path = os.path.join(input_path, filename)
            df = pd.read_csv(file_path)
            
            # 1. REMOVE ALL ROWS belonging to any invalid source_video_path
            # This satisfies: "remove all rows associated with that source_video_path"
            clean_df = df[~df['source_video_path'].isin(invalid_video_paths)].copy()
            
            # 2. DROP REQUESTED COLUMNS
            # They are no longer needed because we've filtered the data strictly
            cols_to_drop = ['face_id', 'success', 'confidence']
            clean_df = clean_df.drop(columns=[c for c in cols_to_drop if c in clean_df.columns])
            
            # Note: Chronological order is preserved as we haven't touched frame_id order
            
            # 3. SAVE
            save_path = os.path.join(output_path, filename)
            clean_df.to_csv(save_path, index=False)

print("\nTask Complete. Your Train, Test, and Validation sets are now synchronized.")