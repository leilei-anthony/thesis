import pandas as pd
import numpy as np
import os
from scipy.stats import linregress

# --- CONFIGURATION ---
INPUT_ROOT = 'Data/Cleaned'
OUTPUT_ROOT = 'Data/Engineered'
SUBFOLDERS = ['3_Changepoint', '4_Changepoint', '5_Changepoint', '6_Changepoint', 'Targeted']

# --- HELPER FUNCTIONS ---

def get_dist(df, p1, p2):
    """Calculates Euclidean distance between two landmark points (x, y)."""
    # Safeguard against missing columns
    if f'x_{p1}' not in df.columns or f'x_{p2}' not in df.columns:
        return 0
    return np.sqrt((df[f'x_{p1}'] - df[f'x_{p2}'])**2 + (df[f'y_{p1}'] - df[f'y_{p2}'])**2)

def get_slope(y):
    """Calculates the linear trend (slope) across the frames for a video."""
    y = y.values
    if len(y) < 2 or np.all(y == y[0]): 
        return 0.0
    x = np.arange(len(y))
    slope, _, _, _, _ = linregress(x, y)
    return slope

def engineer_biometrics(df):
    """Creates relational features from raw landmarks and MediaPipe."""
    
    # 1. Brow Distance (Confusion Indicator - AU4 proxy)
    # Distance between eyebrow inner corner and eye inner corner
    df['feat_brow_dist_left'] = get_dist(df, 21, 39)
    df['feat_brow_dist_right'] = get_dist(df, 22, 42)
    df['feat_brow_dist_avg'] = (df['feat_brow_dist_left'] + df['feat_brow_dist_right']) / 2

    # 2. Eye Aspect Ratio (EAR) - Boredom/Drowsiness proxy
    def calc_ear(df, p):
        v1 = get_dist(df, p+1, p+5)
        v2 = get_dist(df, p+2, p+4)
        h  = get_dist(df, p, p+3)
        return (v1 + v2) / (2.0 * h + 1e-6)
    df['feat_EAR'] = (calc_ear(df, 36) + calc_ear(df, 42)) / 2

    # 3. Mouth Aspect Ratio (MAR) - Talking/Yawning proxy
    df['feat_MAR'] = get_dist(df, 51, 57) / (get_dist(df, 48, 54) + 1e-6)

    # 4. Upper Body Metrics (MediaPipe)
    if 'MP_LEFT_SHOULDER_y' in df.columns:
        # Slump: Distance from nose (30) to shoulder line
        shoulder_mid_y = (df['MP_LEFT_SHOULDER_y'] + df['MP_RIGHT_SHOULDER_y']) / 2
        df['feat_slump'] = shoulder_mid_y - df['y_30']
        
        # Lean In: Shoulder width (Increases as student leans toward camera)
        df['feat_shoulder_width'] = np.sqrt(
            (df['MP_LEFT_SHOULDER_x'] - df['MP_RIGHT_SHOULDER_x'])**2 + 
            (df['MP_LEFT_SHOULDER_y'] - df['MP_RIGHT_SHOULDER_y'])**2
        )
    
    return df

def aggregate_video_data(df):
    """Condenses multiple frames into 1 statistical summary row per video."""
    
    # Identify feature columns to aggregate
    feat_cols = [c for c in df.columns if c.startswith('AU') or 
                 c.startswith('pose_') or c.startswith('gaze_') or 
                 c.startswith('feat_')]
    
    # Metadata and Labels to preserve
    meta_cols = ['person_id', 'Boredom', 'Engagement', 'Confusion', 'Frustration']
    
    # Define aggregation: Mean, Std, Max, Min
    agg_dict = {col: ['mean', 'std', 'max', 'min'] for col in feat_cols}
    for col in meta_cols:
        if col in df.columns: agg_dict[col] = 'first'
    
    # Execute GroupBy
    video_grouped = df.groupby('source_video_path')
    agg_df = video_grouped.agg(agg_dict)
    
    # Flatten multi-index columns (e.g., ('AU01_r', 'mean') -> 'AU01_r_mean')
    agg_df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in agg_df.columns]
    
    # Calculate Range and Slope for every feature
    for col in feat_cols:
        # Range = Intensity of emotional reaction
        agg_df[f'{col}_range'] = agg_df[f'{col}_max'] - agg_df[f'{col}_min']
        
        # Slope = Trend of emotional reaction
        agg_df[f'{col}_slope'] = video_grouped[col].apply(get_slope)

    return agg_df.reset_index()

# --- MAIN RUN LOGIC ---

if __name__ == "__main__":
    print("Starting Advanced Feature Engineering...")
    
    for folder in SUBFOLDERS:
        input_dir = os.path.join(INPUT_ROOT, folder)
        output_dir = os.path.join(OUTPUT_ROOT, folder)
        
        if not os.path.exists(input_dir):
            continue
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        for filename in os.listdir(input_dir):
            if filename.endswith('.csv'):
                print(f" Processing: {folder}/{filename}")
                
                # 1. Load Cleaned Data
                df = pd.read_csv(os.path.join(input_dir, filename))
                
                # 2. Add Biometric Distances
                df = engineer_biometrics(df)
                
                # 3. Aggregate Frames into Video-Level Stats
                final_df = aggregate_video_data(df)
                
                # 4. Save to Engineered folder
                final_df.to_csv(os.path.join(output_dir, filename), index=False)

    print("\nDone! Engineered datasets are ready in:", OUTPUT_ROOT)