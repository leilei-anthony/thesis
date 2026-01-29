import scipy.integrate
if not hasattr(scipy.integrate, 'simps'):
    scipy.integrate.simps = scipy.integrate.simpson

import cv2
import mediapipe as mp
import numpy as np
import os
import pandas as pd
from feat import Detector 
import mediapipe.python.solutions.holistic as mp_holistic
from scipy.signal import find_peaks

# ---------------------------------------------------------
# CONFIGURATION & CONSTANTS
# ---------------------------------------------------------
VISIBILITY_THRESHOLD = 0.5 
NUM_CHANGEPOINTS = 3  # We want the top 3 most significant transitions
PEAK_DISTANCE = 15     # Minimum frames between selected points

DLIB_68_IDXS = [
    127, 234, 93, 132, 58, 172, 150, 176, 152, 400, 379, 378, 365, 397, 288, 361, 323, 454,
    70, 63, 105, 66, 107, 336, 296, 334, 293, 300, 383, 353, 372, 340, 346, 280, 352,
    33, 160, 158, 133, 153, 144,
    362, 385, 387, 263, 373, 380,
    168, 6, 197, 195, 5, 4, 75, 97, 2, 326, 305, 294, 278, 331, 279, 429, 358,
    61, 40, 37, 0, 267, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146,
    78, 81, 13, 311, 402, 14, 178, 87, 317
]
POSE_IDX = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]

detector = Detector(face_model="retinaface", au_model="xgb", emotion_model="resmasknet")

# --- Keep your original helper functions (is_frame_valid, draw_landmarks, etc.) ---
# [Include your original is_frame_valid, draw_landmarks, draw_au_text, and save_frame_data here]

def get_landmark_velocity(landmark_list):
    """Calculates the magnitude of facial change between frames."""
    data = np.array(landmark_list)
    # Normalize by centroid to ignore global head translation
    data -= np.mean(data, axis=1, keepdims=True)
    # Euclidean distance of all points across frame transitions
    diff = np.diff(data, axis=0)
    velocity = np.sqrt(np.sum(diff**2, axis=(1, 2)))
    # Smooth to avoid noise spikes
    return np.convolve(velocity, np.ones(5)/5, mode='same')

def extract_targeted_frames(video_path, output_dir):
    """Modified to use Changepoint Detection instead of temporal sampling."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return

    raw_frames = []
    holistic_results = []
    landmark_coords = []

    # Pass 1: Extract landmarks from all valid frames
    with mp_holistic.Holistic(static_image_mode=False, model_complexity=0) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Use your original validity logic
            if is_frame_valid(results):
                # Store (x,y,z) for all face landmarks to calculate change
                coords = np.array([[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark])
                landmark_coords.append(coords)
                raw_frames.append(frame)
                holistic_results.append(results)

    cap.release()

    if len(landmark_coords) < 10:
        print(f"Skipping {os.path.basename(video_path)} - Not enough valid frames.")
        return

    # Pass 2: Detect Changepoints (Peaks in movement velocity)
    velocity = get_landmark_velocity(landmark_coords)
    peaks, _ = find_peaks(velocity, distance=PEAK_DISTANCE)
    
    # Sort peaks by highest velocity and pick top N
    top_peak_indices = peaks[np.argsort(velocity[peaks])[-NUM_CHANGEPOINTS:]]
    top_peak_indices.sort() # Ensure chronological order

    # Pass 3: Save only the changepoint frames using your original save_frame_data
    for i, idx in enumerate(top_peak_indices):
        save_frame_data(
            raw_frames[idx], 
            holistic_results[idx], 
            output_dir, 
            f"changepoint_{i+1}_frame_{idx}"
        )

# --- Keep your original process_batch and main block ---

def process_batch(input_root, output_root):
    video_extensions = ('.avi', '.mp4', '.mov')
    for dirpath, dirnames, filenames in os.walk(input_root):
        for filename in filenames:
            if filename.lower().endswith(video_extensions):
                full_video_path = os.path.join(dirpath, filename)
                relative_path = os.path.relpath(dirpath, input_root)
                target_output_dir = os.path.join(output_root, relative_path, os.path.splitext(filename)[0])
                
                print(f"Processing: {filename}")
                try:
                    extract_targeted_frames(full_video_path, target_output_dir)
                except Exception as e:
                    print(f"CRITICAL ERROR processing {filename}: {e}")

if __name__ == "__main__":
    INPUT_ROOT = "Train"
    OUTPUT_ROOT = "Changepoint_Dataset"
    
    print(f"Starting batch process from: {INPUT_ROOT}")
    process_batch(INPUT_ROOT, OUTPUT_ROOT)
    print("Batch processing complete.")