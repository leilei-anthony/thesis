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
NUM_CHANGEPOINTS = 5  # We want the top 3 most significant transitions
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

def is_frame_valid(results):
    # 1. Face is strictly required
    if results.face_landmarks is None:
        return False
    
    # 2. Pose Logic
    # If pose detection failed entirely, the frame is still valid
    if results.pose_landmarks is None:
        return True

    landmarks = results.pose_landmarks.landmark
    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]

    is_left_visible = left_shoulder.visibility > VISIBILITY_THRESHOLD
    is_right_visible = right_shoulder.visibility > VISIBILITY_THRESHOLD

    # 3. Symmetry Check
    # "If one shoulder is present, the other must also be present"
    # This XOR check returns False if they don't match (one True, one False)
    if is_left_visible != is_right_visible:
        return False
    
    # 4. Centering Check (Only applies if BOTH are visible)
    if is_left_visible and is_right_visible:
        face_x_center = results.face_landmarks.landmark[5].x
        
        # From camera perspective: 
        # Right side of image (Left Shoulder 11) > Center
        # Left side of image (Right Shoulder 12) < Center
        is_centered = (right_shoulder.x < face_x_center < left_shoulder.x)
        
        if not is_centered:
            return False

    # Returns True if:
    # a) Both shoulders are invisible (Valid)
    # b) Both shoulders are visible AND centered (Valid)
    return True

def draw_landmarks(image, results):
    img_h, img_w, _ = image.shape
    annotated_image = image.copy()

    # Draw Face Landmarks
    if results.face_landmarks:
        for idx in DLIB_68_IDXS:
            pt = results.face_landmarks.landmark[idx]
            x, y = int(pt.x * img_w), int(pt.y * img_h)
            cv2.circle(annotated_image, (x, y), 2, (0, 255, 0), -1)

    # Draw Pose Landmarks ONLY if BOTH shoulders are visible
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        # Check Shoulders (11 and 12) visibility
        left_shoulder_vis = landmarks[11].visibility > VISIBILITY_THRESHOLD
        right_shoulder_vis = landmarks[12].visibility > VISIBILITY_THRESHOLD
        
        # "All or Nothing": Only draw if both shoulders are detected
        if left_shoulder_vis and right_shoulder_vis:
            for idx in POSE_IDX:
                pt = landmarks[idx]
                # Even if shoulders are valid, individual points (like hands) 
                # might still be hidden, so we check them individually too
                if pt.visibility > VISIBILITY_THRESHOLD:
                    x, y = int(pt.x * img_w), int(pt.y * img_h)
                    cv2.circle(annotated_image, (x, y), 5, (255, 0, 0), -1)
    
    return annotated_image

def draw_au_text(image, au_data_row):
    """Helper function to draw AU values onto the image."""
    annotated_img = image.copy()
    img_h, _, _ = annotated_img.shape
    
    pos_x = 10
    pos_y = 30
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    text_color = (255, 255, 255) # White
    outline_color = (0, 0, 0)    # Black outline
    thickness = 1
    line_height = 20

    au_cols = [col for col in au_data_row.index if col.startswith('AU')]
    
    for au_name in au_cols:
        val = au_data_row[au_name]
        text_str = f"{au_name}: {val:.2f}"
        
        if pos_y > img_h - 10: break

        cv2.putText(annotated_img, text_str, (pos_x, pos_y), font, font_scale, outline_color, thickness + 2)
        cv2.putText(annotated_img, text_str, (pos_x, pos_y), font, font_scale, text_color, thickness)
        
        pos_y += line_height
        
    return annotated_img

def save_frame_data(frame, results, output_dir, file_prefix):
    """Saves clean image, passes FILE PATH to Py-Feat, draws overlays, saves annotated image."""
    
    # 1. Save Clean Image
    # We construct the full path first
    clean_filename = f"{file_prefix}.jpg"
    clean_path = os.path.join(output_dir, clean_filename)
    cv2.imwrite(clean_path, frame)

    # 2. Run Py-Feat Detection using the SAVED FILE PATH
    # This bypasses the 'numpy array has no read attribute' error completely
    try:
        detected_data = detector.detect_image(clean_path)
    except Exception as e:
        print(f"  -> Warning: Py-Feat failed on {file_prefix}: {e}")
        detected_data = pd.DataFrame() # Create empty DF so code continues

    # 3. Prepare the base annotated image
    final_annotated_img = draw_landmarks(frame, results)

    # --- DYNAMIC TOP-RIGHT PLACEMENT ---
    img_h, img_w, _ = final_annotated_img.shape
    text_str = f"Frame: {file_prefix}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2

    # Calculate the width and height of the text box
    (t_w, t_h), _ = cv2.getTextSize(text_str, font, scale, thickness)
    
    # Position: image_width - text_width - margin
    # We use 20px as a margin from the right edge
    text_x = img_w - t_w - 20
    text_y = 30 # Distance from the top

    # Draw outline for better visibility
    cv2.putText(final_annotated_img, text_str, (text_x, text_y), font, scale, (0, 0, 0), thickness + 2)
    # Draw actual text (Yellow)
    cv2.putText(final_annotated_img, text_str, (text_x, text_y), font, scale, (0, 255, 255), thickness)


    # 4. If AUs were detected, draw them and save CSV
    if not detected_data.empty:
        # Get row 0 (the first face found)
        au_row = detected_data.iloc[0]
        final_annotated_img = draw_au_text(final_annotated_img, au_row)

        au_columns = [col for col in detected_data.columns if col.startswith('AU')]
        detected_data[au_columns].to_csv(os.path.join(output_dir, f"{file_prefix}_aus.csv"), index=False)
    
    # 5. Save the final annotated image
    cv2.imwrite(os.path.join(output_dir, f"{file_prefix}_landmarks.jpg"), final_annotated_img)

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

def extract_changepoint_frames(video_path, output_dir):
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
                    extract_changepoint_frames(full_video_path, target_output_dir)
                except Exception as e:
                    print(f"CRITICAL ERROR processing {filename}: {e}")

if __name__ == "__main__":
    INPUT_ROOT = "Train"
    OUTPUT_ROOT = str(NUM_CHANGEPOINTS) + "_Changepoint_Dataset"
    
    print(f"Starting batch process from: {INPUT_ROOT}")
    process_batch(INPUT_ROOT, OUTPUT_ROOT)
    print("Batch processing complete.")