import cv2
import mediapipe as mp
import numpy as np
import os
import mediapipe.python.solutions.holistic as mp_holistic

# ---------------------------------------------------------
# CONFIGURATION & CONSTANTS
# ---------------------------------------------------------

VISIBILITY_THRESHOLD = 0.5 

# Mapping: MediaPipe 468 -> dlib-style 68 face landmarks
DLIB_68_IDXS = [
    127, 234, 93, 132, 58, 172, 150, 176, 152, 400, 379, 378, 365, 397, 288, 361, 323, 454,
    70, 63, 105, 66, 107, 336, 296, 334, 293, 300, 383, 353, 372, 340, 346, 280, 352,
    33, 160, 158, 133, 153, 144,
    362, 385, 387, 263, 373, 380,
    168, 6, 197, 195, 5, 4, 75, 97, 2, 326, 305, 294, 278, 331, 279, 429, 358,
    61, 40, 37, 0, 267, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146,
    78, 81, 13, 311, 402, 14, 178, 87, 317
]

# Pose landmarks (Shoulders and Elbows only)
POSE_IDX = [11, 12, 13, 14]

def is_frame_valid(results):
    if results.face_landmarks is None:
        return False
    if results.pose_landmarks is None:
        return False

    landmarks = results.pose_landmarks.landmark
    for idx in POSE_IDX:
        if landmarks[idx].visibility < VISIBILITY_THRESHOLD:
            return False
    return True

def draw_landmarks(image, results):
    img_h, img_w, _ = image.shape
    annotated_image = image.copy()

    # Draw Face Landmarks
    if results.face_landmarks:
        for idx in DLIB_68_IDXS:
            pt = results.face_landmarks.landmark[idx]
            x = int(pt.x * img_w)
            y = int(pt.y * img_h)
            cv2.circle(annotated_image, (x, y), 2, (0, 255, 0), -1)

    # Draw Pose Landmarks
    if results.pose_landmarks:
        for idx in POSE_IDX:
            pt = results.pose_landmarks.landmark[idx]
            if pt.visibility > VISIBILITY_THRESHOLD:
                x = int(pt.x * img_w)
                y = int(pt.y * img_h)
                cv2.circle(annotated_image, (x, y), 5, (255, 0, 0), -1)
                cv2.putText(annotated_image, str(idx), (x+5, y-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return annotated_image

def save_frame_pair(frame, results, output_dir, file_prefix):
    # Save Clean
    cv2.imwrite(os.path.join(output_dir, f"{file_prefix}.jpg"), frame)
    # Save Landmarked
    landmarked_img = draw_landmarks(frame, results)
    cv2.imwrite(os.path.join(output_dir, f"{file_prefix}_landmarks.jpg"), landmarked_img)

def extract_targeted_frames(video_path, output_dir):
    # Create the specific subdirectory for this video clip
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Skipping: Could not open {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing: {os.path.basename(video_path)} | Frames: {total_frames}")

    start_valid_idx = None
    end_valid_idx = None

    with mp_holistic.Holistic(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:

        # 1. Find First Valid Frame
        for i in range(total_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret: break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image_rgb)

            if is_frame_valid(results):
                start_valid_idx = i
                save_frame_pair(frame, results, output_dir, "frame_1_start")
                break
        
        if start_valid_idx is None:
            print(f"  -> Failed: No valid start frame found for {os.path.basename(video_path)}")
            cap.release()
            return

        # 2. Find Last Valid Frame
        for i in range(total_frames - 1, start_valid_idx, -1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret: break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image_rgb)

            if is_frame_valid(results):
                end_valid_idx = i
                save_frame_pair(frame, results, output_dir, "frame_3_end")
                break

        if end_valid_idx is None:
            end_valid_idx = start_valid_idx
            # Re-read start frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, end_valid_idx)
            ret, frame = cap.read()
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image_rgb)
            save_frame_pair(frame, results, output_dir, "frame_3_end")

        # 3. Find Midpoint Frame
        mid_idx = (start_valid_idx + end_valid_idx) // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_idx)
        ret, frame = cap.read()
        if ret:
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image_rgb)
            if results.face_landmarks and results.pose_landmarks:
                 save_frame_pair(frame, results, output_dir, "frame_2_mid")
            else:
                cv2.imwrite(os.path.join(output_dir, "frame_2_mid.jpg"), frame)
        
    cap.release()

def process_batch(input_root, output_root):
    """
    Crawls through input_root, finds .avi/.mp4 files, replicates structure in output_root
    """
    video_extensions = ('.avi', '.mp4', '.mov')
    
    # Walk through the directory tree
    for dirpath, dirnames, filenames in os.walk(input_root):
        for filename in filenames:
            if filename.lower().endswith(video_extensions):
                
                # Full path to the input video
                full_video_path = os.path.join(dirpath, filename)
                
                # Calculate relative path (e.g., "110001\1100011002")
                relative_path = os.path.relpath(dirpath, input_root)
                
                # Construct corresponding output directory
                target_output_dir = os.path.join(output_root, relative_path, os.path.splitext(filename)[0])
                
                try:
                    extract_targeted_frames(full_video_path, target_output_dir)
                except Exception as e:
                    print(f"CRITICAL ERROR processing {filename}: {e}")

# ---------------------------------------------------------
# EXECUTION
# ---------------------------------------------------------
if __name__ == "__main__":
    # Point this to your main Train folder
    INPUT_ROOT = "Train"  
    
    # Where you want the result folders to be created
    OUTPUT_ROOT = "Processed_Dataset"
    
    print(f"Starting batch process from: {INPUT_ROOT}")
    process_batch(INPUT_ROOT, OUTPUT_ROOT)
    print("Batch processing complete.")