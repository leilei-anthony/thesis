import cv2
import sys
import os
import pandas as pd
import numpy as np
import subprocess
import shutil
from pathlib import Path
from scipy.signal import find_peaks

# Robust MediaPipe imports
try:
    from mediapipe.solutions import holistic as mp_holistic
    from mediapipe.solutions import drawing_utils as mp_drawing
except ImportError:
    import mediapipe.python.solutions.holistic as mp_holistic
    import mediapipe.python.solutions.drawing_utils as mp_drawing

class VideoFeatureExtractor:
    def __init__(self, openface_bin_path="FeatureExtraction", output_root="output", 
                 extraction_mode="targeted", num_changepoints=6):
        """
        :param openface_bin_path: Path to the OpenFace 'FeatureExtraction' executable.
        :param output_root: Directory where all results will be saved.
        :param extraction_mode: "targeted" or "changepoint"
        :param num_changepoints: Number of frames to extract in changepoint mode
        """
        self.openface_bin = openface_bin_path
        self.output_root = Path(output_root)
        self.extraction_mode = extraction_mode
        self.num_changepoints = num_changepoints
        
        # Initialize MediaPipe Holistic
        self.holistic = mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            refine_face_landmarks=True
        )
        
        # Landmarks for upper body tracking
        self.upper_body_indices = [11, 12, 13, 14, 15, 16, 23, 24]

    def is_frame_valid(self, results):
        """Checks if face is present and shoulders are visible/centered."""
        if not results.face_landmarks:
            return False
        
        if not results.pose_landmarks:
            return True # Face is enough if pose fails
            
        l_sh = results.pose_landmarks.landmark[11]
        r_sh = results.pose_landmarks.landmark[12]
        
        # Symmetry check (both shoulders should have similar visibility)
        if (l_sh.visibility > 0.5) != (r_sh.visibility > 0.5):
            return False
            
        # Centering check
        if l_sh.visibility > 0.5 and r_sh.visibility > 0.5:
            face_x = results.face_landmarks.landmark[5].x
            # Right shoulder should be to the left of face, Left shoulder to the right (camera view)
            if not (r_sh.x < face_x < l_sh.x):
                return False
        return True

    def get_landmark_velocity(self, landmark_list):
        """Calculates magnitude of facial change between frames."""
        data = np.array(landmark_list) # Shape: (N, 68, 2)
        if len(data) < 2: return np.zeros(len(data))
        
        # Normalize by centroid to ignore global head translation
        data -= np.mean(data, axis=1, keepdims=True)
        
        # Euclidean distance of all points across frame transitions
        diff = np.diff(data, axis=0)
        velocity = np.sqrt(np.sum(diff**2, axis=(1, 2)))
        
        # Smooth to avoid noise spikes
        velocity = np.convolve(velocity, np.ones(5)/5, mode='same')
        # Pad to match original length
        return np.pad(velocity, (0, 1), mode='edge')

    def process_collection(self, video_dir):
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
        videos = [f for f in os.listdir(video_dir) if f.lower().endswith(video_extensions)]
        
        if not videos:
            print(f"No videos found in {video_dir}. Please add some videos and try again.")
            return

        print(f"Found {len(videos)} videos. Starting extraction...")
        for video_file in videos:
            self.process_single_video(os.path.join(video_dir, video_file))

    def process_single_video(self, video_path):
        video_name = Path(video_path).stem
        video_output_dir = self.output_root / video_name
        raw_dir = video_output_dir / "raw_frames"
        vis_dir = video_output_dir / "visualized_frames"
        
        if raw_dir.exists(): shutil.rmtree(raw_dir)
        if vis_dir.exists(): shutil.rmtree(vis_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)
        vis_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n--- Processing: {video_name} ---")
        
        # 1. Run OpenFace
        print("Step 1/3: Running OpenFace FeatureExtraction...")
        openface_temp_dir = video_output_dir / "openface_temp"
        openface_temp_dir.mkdir(exist_ok=True)
        
        try:
            subprocess.run([
                self.openface_bin, 
                "-f", video_path, 
                "-out_dir", str(openface_temp_dir),
                "-2Dfp", "-3Dfp", "-pose", "-aus", "-gaze"
            ], check=True, capture_output=True)
        except Exception as e:
            print(f"Error running OpenFace: {e}. Check if {self.openface_bin} is correct.")
            return

        openface_csv = openface_temp_dir / f"{video_name}.csv"
        if not openface_csv.exists():
            print("OpenFace CSV not found. Skipping.")
            return
        
        df_openface = pd.read_csv(openface_csv)
        df_openface.columns = df_openface.columns.str.strip()

        # 2. Pass 1: Scan video for validity and landmarks
        print(f"Step 2/3: Scanning video (Mode: {self.extraction_mode})...")
        cap = cv2.VideoCapture(video_path)
        valid_indices = []
        valid_landmarks = []
        mediapipe_features_all = []
        
        frame_idx = 0
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break
            
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.holistic.process(image_rgb)
            
            # Extract features for CSV (always do this for all frames)
            feat = {'frame': frame_idx + 1}
            if results.pose_landmarks:
                for idx in self.upper_body_indices:
                    lm = results.pose_landmarks.landmark[idx]
                    name = mp_holistic.PoseLandmark(idx).name
                    feat[f"MP_{name}_x"] = lm.x
                    feat[f"MP_{name}_y"] = lm.y
                    feat[f"MP_{name}_z"] = lm.z
                    feat[f"MP_{name}_v"] = lm.visibility
            else:
                for idx in self.upper_body_indices:
                    name = mp_holistic.PoseLandmark(idx).name
                    feat[f"MP_{name}_x"], feat[f"MP_{name}_y"], feat[f"MP_{name}_z"] = np.nan, np.nan, np.nan
                    feat[f"MP_{name}_v"] = 0
            mediapipe_features_all.append(feat)

            # Check validity
            if self.is_frame_valid(results):
                valid_indices.append(frame_idx)
                if frame_idx < len(df_openface):
                    lms = []
                    for i in range(68):
                        lms.append([df_openface.iloc[frame_idx][f'x_{i}'], df_openface.iloc[frame_idx][f'y_{i}']])
                    valid_landmarks.append(lms)
            
            frame_idx += 1
        cap.release()

        # Select indices to extract
        selected_indices = []
        if self.extraction_mode == "targeted":
            if valid_indices:
                first = valid_indices[0]
                last = valid_indices[-1]
                mid = valid_indices[len(valid_indices)//2]
                selected_indices = sorted(list(set([first, mid, last])))
        elif self.extraction_mode == "changepoint":
            if len(valid_indices) > self.num_changepoints:
                velocity = self.get_landmark_velocity(valid_landmarks)
                peaks, _ = find_peaks(velocity, distance=15)
                # Sort peaks by velocity magnitude and pick top N
                top_peaks = peaks[np.argsort(velocity[peaks])[-self.num_changepoints:]]
                selected_indices = sorted([valid_indices[p] for p in top_peaks])
            else:
                selected_indices = valid_indices

        # 3. Pass 2: Save and Visualize selected frames
        print(f"Step 3/3: Saving {len(selected_indices)} extracted frames...")
        cap = cv2.VideoCapture(video_path)
        au_r_cols = sorted([c for c in df_openface.columns if c.startswith('AU') and c.endswith('_r')])
        au_c_cols = sorted([c for c in df_openface.columns if c.startswith('AU') and c.endswith('_c')])
        
        # Store results for the final filtered CSV
        final_features = []
        
        curr_frame = 0
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break
            
            if curr_frame in selected_indices:
                # Reuse MediaPipe results from Pass 1 (stored in mediapipe_features_all)
                # But we need the 'results' object for drawing, so we re-run only for selected frames
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.holistic.process(image_rgb)
                
                # Save Raw
                cv2.imwrite(str(raw_dir / f"frame_{curr_frame:06d}.jpg"), frame)

                # Draw Visuals
                vis_frame = frame.copy()
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(vis_frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

                if curr_frame < len(df_openface):
                    row = df_openface.iloc[curr_frame]
                    
                    # Add to final features list (OpenFace + MediaPipe)
                    combined_row = row.to_dict()
                    combined_row.update(mediapipe_features_all[curr_frame])
                    final_features.append(combined_row)

                    for i in range(68):
                        cv2.circle(vis_frame, (int(row[f'x_{i}']), int(row[f'y_{i}'])), 1, (0, 255, 0), -1)
                    
                    # Gaze
                    lx, ly = int(row['x_36']), int(row['y_36'])
                    rx, ry = int(row['x_45']), int(row['y_45'])
                    cv2.line(vis_frame, (lx, ly), (int(lx + row['gaze_0_x'] * 60), int(ly + row['gaze_0_y'] * 60)), (255, 0, 0), 2)
                    cv2.line(vis_frame, (rx, ry), (int(rx + row['gaze_1_x'] * 60), int(ry + row['gaze_1_y'] * 60)), (255, 0, 0), 2)

                    # FAU Overlay
                    y_offset = 30
                    cv2.putText(vis_frame, "AUs: Intensity (Yellow) | Presence (Blue)", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    
                    curr_y = y_offset + 25
                    for au in au_r_cols:
                        val = row[au]
                        cv2.putText(vis_frame, f"{au}: {val:.2f}", (15, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                        curr_y += 18
                    
                    curr_y = y_offset + 25
                    for au in au_c_cols:
                        val = row[au]
                        cv2.putText(vis_frame, f"{au}: {int(val)}", (180, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                        curr_y += 18

                cv2.imwrite(str(vis_dir / f"frame_{curr_frame:06d}_vis.jpg"), vis_frame)
            
            curr_frame += 1
        cap.release()

        # Final CSV Merge (Only selected frames)
        df_final = pd.DataFrame(final_features)
        df_final.to_csv(video_output_dir / f"{video_name}_selected_features.csv", index=False)
        shutil.rmtree(openface_temp_dir)
        print(f"Success: {video_name} processed ({len(selected_indices)} frames extracted).")

if __name__ == "__main__":
    # --- CONFIGURATION ---
    OPENFACE_PATH = "C:\\OpenFace\\FeatureExtraction.exe"
    INPUT_DIR = "..\\Train\\110002\\1100021015"
    OUTPUT_DIR = "./output"
    
    # Extraction Mode: "targeted", or "changepoint"
    EXTRACTION_MODE = "changepoint"
    NUM_CHANGEPOINTS = 6 # Only used in "changepoint" mode
    # ---------------------
    
    extractor = VideoFeatureExtractor(
        openface_bin_path=OPENFACE_PATH, 
        output_root=OUTPUT_DIR,
        extraction_mode=EXTRACTION_MODE,
        num_changepoints=NUM_CHANGEPOINTS
    )
    
    if not os.path.exists(INPUT_DIR):
        os.makedirs(INPUT_DIR)
        print(f"Created input directory. Please put your videos in: {INPUT_DIR}")
    else:
        extractor.process_collection(INPUT_DIR)
