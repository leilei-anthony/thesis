import cv2
import sys
import os
import pandas as pd
import numpy as np
import subprocess
import shutil
from pathlib import Path

# Robust MediaPipe imports
try:
    from mediapipe.solutions import holistic as mp_holistic
    from mediapipe.solutions import drawing_utils as mp_drawing
except ImportError:
    import mediapipe.python.solutions.holistic as mp_holistic
    import mediapipe.python.solutions.drawing_utils as mp_drawing

class VideoFeatureExtractor:
    def __init__(self, openface_bin_path="FeatureExtraction", output_root="output"):
        """
        :param openface_bin_path: Path to the OpenFace 'FeatureExtraction' executable.
        :param output_root: Directory where all results will be saved.
        """
        self.openface_bin = openface_bin_path
        self.output_root = Path(output_root)
        
        # Initialize MediaPipe Holistic
        self.holistic = mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            refine_face_landmarks=True
        )
        
        # Landmarks for upper body tracking
        self.upper_body_indices = [11, 12, 13, 14, 15, 16, 23, 24]

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

        # 2. Run MediaPipe & Generate Visuals
        print("Step 2/3: Running MediaPipe & Generating Visuals...")
        cap = cv2.VideoCapture(video_path)
        mediapipe_data = []
        frame_idx = 0
        
        # Identify AU columns for visualization
        au_r_cols = sorted([c for c in df_openface.columns if c.startswith('AU') and c.endswith('_r')])
        au_c_cols = sorted([c for c in df_openface.columns if c.startswith('AU') and c.endswith('_c')])

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            # MediaPipe
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.holistic.process(image_rgb)
            
            upper_body_features = {'frame': frame_idx + 1}
            if results.pose_landmarks:
                for idx in self.upper_body_indices:
                    lm = results.pose_landmarks.landmark[idx]
                    name = mp_holistic.PoseLandmark(idx).name
                    upper_body_features[f"MP_{name}_x"] = lm.x
                    upper_body_features[f"MP_{name}_y"] = lm.y
                    upper_body_features[f"MP_{name}_z"] = lm.z
                    upper_body_features[f"MP_{name}_v"] = lm.visibility
            else:
                for idx in self.upper_body_indices:
                    name = mp_holistic.PoseLandmark(idx).name
                    upper_body_features[f"MP_{name}_x"] = np.nan
                    upper_body_features[f"MP_{name}_y"] = np.nan
                    upper_body_features[f"MP_{name}_z"] = np.nan
                    upper_body_features[f"MP_{name}_v"] = 0
            
            mediapipe_data.append(upper_body_features)

            # Save Raw
            cv2.imwrite(str(raw_dir / f"frame_{frame_idx:06d}.jpg"), frame)

            # Draw Visuals
            vis_frame = frame.copy()
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(vis_frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

            if frame_idx < len(df_openface):
                row = df_openface.iloc[frame_idx]
                for i in range(68):
                    cv2.circle(vis_frame, (int(row[f'x_{i}']), int(row[f'y_{i}'])), 1, (0, 255, 0), -1)
                
                # Gaze
                lx, ly = int(row['x_36']), int(row['y_36'])
                rx, ry = int(row['x_45']), int(row['y_45'])
                cv2.line(vis_frame, (lx, ly), (int(lx + row['gaze_0_x'] * 60), int(ly + row['gaze_0_y'] * 60)), (255, 0, 0), 2)
                cv2.line(vis_frame, (rx, ry), (int(rx + row['gaze_1_x'] * 60), int(ry + row['gaze_1_y'] * 60)), (255, 0, 0), 2)

                # FAU Overlay (Intensity in Yellow, Presence in Blue)
                y_offset = 30
                cv2.putText(vis_frame, "AUs: Intensity (Yellow) | Presence (Blue)", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Draw Intensity (_r) in Column 1
                curr_y = y_offset + 25
                for au in au_r_cols:
                    val = row[au]
                    label = f"{au}: {val:.2f}"
                    cv2.putText(vis_frame, label, (15, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                    curr_y += 18
                
                # Draw Presence (_c) in Column 2
                curr_y = y_offset + 25
                for au in au_c_cols:
                    val = row[au]
                    label = f"{au}: {int(val)}"
                    cv2.putText(vis_frame, label, (180, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                    curr_y += 18

            cv2.imwrite(str(vis_dir / f"frame_{frame_idx:06d}_vis.jpg"), vis_frame)
            frame_idx += 1

        cap.release()

        # 3. Merge & Cleanup
        print("Step 3/3: Merging data and cleaning up...")
        df_mediapipe = pd.DataFrame(mediapipe_data)
        df_final = pd.merge(df_openface, df_mediapipe, on='frame', how='left')
        df_final.to_csv(video_output_dir / f"{video_name}_combined_features.csv", index=False)
        shutil.rmtree(openface_temp_dir)
        print(f"Success: {video_name} processed.")

if __name__ == "__main__":
    # CONFIGURATION
    OPENFACE_PATH = "C:\\OpenFace\\FeatureExtraction.exe"
    INPUT_DIR = "..\\Train\\110002\\1100022014"
    OUTPUT_DIR = "./output"
    
    extractor = VideoFeatureExtractor(openface_bin_path=OPENFACE_PATH, output_root=OUTPUT_DIR)
    
    if not os.path.exists(INPUT_DIR):
        os.makedirs(INPUT_DIR)
        print(f"Created input directory. Please put your videos in: {INPUT_DIR}")
    else:
        extractor.process_collection(INPUT_DIR)
