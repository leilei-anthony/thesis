import cv2
import sys
import os
import pandas as pd
import numpy as np
import subprocess
import shutil
from pathlib import Path
from scipy.signal import find_peaks
from rembg import remove, new_session

# --- Windows CUDA DLL Fix ---
if os.name == 'nt':
    import site
    try:
        search_paths = set()
        try: search_paths.update(site.getsitepackages())
        except: pass
        search_paths.update(sys.path)
        
        for sp in search_paths:
            if not sp: continue
            nvidia_path = Path(sp) / "nvidia"
            if nvidia_path.exists() and nvidia_path.is_dir():
                for bin_dir in nvidia_path.glob("**/bin"):
                    if bin_dir.is_dir():
                        try:
                            os.add_dll_directory(str(bin_dir))
                            os.environ['PATH'] = str(bin_dir) + os.pathsep + os.environ['PATH']
                        except Exception:
                            pass
    except Exception:
        pass
# ----------------------------

# Robust MediaPipe imports
try:
    from mediapipe.solutions import holistic as mp_holistic
    from mediapipe.solutions import drawing_utils as mp_drawing
except ImportError:
    import mediapipe.python.solutions.holistic as mp_holistic
    import mediapipe.python.solutions.drawing_utils as mp_drawing

class VideoFeatureExtractor:
    def __init__(self, openface_bin_path="FeatureExtraction", output_root="output", 
                 extraction_mode="targeted", num_changepoints=6, labels_csv_path=None):
        """
        :param openface_bin_path: Path to the OpenFace 'FeatureExtraction' executable.
        :param output_root: Directory where all results will be saved.
        :param extraction_mode: "targeted" or "changepoint"
        :param num_changepoints: Number of frames to extract in changepoint mode
        :param labels_csv_path: Path to AllLabels.csv
        """
        self.openface_bin = openface_bin_path
        self.output_root = Path(output_root)
        self.extraction_mode = extraction_mode
        self.num_changepoints = num_changepoints
        
        # Load labels if provided
        self.labels_df = None
        if labels_csv_path and os.path.exists(labels_csv_path):
            print(f"Loading labels from {labels_csv_path}...")
            self.labels_df = pd.read_csv(labels_csv_path)
            # Ensure ClipID is string and stripped
            if 'ClipID' in self.labels_df.columns:
                self.labels_df['ClipID'] = self.labels_df['ClipID'].astype(str).str.strip()
            else:
                print("Warning: 'ClipID' column not found in labels CSV.")
        
        # Initialize MediaPipe Holistic
        self.holistic = mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            refine_face_landmarks=True
        )
        
        # Initialize Rembg session
        print("Initializing Rembg session (u2net_human_seg) with GPU support...")
        # Prioritize CUDA (GPU), fall back to CPU if unavailable
        self.rembg_session = new_session(
            "u2net_human_seg", 
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        
        # Landmarks for upper body tracking
        self.upper_body_indices = [11, 12, 13, 14, 15, 16, 23, 24]

    def apply_rembg(self, frame_bgr):
        """Applies rembg to a BGR frame and returns a BGR frame with black background."""
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        # Apply rembg with alpha matting for better edges
        res_rgba = np.array(remove(
            img_rgb, 
            session=self.rembg_session,
            alpha_matting=True,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=10,
            alpha_matting_erode_size=10
        ))
        # Convert back to BGR and apply alpha mask
        res_bgr = cv2.cvtColor(res_rgba[:, :, :3], cv2.COLOR_RGB2BGR)
        alpha = (res_rgba[:, :, 3] / 255.0)[:, :, np.newaxis]
        return (res_bgr * alpha).astype(np.uint8)

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
        video_paths = []
        
        # Recursive search using os.walk
        for root, dirs, files in os.walk(video_dir):
            for file in files:
                if file.lower().endswith(video_extensions):
                    video_paths.append(os.path.join(root, file))
        
        if not video_paths:
            print(f"No videos found in {video_dir}. Please check the path.")
            return

        print(f"Found {len(video_paths)} videos across all subfolders. Starting batch extraction...")
        
        all_results = []
        for i, video_path in enumerate(video_paths):
            print(f"\n[Video {i+1}/{len(video_paths)}]")
            df = self.process_single_video(video_path)
            
            if df is not None and not df.empty:
                # Add metadata: where did this video come from?
                rel_path = os.path.relpath(video_path, video_dir)
                # Extract first 6 characters for person_id (e.g., '110002')
                person_id = rel_path[:6]
                df.insert(0, 'person_id', person_id)
                df.insert(1, 'source_video_path', rel_path)
                all_results.append(df)
        
        if all_results:
            print("\n--- Merging all features into collective CSV ---")
            master_df = pd.concat(all_results, ignore_index=True)
            master_path = self.output_root / "collective_features.csv"
            master_df.to_csv(master_path, index=False)
            print(f"Success! Master CSV saved to: {master_path}")
            print(f"Total rows in master file: {len(master_df)}")
        else:
            print("\nNo features were extracted from any videos.")

    def print_progress(self, iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█', print_end="\r"):
        """
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            length      - Optional  : character length of bar (Int)
            fill        - Optional  : bar fill character (Str)
            print_end   - Optional  : end character (e.g. "\r", "\r\n") (Str)
        """
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filled_length = int(length * iteration // total)
        bar = fill * filled_length + '-' * (length - filled_length)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_end)
        # Print New Line on Complete
        if iteration == total:
            print()

    def process_single_video(self, video_path):
        video_filename = Path(video_path).name
        video_name = Path(video_path).stem
        video_output_dir = self.output_root / video_name
        raw_dir = video_output_dir / "raw_frames"
        vis_dir = video_output_dir / "visualized_frames"
        
        if raw_dir.exists(): shutil.rmtree(raw_dir)
        if vis_dir.exists(): shutil.rmtree(vis_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)
        vis_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n--- Processing: {video_name} ---")
        
        # 1. Pass 1: Scan video for validity and landmarks (Original Video)
        print(f"Step 1/4: Scanning video for {self.extraction_mode} selection...")
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        valid_indices = []
        valid_landmarks = []
        mediapipe_features_all = []
        
        frame_idx = 0
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break
            
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.holistic.process(image_rgb)
            
            # Extract features for CSV
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
                if results.face_landmarks:
                    lms = [[lm.x, lm.y] for lm in results.face_landmarks.landmark]
                    valid_landmarks.append(lms)
            
            frame_idx += 1
            if total_frames > 0:
                self.print_progress(frame_idx, total_frames, prefix='Scanning:', suffix='Complete', length=40)
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
                top_peaks = peaks[np.argsort(velocity[peaks])[-self.num_changepoints:]]
                selected_indices = sorted([valid_indices[p] for p in top_peaks])
            else:
                selected_indices = valid_indices

        if not selected_indices:
            print(f"No valid frames found for {video_name}. Skipping.")
            return

        # 2. Pass 2: Extract selected frames
        print(f"Step 2/4: Extracting {len(selected_indices)} original frames...")
        cap = cv2.VideoCapture(video_path)
        curr_frame = 0
        extracted_count = 0
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break
            
            if curr_frame in selected_indices:
                # Save as Raw (Original)
                cv2.imwrite(str(raw_dir / f"frame_{curr_frame:06d}.jpg"), frame)
                extracted_count += 1
                self.print_progress(extracted_count, len(selected_indices), prefix='Extracting:', suffix='Complete', length=40)
            
            curr_frame += 1
        cap.release()

        # 3. Run OpenFace on the images
        print("Step 3/4: Running OpenFace on extracted images...")
        openface_temp_dir = video_output_dir / "openface_temp"
        openface_temp_dir.mkdir(exist_ok=True)
        
        try:
            subprocess.run([
                self.openface_bin, 
                "-fdir", str(raw_dir), 
                "-out_dir", str(openface_temp_dir),
                "-2Dfp", "-3Dfp", "-pose", "-aus", "-gaze"
            ], check=True, capture_output=True)
        except Exception as e:
            print(f"Error running OpenFace: {e}.")
            return

        openface_csv = openface_temp_dir / "raw_frames.csv"
        if not openface_csv.exists():
            print("OpenFace CSV not found. Skipping.")
            return
        
        df_openface = pd.read_csv(openface_csv)
        df_openface.columns = df_openface.columns.str.strip()
        
        # Remove timestamp column as requested
        if 'timestamp' in df_openface.columns:
            df_openface = df_openface.drop(columns=['timestamp'])

        # 4. Pass 3: Visualize on CLEAN images
        print("Step 4/4: Generating final visualizations...")
        au_r_cols = sorted([c for c in df_openface.columns if c.startswith('AU') and c.endswith('_r')])
        au_c_cols = sorted([c for c in df_openface.columns if c.startswith('AU') and c.endswith('_c')])
        
        name_col = None
        for col in ['name', 'filename', 'file']:
            if col in df_openface.columns:
                name_col = col
                break
        
        if name_col is None:
            print("Warning: Could not find filename column in OpenFace CSV.")
            if len(df_openface) == len(selected_indices):
                print("Attempting fallback: matching rows by order...")
            else:
                return

        final_features = []
        total_vis = len(df_openface)
        for i, row in df_openface.iterrows():
            if name_col:
                fname = str(row[name_col]).strip()
                fname_stem = Path(fname).stem
                try:
                    frame_num = int(fname_stem.split('_')[1])
                except (IndexError, ValueError):
                    continue
            else:
                frame_num = selected_indices[i]
                fname_stem = f"frame_{frame_num:06d}"
            
            clean_path = raw_dir / f"{fname_stem}.jpg"
            if not clean_path.exists():
                clean_path = raw_dir / fname
                if not clean_path.exists(): continue
            
            clean_frame = cv2.imread(str(clean_path))
            if clean_frame is None: continue
            
            # Apply background removal for visualization ONLY
            vis_frame = self.apply_rembg(clean_frame)
            
            image_rgb = cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB)
            results = self.holistic.process(image_rgb)
            
            # Create combined row with frame_id as the first entry
            combined_row = {'frame_id': i}
            
            # Add labels if available
            if self.labels_df is not None:
                video_labels = self.labels_df[self.labels_df['ClipID'] == video_filename]
                if not video_labels.empty:
                    # Add Boredom, Engagement, Confusion, Frustration
                    for label_col in ['Boredom', 'Engagement', 'Confusion', 'Frustration']:
                        if label_col in video_labels.columns:
                            combined_row[label_col] = video_labels.iloc[0][label_col]
                else:
                    # Fill with NaN if not found
                    for label_col in ['Boredom', 'Engagement', 'Confusion', 'Frustration']:
                        combined_row[label_col] = np.nan

            combined_row.update(row.to_dict())
            
            if frame_num < len(mediapipe_features_all):
                combined_row.update(mediapipe_features_all[frame_num])
            final_features.append(combined_row)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(vis_frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

            for j in range(68):
                cv2.circle(vis_frame, (int(row[f'x_{j}']), int(row[f'y_{j}'])), 1, (0, 255, 0), -1)
            
            lx, ly = int(row['x_36']), int(row['y_36'])
            rx, ry = int(row['x_45']), int(row['y_45'])
            cv2.line(vis_frame, (lx, ly), (int(lx + row['gaze_0_x'] * 60), int(ly + row['gaze_0_y'] * 60)), (255, 0, 0), 2)
            cv2.line(vis_frame, (rx, ry), (int(rx + row['gaze_1_x'] * 60), int(ry + row['gaze_1_y'] * 60)), (255, 0, 0), 2)

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

            cv2.imwrite(str(vis_dir / f"frame_{frame_num:06d}_vis.jpg"), vis_frame)
            self.print_progress(i + 1, total_vis, prefix='Visualizing:', suffix='Complete', length=40)

        df_final = pd.DataFrame(final_features)
        df_final.to_csv(video_output_dir / f"{video_name}_selected_features.csv", index=False)
        shutil.rmtree(openface_temp_dir)
        print(f"Done: {video_name} processed.")
        return df_final

if __name__ == "__main__":
    # --- CONFIGURATION ---
    OPENFACE_PATH = "C:\\OpenFace\\FeatureExtraction.exe"
    INPUT_DIR = "Train"
    LABELS_CSV = "AllLabels.csv"
    
    # Extraction Mode: "targeted", or "changepoint"
    EXTRACTION_MODE = "changepoint"
    NUM_CHANGEPOINTS = 6 # Only used in "changepoint" mode
    # ---------------------
    
    # Calculate dynamic output directory
    if EXTRACTION_MODE == "targeted":
        OUTPUT_DIR = "./Datasets/Targeted_Dataset"
    else:
        OUTPUT_DIR = f"./Datasets/{NUM_CHANGEPOINTS}_Changepoint_Dataset"
    
    extractor = VideoFeatureExtractor(
        openface_bin_path=OPENFACE_PATH, 
        output_root=OUTPUT_DIR,
        extraction_mode=EXTRACTION_MODE,
        num_changepoints=NUM_CHANGEPOINTS,
        labels_csv_path=LABELS_CSV
    )
    
    if not os.path.exists(INPUT_DIR):
        os.makedirs(INPUT_DIR)
        print(f"Created input directory. Please put your videos in: {INPUT_DIR}")
    else:
        extractor.process_collection(INPUT_DIR)
