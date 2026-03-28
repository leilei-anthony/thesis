import cv2
import sys
import os
import pandas as pd
import numpy as np
import subprocess
import shutil
import multiprocessing
import gc
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
    def __init__(self, openface_bin, output_root, extraction_mode, 
                 num_changepoints, frame_skip, fast_rembg, mp_complexity, labels_df):
        """
        Configuration only. We DO NOT load AI models here to ensure 
        multiprocessing works without memory serialization/VRAM allocation errors.
        """
        self.openface_bin = openface_bin
        self.output_root = Path(output_root)
        self.extraction_mode = extraction_mode
        self.num_changepoints = num_changepoints
        
        # Performance configurations
        self.frame_skip = frame_skip
        self.fast_rembg = fast_rembg
        self.mp_complexity = mp_complexity
        
        self.labels_df = labels_df
        self.upper_body_indices =[11, 12, 13, 14, 15, 16, 23, 24]

    def init_models(self):
        """Initializes AI models locally inside the spawned CPU process."""
        self.holistic = mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=self.mp_complexity, 
            enable_segmentation=False,
            refine_face_landmarks=True
        )
        
        self.rembg_session = new_session(
            "u2net_human_seg", 
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )

    def apply_rembg(self, frame_bgr):
        """Applies fast GPU background removal."""
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        kwargs = {"session": self.rembg_session}
        if not self.fast_rembg:
            kwargs.update({
                "alpha_matting": True,
                "alpha_matting_foreground_threshold": 240,
                "alpha_matting_background_threshold": 10,
                "alpha_matting_erode_size": 10
            })
            
        res_rgba = np.array(remove(img_rgb, **kwargs))
        res_bgr = cv2.cvtColor(res_rgba[:, :, :3], cv2.COLOR_RGB2BGR)
        alpha = (res_rgba[:, :, 3] / 255.0)[:, :, np.newaxis]
        return (res_bgr * alpha).astype(np.uint8)

    def is_frame_valid(self, results):
        if not results.face_landmarks:
            return False
        if not results.pose_landmarks:
            return True 
            
        l_sh = results.pose_landmarks.landmark[11]
        r_sh = results.pose_landmarks.landmark[12]
        
        if (l_sh.visibility > 0.5) != (r_sh.visibility > 0.5):
            return False
            
        if l_sh.visibility > 0.5 and r_sh.visibility > 0.5:
            face_x = results.face_landmarks.landmark[5].x
            if not (r_sh.x < face_x < l_sh.x):
                return False
        return True

    def get_landmark_velocity(self, landmark_list):
        data = np.array(landmark_list) 
        if len(data) < 2: return np.zeros(len(data))
        data -= np.mean(data, axis=1, keepdims=True)
        diff = np.diff(data, axis=0)
        velocity = np.sqrt(np.sum(diff**2, axis=(1, 2)))
        velocity = np.convolve(velocity, np.ones(5)/5, mode='same')
        return np.pad(velocity, (0, 1), mode='edge')

    def process_single_video(self, video_path, relative_dir=""):
        # Load AI models if this process hasn't loaded them yet
        if not hasattr(self, 'holistic'):
            self.init_models()

        video_filename = Path(video_path).name
        video_name = Path(video_path).stem
        
        # Mirror the input directory structure to prevent identical filename clashes
        video_output_dir = self.output_root / relative_dir / video_name
        raw_dir = video_output_dir / "raw_frames"
        vis_dir = video_output_dir / "visualized_frames"
        
        if raw_dir.exists(): shutil.rmtree(raw_dir)
        if vis_dir.exists(): shutil.rmtree(vis_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)
        vis_dir.mkdir(parents=True, exist_ok=True)

        # 1. Pass 1: Scan video
        cap = cv2.VideoCapture(video_path)
        valid_indices = []
        valid_landmarks =[]
        mediapipe_features_all = {} 
        
        frame_idx = 0
        while cap.isOpened():
            # Zero-cost frame skipping
            if frame_idx % self.frame_skip != 0:
                has_frame = cap.grab()
                frame = None
            else:
                has_frame, frame = cap.read()
                
            if not has_frame: 
                break
                
            if frame is not None:
                # Downscale for faster MediaPipe processing
                h, w = frame.shape[:2]
                max_dim = 640
                if max(h, w) > max_dim:
                    scale = max_dim / max(h, w)
                    process_frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
                else:
                    process_frame = frame
                    
                image_rgb = cv2.cvtColor(process_frame, cv2.COLOR_BGR2RGB)
                results = self.holistic.process(image_rgb)
                
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
                        
                mediapipe_features_all[frame_idx] = feat

                if self.is_frame_valid(results):
                    valid_indices.append(frame_idx)
                    if results.face_landmarks:
                        lms = [[lm.x, lm.y] for lm in results.face_landmarks.landmark]
                        valid_landmarks.append(lms)
            
            frame_idx += 1
        cap.release()

        # Selection logic
        selected_indices =[]
        if self.extraction_mode == "targeted":
            if valid_indices:
                first = valid_indices[0]
                last = valid_indices[-1]
                mid = valid_indices[len(valid_indices)//2]
                selected_indices = sorted(list(set([first, mid, last])))
        elif self.extraction_mode == "changepoint":
            if len(valid_indices) > self.num_changepoints:
                velocity = self.get_landmark_velocity(valid_landmarks)
                distance = max(1, 15 // self.frame_skip)
                peaks, _ = find_peaks(velocity, distance=distance)
                top_peaks = peaks[np.argsort(velocity[peaks])[-self.num_changepoints:]]
                selected_indices = sorted([valid_indices[p] for p in top_peaks])
            else:
                selected_indices = valid_indices

        if not selected_indices:
            # Clean up empty directories if skipped
            shutil.rmtree(raw_dir, ignore_errors=True)
            shutil.rmtree(vis_dir, ignore_errors=True)
            return None 

        # 2. Pass 2: Extract selected frames in O(1) time
        cap = cv2.VideoCapture(video_path)
        for idx in selected_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            success, frame = cap.read()
            if success:
                cv2.imwrite(str(raw_dir / f"frame_{idx:06d}.jpg"), frame)
        cap.release()

        # 3. Run OpenFace
        openface_temp_dir = video_output_dir / "openface_temp"
        openface_temp_dir.mkdir(exist_ok=True)
        
        try:
            # Added a 45-second timeout to prevent corrupted videos from freezing the script
            subprocess.run([
                self.openface_bin, 
                "-fdir", str(raw_dir), 
                "-out_dir", str(openface_temp_dir),
                "-2Dfp", "-3Dfp", "-pose", "-aus", "-gaze"
            ], check=True, capture_output=True, timeout=45)
        except subprocess.TimeoutExpired:
            print(f"  -> OpenFace timed out on {video_name}")
            return None
        except Exception as e:
            print(f"  -> Error running OpenFace on {video_name}: {e}")
            return None

        openface_csv = openface_temp_dir / "raw_frames.csv"
        if not openface_csv.exists():
            return None
        
        df_openface = pd.read_csv(openface_csv)
        df_openface.columns = df_openface.columns.str.strip()
        if 'timestamp' in df_openface.columns:
            df_openface = df_openface.drop(columns=['timestamp'])

        # 4. Final Visualizations
        au_r_cols = sorted([c for c in df_openface.columns if c.startswith('AU') and c.endswith('_r')])
        au_c_cols = sorted([c for c in df_openface.columns if c.startswith('AU') and c.endswith('_c')])
        name_col = next((c for c in ['name', 'filename', 'file'] if c in df_openface.columns), None)

        final_features =[]
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
            
            vis_frame = self.apply_rembg(clean_frame)
            image_rgb = cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB)
            results = self.holistic.process(image_rgb)
            
            combined_row = {'frame_id': i}
            if self.labels_df is not None:
                video_labels = self.labels_df[self.labels_df['ClipID'] == video_filename]
                if not video_labels.empty:
                    for label_col in['Boredom', 'Engagement', 'Confusion', 'Frustration']:
                        if label_col in video_labels.columns:
                            combined_row[label_col] = video_labels.iloc[0][label_col]
                else:
                    for label_col in['Boredom', 'Engagement', 'Confusion', 'Frustration']:
                        combined_row[label_col] = np.nan

            combined_row.update(row.to_dict())
            if frame_num in mediapipe_features_all:
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
                cv2.putText(vis_frame, f"{au}: {row[au]:.2f}", (15, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                curr_y += 18
            
            curr_y = y_offset + 25
            for au in au_c_cols:
                cv2.putText(vis_frame, f"{au}: {int(row[au])}", (180, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                curr_y += 18

            cv2.imwrite(str(vis_dir / f"frame_{frame_num:06d}_vis.jpg"), vis_frame)

        df_final = pd.DataFrame(final_features)
        df_final.to_csv(video_output_dir / f"{video_name}_selected_features.csv", index=False)
        shutil.rmtree(openface_temp_dir, ignore_errors=True)
        return df_final


# --- TOP LEVEL WORKER FUNCTION FOR MULTIPROCESSING ---
def worker_function(video_path, config, input_dir):
    try:
        extractor = VideoFeatureExtractor(
            openface_bin=config['openface_bin'],
            output_root=config['output_root'],
            extraction_mode=config['extraction_mode'],
            num_changepoints=config['num_changepoints'],
            frame_skip=config['frame_skip'],
            fast_rembg=config['fast_rembg'],
            mp_complexity=config['mp_complexity'],
            labels_df=config['labels_df']
        )
        
        # Determine relative path structure
        rel_path = os.path.relpath(video_path, input_dir)
        relative_dir = Path(rel_path).parent
        
        df = extractor.process_single_video(video_path, relative_dir=relative_dir)
        
        if df is not None and not df.empty:
            person_id = str(Path(rel_path).parts[0])[:6] 
            df.insert(0, 'person_id', person_id)
            df.insert(1, 'source_video_path', rel_path)
            return df
    except Exception as e:
        print(f"Error processing {Path(video_path).name}: {e}")
    return None

def process_task_wrapper(args):
    """Wrapper to unpack arguments and force memory cleanup."""
    vp, config, in_dir = args
    df = worker_function(vp, config, in_dir)
    
    # Force Python to clear leftover RAM immediately after every single video
    gc.collect()
    
    return vp, df

if __name__ == "__main__":
    # REQUIRED FOR WINDOWS MULTIPROCESSING
    multiprocessing.freeze_support()
    
    # --- CONFIGURATION ---
    OPENFACE_PATH = "C:\\OpenFace\\FeatureExtraction.exe"
    INPUT_DIR = "C:\\Users\\PC\\Desktop\\DAiSEE\\DataSet\\Validation"
    LABELS_CSV = "AllLabels.csv"
    
    EXTRACTION_MODE = "changepoint"  # "targeted" or "changepoint"
    NUM_CHANGEPOINTS = 6 
    
    # --- PERFORMANCE & PARALLEL CONFIG ---
    FRAME_SKIP = 2         
    FAST_REMBG = True      
    MP_COMPLEXITY = 1      
    MAX_CONCURRENT_VIDEOS = 4 # If it crashes/freezes again, lower this to 2
    # ------------------------------------
    
    OUTPUT_DIR = f"./Datasets/{NUM_CHANGEPOINTS}_Changepoint_Dataset_Validation" if EXTRACTION_MODE == "changepoint" else "./Datasets/Targeted_Dataset_Validation"
    
    if not os.path.exists(INPUT_DIR):
        os.makedirs(INPUT_DIR)
        print(f"Directory created. Please place videos in: {INPUT_DIR}")
        sys.exit()

    # Pre-load labels
    labels_df = None
    if os.path.exists(LABELS_CSV):
        print(f"Loading labels from {LABELS_CSV}...")
        labels_df = pd.read_csv(LABELS_CSV)
        if 'ClipID' in labels_df.columns:
            labels_df['ClipID'] = labels_df['ClipID'].astype(str).str.strip()

    # Find videos
    video_paths =[]
    for root, dirs, files in os.walk(INPUT_DIR):
        for file in files:
            if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_paths.append(os.path.join(root, file))

    if not video_paths:
        print(f"No videos found in {INPUT_DIR}.")
        sys.exit()

    print(f"Found {len(video_paths)} videos. Starting parallel processing on {MAX_CONCURRENT_VIDEOS} workers...")

    # Pack config for workers
    worker_config = {
        'openface_bin': OPENFACE_PATH,
        'output_root': OUTPUT_DIR,
        'extraction_mode': EXTRACTION_MODE,
        'num_changepoints': NUM_CHANGEPOINTS,
        'frame_skip': FRAME_SKIP,
        'fast_rembg': FAST_REMBG,
        'mp_complexity': MP_COMPLEXITY,
        'labels_df': labels_df
    }

    # Bundle arguments for the pool map
    tasks =[(vp, worker_config, INPUT_DIR) for vp in video_paths]
    all_results =[]
    
    # START PARALLEL PROCESSING (WITH ANTI-MEMORY-LEAK PROTECTION)
    # maxtasksperchild=10 means: After a background worker processes 10 videos, 
    # it is automatically destroyed and cleanly restarted, flushing all VRAM/RAM.
    with multiprocessing.Pool(processes=MAX_CONCURRENT_VIDEOS, maxtasksperchild=10) as pool:
        for i, (video_path, df) in enumerate(pool.imap_unordered(process_task_wrapper, tasks)):
            try:
                if df is not None:
                    all_results.append(df)
                    print(f"[{i+1}/{len(video_paths)}] Successfully finished: {Path(video_path).name}")
                else:
                    print(f"[{i+1}/{len(video_paths)}] Skipped (No valid frames/error): {Path(video_path).name}")
            except Exception as e:
                print(f"[{i+1}/{len(video_paths)}] FAILED completely: {Path(video_path).name} - {e}")

    # Combine all DataFrames
    if all_results:
        print("\n--- Merging features into collective CSV ---")
        master_df = pd.concat(all_results, ignore_index=True)
        master_path = Path(OUTPUT_DIR) / "collective_features.csv"
        master_df.to_csv(master_path, index=False)
        print(f"Success! Master CSV saved to: {master_path}")
        print(f"Total extracted rows: {len(master_df)}")
    else:
        print("\nNo features were extracted from any videos.")