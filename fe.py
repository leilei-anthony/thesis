import cv2
import os

def extract_sme_frames(video_path, output_dir):
    """
    Extracts the first, middle, and last frames (SME) from a video.
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Get total frame count
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # Define SME indices
    # Start: 0, Middle: total//2, End: total-1
    target_indices = {
        "start": 0,
        "middle": total_frames // 2,
        "end": total_frames - 1
    }

    for label, frame_idx in target_indices.items():
        # Set the video position to the specific frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = cap.read()

        if success:
            # Save the frame as an image
            filename = f"{video_name}_{label}_frame.jpg"
            save_path = os.path.join(output_dir, filename)
            cv2.imwrite(save_path, frame)
            print(f"Saved: {save_path}")
        else:
            print(f"Failed to extract {label} frame at index {frame_idx}")

    cap.release()

# Example usage:
# extract_sme_frames("DAiSEE_sample.mp4", "extracted_frames")

extract_sme_frames("5000441001.avi", "extracted_frames\\5000441001")