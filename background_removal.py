import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from rembg import remove, new_session
import onnxruntime as ort
import os
import time

def add_labeled_header(img, text, avg_ms):
    """Adds a header with the model name and average processing time."""
    h, w = img.shape[:2]
    header_h = 80
    header = np.zeros((header_h, w, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_DUPLEX
    
    # Model Name
    cv2.putText(header, text, (10, 30), font, 0.7, (255, 255, 255), 1)
    # Average Time
    cv2.putText(header, f"{avg_ms:.2f} ms", (10, 60), font, 0.6, (0, 255, 0), 1)
    
    return np.vstack((header, img))

def benchmark_and_show(image_path):
    # Load and prepare images
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_bgr.shape[:2]
    iterations = 1

    # --- SETUP MODELS ---
    # MediaPipe
    model_path = 'selfie_segmenter.tflite'
    base_options = python.BaseOptions(model_asset_path=model_path)
    mp_options = vision.ImageSegmenterOptions(base_options=base_options, output_category_mask=True)
    mp_segmenter = vision.ImageSegmenter.create_from_options(mp_options)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

    # Rembg
    rembg_session = new_session("u2net_human_seg")

    # Raw ONNX (Using the file downloaded by rembg)
    onnx_path = os.path.expanduser("~/.u2net/u2net_human_seg.onnx")
    ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

    # --- BENCHMARK: MediaPipe ---
    start = time.time()
    for _ in range(iterations):
        res = mp_segmenter.segment(mp_image)
        mask = np.squeeze(res.category_mask.numpy_view())
        mp_out = np.where(~(mask > 0.5)[:, :, np.newaxis], img_bgr, 0).astype(np.uint8)
    mp_avg = ((time.time() - start) / iterations) * 1000

    # --- BENCHMARK: Rembg ---
    start = time.time()
    for _ in range(iterations):
        res_rgba = np.array(remove(
            img_rgb, 
            session=rembg_session,
            alpha_matting=True,
            alpha_matting_foreground_threshold=240, # Higher = stricter foreground
            alpha_matting_background_threshold=10,  # Lower = stricter background
            alpha_matting_erode_size=10             # Erodes the edges of the mask
        ))
        res_bgr = cv2.cvtColor(res_rgba[:, :, :3], cv2.COLOR_RGB2BGR)
        alpha = (res_rgba[:, :, 3] / 255.0)[:, :, np.newaxis]
        rembg_out = (res_bgr * alpha).astype(np.uint8)
    rem_avg = ((time.time() - start) / iterations) * 1000

    # --- BENCHMARK: Raw ONNX ---
    start = time.time()
    for _ in range(iterations):
        input_data = cv2.resize(img_rgb, (320, 320)).astype(np.float32) / 255.0
        input_data = np.transpose(input_data, (2, 0, 1))[np.newaxis, :]
        onnx_mask_raw = ort_session.run(None, {ort_session.get_inputs()[0].name: input_data})[0]
        onnx_mask = cv2.resize(np.squeeze(onnx_mask_raw[0, 0]), (w, h))[:, :, np.newaxis]
        onnx_out = (img_bgr * onnx_mask).astype(np.uint8)
    onx_avg = ((time.time() - start) / iterations) * 1000

    # --- Visualization ---
    # Create final labeled versions
    vis_orig = add_labeled_header(img_bgr, "ORIGINAL", 0.0)
    vis_mp = add_labeled_header(mp_out, "MEDIAPIPE", mp_avg)
    vis_rem = add_labeled_header(rembg_out, "REMBG", rem_avg)
    vis_onx = add_labeled_header(onnx_out, "RAW ONNX", onx_avg)

    # Combine images horizontally
    combined = np.hstack((vis_orig, vis_mp, vis_rem, vis_onx))
    
    # Scale for display
    screen_w = 1600
    display_h = int(combined.shape[0] * (screen_w / combined.shape[1]))
    cv2.imshow("Background Removal Comparison", cv2.resize(combined, (screen_w, display_h)))
    
    print(f"\nAverage Performance (over {iterations} runs):")
    print(f"MediaPipe: {mp_avg:.2f} ms")
    print(f"Rembg:     {rem_avg:.2f} ms")
    print(f"Raw ONNX:  {onx_avg:.2f} ms")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    mp_segmenter.close()

if __name__ == "__main__":
    test_path = r'6_Changepoint_Dataset\110001\1100011019\1100011019\changepoint_1_frame_2.jpg'
    benchmark_and_show(test_path)