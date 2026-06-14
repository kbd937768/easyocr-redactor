import os
import cv2
import numpy as np
import time
import logging
from glob import glob
from paddleocr import PaddleOCR

# ==========================================
# CONFIGURATION
# ==========================================
# Silence PaddleOCR internal info logs to keep the console clean
logging.getLogger("ppocr").setLevel(logging.WARNING)

INPUT_FOLDER = "./frames"          # Folder containing your PNG frames
OUTPUT_FOLDER = "./processed"      # Folder where redacted frames will be saved
LOG_FILE_PATH = "./ocr_results.txt"# TXT file to save all detected text
LOGO_FOLDER = "./logos"            # Folder containing your reference logo images
LOGO_THRESHOLD = 0.8               # Sensitivity for logo matching (0.0 to 1.0)
WORKER_ID = 1                      # ID for console logging

KEYWORDS = [
    'stake.com', 'stake .com.', 'stake.com.', 'stake.com', 
    'https://', 'Stake', 'stakecom', 'stake', 'Stake Originals', 'Only on Stake'
]

# Initialize PaddleOCR (enable_mkldnn=False fixes the PIR oneDNN CPU crash)
ocr = PaddleOCR(use_textline_orientation=True, lang='en', enable_mkldnn=False)

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def load_logos(logo_dir):
    """Loads all reference logos from the specified directory."""
    logos = []
    if not os.path.exists(logo_dir):
        print(f"⚠️ Logo directory '{logo_dir}' not found. Skipping logo detection.")
        return logos
    
    for ext in ('*.png', '*.jpg', '*.jpeg'):
        for path in glob(os.path.join(logo_dir, ext)):
            logo = cv2.imread(path)
            if logo is not None:
                logos.append((os.path.basename(path), logo))
    return logos

def check_keywords(text, keyword_list):
    """Checks if any keyword is present in the detected text (case-insensitive)."""
    text_lower = text.lower().strip()
    for kw in keyword_list:
        if kw.lower() in text_lower:
            return True
    return False

# ==========================================
# MAIN PROCESSING LOOP
# ==========================================
def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    frame_files = sorted(glob(os.path.join(INPUT_FOLDER, "*.png")))
    logos = load_logos(LOGO_FOLDER)
    
    if not frame_files:
        print(f"❌ No PNG frames found in '{INPUT_FOLDER}'. Please check the path.")
        return

    print(f"🚀 Starting processing for {len(frame_files)} frames...")
    
    start_time = time.time()
    frames_processed = 0

    with open(LOG_FILE_PATH, "w", encoding="utf-8") as log_file:
        for idx, frame_path in enumerate(frame_files):
            frame_name = os.path.basename(frame_path)
            img = cv2.imread(frame_path)
            if img is None:
                continue
                
            log_file.write(f"--- Frame: {frame_name} ---\n")
            
            # --------------------------------------
            # 1. PADDLEOCR TEXT DETECTION & REDACTION
            # --------------------------------------
            result = ocr.predict(frame_path)
            
            if result:
                for res in result:
                    if not res or 'dt_polys' not in res:
                        continue
                        
                    boxes = res['dt_polys']
                    texts = res['rec_texts']
                    scores = res['rec_scores']
                    
                    for box, text, score in zip(boxes, texts, scores):
                        # Log everything found to the text file
                        log_file.write(f"[{score:.2f}] {text}\n")
                        
                        # If keyword matches, draw a black box
                        if check_keywords(text, KEYWORDS):
                            pts = np.array(box, dtype=np.int32)
                            cv2.fillPoly(img, [pts], (0, 0, 0))

            # --------------------------------------
            # 2. OPENCV LOGO MATCHING & REDACTION
            # --------------------------------------
            for logo_name, logo_img in logos:
                h, w = logo_img.shape[:2]
                
                # Perform Template Matching
                res_match = cv2.matchTemplate(img, logo_img, cv2.TM_CCOEFF_NORMED)
                loc = np.where(res_match >= LOGO_THRESHOLD)
                
                # Draw black boxes over matching logo locations
                for pt in zip(*loc[::-1]):
                    cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 0, 0), -1)
            
            # --------------------------------------
            # 3. SAVE AND PROGRESS TRACKING
            # --------------------------------------
            output_path = os.path.join(OUTPUT_FOLDER, frame_name)
            cv2.imwrite(output_path, img)
            
            frames_processed += 1
            log_file.write("\n")
            
            # Print real-time progress to console
            frames_left = len(frame_files) - frames_processed
            current_elapsed = time.time() - start_time
            current_fps = frames_processed / current_elapsed if current_elapsed > 0 else 0
            print(f"Processing: {frames_processed}/{len(frame_files)} | Left: {frames_left} | Current FPS: {current_fps:.2f}", end="\r")

    # Final Summary Benchmarking Statement
    elapsed = time.time() - start_time
    fps = frames_processed / elapsed if elapsed > 0 else 0
    print(f"\n✅ Worker {WORKER_ID} done: {len(frame_files)} frames in {elapsed:.4f} sec avg {fps:.4f} FPS)")
    print(f"📝 Full OCR log saved to: {LOG_FILE_PATH}")

if __name__ == "__main__":
    main()