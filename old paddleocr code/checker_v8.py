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
logging.getLogger("ppocr").setLevel(logging.WARNING)

INPUT_FOLDER = "./frames"
OUTPUT_FOLDER = "./processed"
LOG_FILE_PATH = "./ocr_results.txt"
LOGO_FOLDER = "./logos"
LOGO_THRESHOLD = 0.95
WORKER_ID = 1

# --- MERGED DEFENSIVE CONFIGURATIONS ---
PADDING = 20                 # Expands the black box outwards to fully cover text
CONFIDENCE_THRESHOLD = 0.65  # Filters out low-confidence background noise
MIN_ASPECT_RATIO = 2.0       # Rejects squarish items (coins are ~1.6). Words are wide (> 2.0).
# --------------------------------------

KEYWORDS = [
    'stake.com', 'stake .com.', 'stake.com.', 'stake.com',
    'https://', 'Stake', 'stakecom', 'stake', 'Stake Originals', 'Only on Stake', 'bitcoin'
]

# Initialize PaddleOCR with corrected true 4K scaling parameter names
ocr = PaddleOCR(
    use_textline_orientation=True,
    lang='en',
    enable_mkldnn=False,
    det_limit_side_len=3200,     
    det_db_box_thresh=0.3,       
    det_db_thresh=0.2,           
    det_db_unclip_ratio=1.5      
)

# ==========================================
# HELPERS
# ==========================================
def load_logos(logo_dir):
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
    text_lower = (text or "").lower().strip()
    for kw in keyword_list:
        if kw.lower() in text_lower:
            return True
    return False

def _log_detection(pts_off, text, score, status, log_file):
    """Utility to log full geometry and status to ocr_results.txt"""
    xmin = int(pts_off[:, 0].min())
    xmax = int(pts_off[:, 0].max())
    ymin = int(pts_off[:, 1].min())
    ymax = int(pts_off[:, 1].max())
    poly_str = ", ".join([f"({int(x)},{int(y)})" for x, y in pts_off.tolist()])
    log_file.write(f"  - [{float(score):.2f}] {text} ({status})\n")
    log_file.write(f"    polygon: [{poly_str}]\n")
    log_file.write(f"    bbox: xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}\n")

def process_and_draw_box(img, pts_off, text, score, log_file):
    """Checks aspect ratio to filter false positives, then covers text with padded box."""
    x, y, w, h = cv2.boundingRect(pts_off)
    
    if h == 0: 
        return False
        
    aspect_ratio = w / float(h)
    
    # Check if shape is a wide rectangle (text) or squarish (coin/button)
    if aspect_ratio < MIN_ASPECT_RATIO:
        _log_detection(pts_off, text, score, "Ignored: Failed Aspect Ratio Check", log_file)
        return False

    # Apply generous padding to cover surrounding text margins
    x1 = max(0, x - PADDING)
    y1 = max(0, y - PADDING)
    x2 = min(img.shape[1], x + w + PADDING)
    y2 = min(img.shape[0], y + h + PADDING)
    
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), -1)
    _log_detection(pts_off, text, score, "Redacted", log_file)
    return True

def _ocr_tile_and_redact(tile, x_off, y_off, full_img, keywords, log_file):
    """Performs OCR over the slice and handles coordinates mapping."""
    try:
        res = ocr.predict(tile, cls=True)
    except TypeError:
        res = ocr.predict(tile)
        
    detected = False
    if not res:
        return detected

    for entry in res:
        if isinstance(entry, dict):
            boxes = entry.get('dt_polys') or entry.get('boxes') or []
            texts = entry.get('rec_texts') or entry.get('texts') or []
            scores = entry.get('rec_scores') or entry.get('scores') or []
            
            for box, text, score in zip(boxes, texts, scores):
                if float(score) < CONFIDENCE_THRESHOLD:
                    continue
                
                pts = np.array(box, dtype=np.int32)
                pts[:, 0] += x_off
                pts[:, 1] += y_off
                
                if check_keywords(text, keywords):
                    if process_and_draw_box(full_img, pts, text, score, log_file):
                        detected = True
        else:
            try:
                box = np.array(entry[0], dtype=np.int32)
                text, score = entry[1][0], entry[1][1]
            except Exception:
                try:
                    box = np.array(entry[0], dtype=np.int32)
                    text = entry[1]
                    score = float(entry[2]) if len(entry) > 2 else 1.0
                except Exception:
                    continue
            
            if float(score) < CONFIDENCE_THRESHOLD:
                continue
                    
            box[:, 0] += x_off
            box[:, 1] += y_off
            
            if check_keywords(text, keywords):
                if process_and_draw_box(full_img, box, text, score, log_file):
                    detected = True
                
    return detected

def ocr_and_redact_halves(img, keywords, log_file, overlap_px=128, max_side_limit=3200):
    """Processes image in large side-by-side halves to keep text lines unbroken."""
    h, w = img.shape[:2]
    if w < 2 or h < 2:
        return False

    mid = w // 2
    ov_half = overlap_px // 2
    halves = [(0, min(w, mid + ov_half)), (max(0, mid - ov_half), w)]

    any_detected = False

    for x1, x2 in halves:
        half_w = x2 - x1
        half_h = h

        if max(half_w, half_h) > max_side_limit:
            row_h = min(max_side_limit, half_h)
            step = max(1, int(row_h * 0.9))
            for y in range(0, half_h, step):
                y0 = y
                y1 = min(half_h, y + row_h)
                tile = img[y0:y1, x1:x2]
                any_detected |= _ocr_tile_and_redact(tile, x1, y0, img, keywords, log_file)
        else:
            tile = img[:, x1:x2]
            any_detected |= _ocr_tile_and_redact(tile, x1, 0, img, keywords, log_file)

    return any_detected

# ==========================================
# MAIN
# ==========================================
def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    frame_files = sorted(glob(os.path.join(INPUT_FOLDER, "*.png")))
    frame_files.extend(sorted(glob(os.path.join(INPUT_FOLDER, "*.jpg"))))
    
    logos = load_logos(LOGO_FOLDER)

    if not frame_files:
        print(f"❌ No frames found in '{INPUT_FOLDER}'. Please check the path.")
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
            log_file.write(f"frame size: {img.shape[1]}x{img.shape[0]}\n")

            # 1) OCR on unbroken halves and redact safely
            detected = ocr_and_redact_halves(img, KEYWORDS, log_file, overlap_px=128, max_side_limit=3200)
            if not detected:
                log_file.write("  - [No text redacted]\n")

            # 2) Logo matching & redaction
            for logo_name, logo_img in logos:
                lh, lw = logo_img.shape[:2]
                if lh == 0 or lw == 0 or img.shape[0] < lh or img.shape[1] < lw:
                    continue
                res_match = cv2.matchTemplate(img, logo_img, cv2.TM_CCOEFF_NORMED)
                loc = np.where(res_match >= LOGO_THRESHOLD)
                for pt in zip(*loc[::-1]):
                    cv2.rectangle(img, pt, (pt[0] + lw, pt[1] + lh), (0, 0, 0), -1)

            # 3) Save & progress
            output_path = os.path.join(OUTPUT_FOLDER, frame_name)
            cv2.imwrite(output_path, img)

            frames_processed += 1
            log_file.write("\n")
            log_file.flush()

            frames_left = len(frame_files) - frames_processed
            current_elapsed = time.time() - start_time
            sec_per_frame = current_elapsed / frames_processed if frames_processed > 0 else 0
            print(f"Processing: {frames_processed}/{len(frame_files)} | Left: {frames_left} | Sec/Frame: {sec_per_frame:.2f}s", end="\r")

    elapsed = time.time() - start_time
    avg_sec_per_frame = elapsed / frames_processed if frames_processed > 0 else 0
    print(f"\n✅ Worker {WORKER_ID} done: {len(frame_files)} frames in {elapsed:.4f} sec (avg {avg_sec_per_frame:.4f} sec/frame)")
    print(f"📝 Full OCR log saved to: {LOG_FILE_PATH}")

if __name__ == "__main__":
    main()