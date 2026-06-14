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
LOGO_THRESHOLD = 0.8
WORKER_ID = 1

KEYWORDS = [
    'stake.com', 'stake .com.', 'stake.com.', 'stake.com',
    'https://', 'Stake', 'stakecom', 'stake', 'Stake Originals', 'Only on Stake', 'bitcoin'
]

# Initialize PaddleOCR
ocr = PaddleOCR(
    use_textline_orientation=True,
    lang='en',
    enable_mkldnn=False,
    text_det_limit_side_len=3200,
    text_det_box_thresh=0.3,
    text_det_thresh=0.2,
    text_det_unclip_ratio=2.0
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

def ocr_and_redact_halves(img, keywords, log_file, frame_name, overlap_px=128):
    """
    Split 4K 16:9 image into left/right halves with overlap, OCR each tile (numpy input),
    write detections to log_file, and redact keywords on img in-place.
    Returns True if any text detected.
    """
    h, w = img.shape[:2]
    if w < 2 or h < 2:
        return False

    mid = w // 2
    # left half: x from 0 to mid+ov_half; right half: x from mid-ov_half to w
    ov_half = overlap_px // 2
    halves = [
        (0, min(w, mid + ov_half)),              # left: x1, x2
        (max(0, mid - ov_half), w)               # right: x1, x2
    ]

    any_detected = False

    for x1, x2 in halves:
        tile = img[:, x1:x2]
        # call OCR on numpy image; many PaddleOCR versions accept ocr.ocr(tile, cls=True)
        try:
            res = ocr.ocr(tile, cls=True)
        except TypeError:
            res = ocr.ocr(tile)

        if not res:
            continue

        # res may be list of [box, (text, score)] or list of dicts; handle common formats
        for entry in res:
            if isinstance(entry, dict):
                # dict variant: expect keys like 'dt_polys', 'rec_texts', 'rec_scores'
                boxes = entry.get('dt_polys') or entry.get('boxes') or []
                texts = entry.get('rec_texts') or entry.get('texts') or []
                scores = entry.get('rec_scores') or entry.get('scores') or []
                for box, text, score in zip(boxes, texts, scores):
                    any_detected = True
                    log_file.write(f"  - [{float(score):.2f}] {text}\n")
                    # offset x coords by x1
                    pts = np.array(box, dtype=np.int32)
                    pts[:, 0] += x1
                    if check_keywords(text, keywords):
                        cv2.fillPoly(img, [pts], (0, 0, 0))
            else:
                # list variant: [box, (text, score)]
                try:
                    box = np.array(entry[0], dtype=np.int32)
                    text, score = entry[1][0], entry[1][1]
                except Exception:
                    # fallback if format differs slightly
                    continue
                any_detected = True
                log_file.write(f"  - [{float(score):.2f}] {text}\n")
                box[:, 0] += x1
                if check_keywords(text, keywords):
                    cv2.fillPoly(img, [box], (0, 0, 0))

    return any_detected

# ==========================================
# MAIN
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
            # 1) OCR on halves and redact
            detected = ocr_and_redact_halves(img, KEYWORDS, log_file, frame_name, overlap_px=128)
            if not detected:
                log_file.write("  - [No text detected]\n")

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
