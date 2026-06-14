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

KEYWORDS = [
    'stake.com', 'stake .com.', 'stake.com.', 'stake.com',
    'https://', 'Stake', 'stakecom', 'stake', 'Stake Originals', 'Only on Stake', 'bitcoin'
]

# Detector limit: set to tile-size upper bound to avoid auto-resize inside PaddleOCR.
TEXT_DET_LIMIT = 960
ocr = PaddleOCR(
    use_textline_orientation=True,
    lang='en',
    enable_mkldnn=False,
    text_det_limit_side_len=960,
    text_det_box_thresh=0.3,
    text_det_thresh=0.2,
    text_det_unclip_ratio=1.5
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

def _log_polygon_and_bbox(pts, x_off, y_off, log_file):
    pts_off = pts.copy().astype(np.int32)
    pts_off[:, 0] += int(x_off)
    pts_off[:, 1] += int(y_off)
    xmin = int(pts_off[:, 0].min())
    xmax = int(pts_off[:, 0].max())
    ymin = int(pts_off[:, 1].min())
    ymax = int(pts_off[:, 1].max())
    poly_str = ", ".join([f"({int(x)},{int(y)})" for x, y in pts_off.tolist()])
    log_file.write(f"    polygon: [{poly_str}]\n")
    log_file.write(f"    bbox: xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}\n")
    return pts_off, (xmin, ymin, xmax, ymax)

def _process_ocr_result_entries(res, x_off, y_off, full_img, keywords, log_file):
    detected = False
    for entry in res:
        if isinstance(entry, dict):
            boxes = entry.get('dt_polys') or entry.get('boxes') or []
            texts = entry.get('rec_texts') or entry.get('texts') or []
            scores = entry.get('rec_scores') or entry.get('scores') or []
            for box, text, score in zip(boxes, texts, scores):
                detected = True
                log_file.write(f"  - [{float(score):.2f}] {text}\n")
                pts = np.array(box, dtype=np.int32)
                pts_off, _ = _log_polygon_and_bbox(pts, x_off, y_off, log_file)
                if check_keywords(text, keywords):
                    cv2.fillPoly(full_img, [pts_off], (0, 0, 0))
        else:
            try:
                box = np.array(entry[0], dtype=np.int32)
                if isinstance(entry[1], (list, tuple)):
                    text = entry[1][0]
                else:
                    text = entry[1]
                score = float(entry[2]) if len(entry) > 2 else 1.0
            except Exception:
                continue
            detected = True
            log_file.write(f"  - [{float(score):.2f}] {text}\n")
            pts_off, _ = _log_polygon_and_bbox(box, x_off, y_off, log_file)
            if check_keywords(text, keywords):
                cv2.fillPoly(full_img, [pts_off], (0, 0, 0))
    return detected

def ocr_and_redact_tiles(img, keywords, log_file, tile_w=540, tile_h=960, overlap=128):
    """
    Strictly tile the image and run OCR only on each tile (never on the full image).
    Logs each tile's shape; if a tile exceeds the OCR detector limit, the tile is resized
    down to TEXT_DET_LIMIT while preserving aspect ratio before calling OCR.
    """
    h, w = img.shape[:2]
    if w < 2 or h < 2:
        return False

    any_detected = False
    step_x = max(1, tile_w - overlap)
    step_y = max(1, tile_h - overlap)

    for y in range(0, h, step_y):
        y1 = min(h, y + tile_h)
        y_start = y
        if y1 == h:
            y_start = max(0, h - tile_h)
            y = y_start
            y1 = h
        for x in range(0, w, step_x):
            x1 = min(w, x + tile_w)
            x_start = x
            if x1 == w:
                x_start = max(0, w - tile_w)
                x = x_start
                x1 = w

            tile = img[y:x1 and y1 or y:y1, x:x1].copy() if False else img[y:y1, x:x1]
            if tile.size == 0:
                continue

            # Log tile location & shape
            th, tw = tile.shape[:2]
            log_file.write(f"    tile @ x={x},y={y} size={tw}x{th}\n")

            # Convert BGR -> RGB for PaddleOCR
            tile_rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)

            # Ensure tile fits PaddleOCR detection limit; if not, resize tile (preserve ratio)
            max_side = max(th, tw)
            if max_side > TEXT_DET_LIMIT:
                scale = TEXT_DET_LIMIT / float(max_side)
                new_w = max(1, int(round(tw * scale)))
                new_h = max(1, int(round(th * scale)))
                log_file.write(f"      resizing tile to {new_w}x{new_h} to fit detector limit {TEXT_DET_LIMIT}\n")
                tile_rgb = cv2.resize(tile_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # Always pass ONLY the tile to OCR.
            try:
                # prefer the high-level API
                res = ocr.ocr(tile_rgb)
            except TypeError:
                # fallback to predict without unsupported kwargs
                try:
                    res = ocr.predict(tile_rgb)
                except Exception as e:
                    log_file.write(f"      OCR failed for tile x={x},y={y}: {e}\n")
                    continue
            except Exception as e:
                log_file.write(f"      OCR failed for tile x={x},y={y}: {e}\n")
                continue

            if not res:
                continue

            detected = _process_ocr_result_entries(res, x, y, img, keywords, log_file)
            any_detected = any_detected or detected

    return any_detected

# ==========================================
# MAIN
# ==========================================
def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    frame_files = sorted(glob(os.path.join(INPUT_FOLDER, "*.png")))
    # frame_files.extend(sorted(glob(os.path.join(INPUT_FOLDER, "*.jpg"))))  # optional

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

            # 1) OCR on tiles and redact
            detected = ocr_and_redact_tiles(img, KEYWORDS, log_file, tile_w=540, tile_h=960, overlap=128)
            if not detected:
                log_file.write("  - [No text detected]\n")

            # 2) Logo matching & redaction (works on full image)
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
