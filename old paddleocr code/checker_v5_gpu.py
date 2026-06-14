# ==========================================
# PADDLEOCR REDACTOR (NO LOGOS) - Updated for PaddleOCR 3.x
# PP-OCRv6 FIXED 4K VERSION
# WINDOWS + CUDA
# ==========================================

import os
import sys
import cv2
import re
import time
import logging
import numpy as np
import multiprocessing as mp
from glob import glob
from paddleocr import PaddleOCR

# ==========================================
# CONFIG
# ==========================================
logging.getLogger("ppocr").setLevel(logging.WARNING)

INPUT_FOLDER = "./frames"
OCR_DEBUG_FOLDER = "./ocr_debug"

# Keyword confidence threshold
CONFIDENCE_THRESHOLD = 0.50

# Tile Settings
TILE_SIZE = 1024
OVERLAP = 128
STEP = TILE_SIZE - OVERLAP

KEYWORDS = [
    'stake.com', 'stake .com.', 'stake.com.', 'https://', 'Stake', 'stakecom',
    'stake', 'Stake Originals', 'Only on Stake', 'casino', 'live casino',
    'slots', 'blackjack', 'baccarat', 'Roulette', 'ustake'
]

NORMALIZED_KEYWORDS = set(re.sub(r'[^a-z0-9]', '', kw.lower()) for kw in KEYWORDS)

# ==========================================
# HELPERS
# ==========================================
def normalize_text(text):
    return re.sub(r'[^a-z0-9]', '', (text or "").lower())

def check_keywords(text):
    normalized = normalize_text(text)
    print(f"OCR SAW: {normalized}")
    for kw in NORMALIZED_KEYWORDS:
        if kw in normalized:
            return True
    return False

def calculate_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea <= 0: return 0.0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(areaA + areaB - interArea)

def deduplicate_detections(detections, iou_threshold=0.5):
    detections = sorted(detections, key=lambda x: x["score"], reverse=True)
    final = []
    for det in detections:
        if not any(calculate_iou(det["bbox"], ext["bbox"]) > iou_threshold for ext in final):
            final.append(det)
    return final

# ==========================================
# ENGINE SETUP
# ==========================================
def create_ocr():
    import paddle
    print("Creating GPU OCR Engine...", flush=True)
    paddle.set_device('gpu')
    paddle.set_flags({'FLAGS_check_nan_inf': False})

    ocr = PaddleOCR(
        lang='en',
        det_limit_type='max',
        det_limit_side_len=1024,
        det_db_thresh=0.30,
        det_db_box_thresh=0.60,
        det_db_unclip_ratio=1.5,
        use_angle_cls=False,
        use_tensorrt=False,
        precision='fp32'
    )
    print("GPU OCR Engine Ready", flush=True)
    return ocr

# ==========================================
# OCR PROCESSING (NO BATCH STITCHING)
# Adapted to handle PaddleOCR 3.x return formats
# ==========================================
def _normalize_ocr_results(results):
    """
    Normalize multiple possible PaddleOCR return formats into a list of lines:
    Each line is [box, (text, score)]
    """
    lines = []

    # None or empty
    if not results:
        return lines

    # Case A: older style: results = [ [ [box, (text, score)], ... ] ]
    if isinstance(results, list) and results and isinstance(results[0], list):
        # assume first element holds the detections for the image
        first = results[0]
        for item in first:
            # item is [box, (text, score)]
            lines.append(item)
        return lines

    # Case B: new-style list of dicts (one dict per image or per detection)
    if isinstance(results, list) and isinstance(results[0], dict):
        # Try patterns:
        # - per-image dict: {'boxes': [...], 'texts': [...], 'scores': [...]}
        # - per-detection dicts: [{'box': [...], 'text': '...', 'score': ...}, ...]
        # Detect per-image dict:
        first = results[0]
        if any(k in first for k in ('boxes', 'texts', 'scores', 'confs', 'scores_list')):
            for item in results:
                boxes = item.get('boxes') or item.get('boxes_list') or item.get('bbox') or []
                texts = item.get('texts') or item.get('text') or []
                scores = item.get('scores') or item.get('confs') or item.get('scores_list') or []
                # If boxes is nested list of boxes
                if boxes and isinstance(boxes[0], (list, tuple, np.ndarray)):
                    # ensure lengths align using zip
                    for idx, b in enumerate(boxes):
                        t = texts[idx] if idx < len(texts) else ""
                        s = scores[idx] if idx < len(scores) else 0.0
                        lines.append([b, (t, s)])
                else:
                    # fallback single-item
                    if boxes:
                        t = texts if isinstance(texts, str) else (texts[0] if texts else "")
                        s = scores if isinstance(scores, (float, int)) else (scores[0] if scores else 0.0)
                        lines.append([boxes, (t, s)])
            return lines

        # Else assume list of per-detection dicts
        for det in results:
            if not isinstance(det, dict):
                continue
            b = det.get('box') or det.get('bbox') or det.get('boxes') or det.get('position')
            t = det.get('text') or det.get('texts') or ""
            s = det.get('score') or det.get('scores') or det.get('conf') or det.get('confs') or 0.0
            if b is not None:
                lines.append([b, (t, s)])
        return lines

    # Unknown format - return empty; caller can print results to debug
    return lines

def process_frame(img, ocr):
    detections = []
    raw_ocr_entries = []
    h, w = img.shape[:2]

    for y in range(0, h, STEP):
        for x in range(0, w, STEP):
            x2 = min(w, x + TILE_SIZE)
            y2 = min(h, y + TILE_SIZE)
            tile = img[y:y2, x:x2]

            # Call OCR without deprecated kwargs
            results = ocr.ocr(tile)

            if not results:
                continue

            lines = _normalize_ocr_results(results)

            if not lines:
                # If normalization failed, dump raw result once for debugging and continue
                print("Unexpected OCR return format (dumping once):", type(results), flush=True)
                print(results, flush=True)
                continue

            for line in lines:
                try:
                    box = np.array(line[0], dtype=np.int32)
                except Exception:
                    # skip invalid boxes
                    continue

                text = str(line[1][0]) if isinstance(line[1], (list, tuple)) else str(line[1])
                try:
                    score = float(line[1][1]) if isinstance(line[1], (list, tuple)) else float(line[2]) if len(line) > 2 else 0.0
                except Exception:
                    # fallback to 1.0 if score missing
                    try:
                        score = float(line[1]) if isinstance(line[1], (float, int)) else 0.0
                    except:
                        score = 0.0

                # Shift polygon coordinates by the tile's offset in the main image
                if box.ndim == 2 and box.shape[1] >= 2:
                    box[:, 0] += x
                    box[:, 1] += y
                else:
                    # handle degenerate box format (e.g., [xmin,ymin,xmax,ymax])
                    if box.size == 4:
                        bx = box.copy()
                        bx[0] += x
                        bx[1] += y
                        bx[2] += x
                        bx[3] += y
                        xmin, ymin, xmax, ymax = int(min(bx[0], bx[2])), int(min(bx[1], bx[3])), int(max(bx[0], bx[2])), int(max(bx[1], bx[3]))
                        raw_ocr_entries.append({
                            "text": text,
                            "score": score,
                            "bbox": (xmin, ymin, xmax, ymax)
                        })
                        if score < CONFIDENCE_THRESHOLD:
                            continue
                        if not check_keywords(text):
                            continue
                        pad_x = max(2, int((xmax - xmin) * 0.02))
                        pad_y = max(2, int((ymax - ymin) * 0.05))
                        cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
                        padded_poly = np.array([
                            [xmin - pad_x, ymin - pad_y],
                            [xmax + pad_x, ymin - pad_y],
                            [xmax + pad_x, ymax + pad_y],
                            [xmin - pad_x, ymax + pad_y]
                        ], dtype=np.int32)
                        detections.append({
                            "poly": padded_poly,
                            "bbox": (xmin - pad_x, ymin - pad_y, xmax + pad_x, ymax + pad_y),
                            "score": score,
                            "text": text
                        })
                        continue
                    else:
                        continue

                xmin, ymin = int(np.min(box[:, 0])), int(np.min(box[:, 1]))
                xmax, ymax = int(np.max(box[:, 0])), int(np.max(box[:, 1]))

                raw_ocr_entries.append({
                    "text": text,
                    "score": score,
                    "bbox": (xmin, ymin, xmax, ymax)
                })

                print(f"OCR FOUND: {text} ({score:.3f})")

                if score < CONFIDENCE_THRESHOLD:
                    continue

                if not check_keywords(text):
                    continue

                pad_x = max(2, int((xmax - xmin) * 0.02))
                pad_y = max(2, int((ymax - ymin) * 0.05))

                cx, cy = np.mean(box[:, 0]), np.mean(box[:, 1])
                padded_pts = []
                for bx, by in box:
                    px = int(bx + pad_x) if bx > cx else int(bx - pad_x)
                    py = int(by + pad_y) if by > cy else int(by - pad_y)
                    padded_pts.append([px, py])

                padded_poly = np.array(padded_pts, dtype=np.int32)

                detections.append({
                    "poly": padded_poly,
                    "bbox": (xmin - pad_x, ymin - pad_y, xmax + pad_x, ymax + pad_y),
                    "score": score,
                    "text": text
                })

    return detections, raw_ocr_entries

# ==========================================
# WORKER
# ==========================================
def worker_process(worker_id, frame_files, frame_index, progress_counter, total_frames):
    cv2.setNumThreads(1)
    ocr = create_ocr()

    os.makedirs(OCR_DEBUG_FOLDER, exist_ok=True)

    while True:
        with frame_index.get_lock():
            idx = frame_index.value
            frame_index.value += 1

        if idx >= total_frames:
            break

        frame_path = frame_files[idx]
        frame_name = os.path.basename(frame_path)
        img = cv2.imread(frame_path)

        if img is None:
            continue

        detections, raw_ocr_entries = process_frame(img, ocr)
        detections = deduplicate_detections(detections, iou_threshold=0.5)

        # Draw Polygons
        for det in detections:
            try:
                cv2.fillPoly(img, [det["poly"]], (0, 0, 0))
            except Exception:
                # fallback draw rectangle
                x1, y1, x2, y2 = det["bbox"]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), thickness=-1)

        # Save Image
        _, ext = os.path.splitext(frame_path)
        if ext.lower() in ['.jpg', '.jpeg']:
            cv2.imwrite(frame_path, img, [cv2.IMWRITE_JPEG_QUALITY, 100])
        else:
            cv2.imwrite(frame_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 3])

        # Write Debug Log
        txt_path = os.path.join(OCR_DEBUG_FOLDER, os.path.splitext(frame_name)[0] + ".txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            if not raw_ocr_entries:
                f.write("NO DETECTIONS\n")
            else:
                for entry in raw_ocr_entries:
                    f.write(f"TEXT: {entry['text']}\nSCORE: {entry['score']:.4f}\nBBOX: {entry['bbox']}\n{'-'*30}\n")

        with progress_counter.get_lock():
            progress_counter.value += 1
            done = progress_counter.value

        print(f"[Worker {worker_id}] {done}/{total_frames} | {frame_name}", flush=True)

    print(f"✅ Worker {worker_id} finished", flush=True)

# ==========================================
# MAIN
# ==========================================
def main():
    # Windows CUDA DLL Path Fix (keep as in your original script)
    site = os.path.join(os.path.dirname(sys.executable), "Lib", "site-packages")
    paths = [
        os.path.join(site, "nvidia", "cu13", "bin", "x86_64"),
        os.path.join(site, "nvidia", "cudnn", "bin"),
    ]
    for p in paths:
        if os.path.exists(p):
            try:
                os.add_dll_directory(p)
            except Exception:
                pass
    os.environ["PATH"] = ";".join([p for p in paths if os.path.exists(p)]) + ";" + os.environ.get("PATH", "")

    frame_files = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        frame_files.extend(glob(os.path.join(INPUT_FOLDER, ext)))

    frame_files = sorted(frame_files)

    if not frame_files:
        print(f"❌ No frames found in '{INPUT_FOLDER}'")
        return

    print(f"\n📦 Total frames: {len(frame_files)}")

    worker_input = input("\nHow many GPU workers? (recommended=1): ")
    try:
        worker_count = int(worker_input) if worker_input.strip() else 1
    except:
        worker_count = 1

    worker_count = max(1, worker_count)
    print(f"\n🚀 Starting {worker_count} GPU worker(s)...")

    frame_index = mp.Value('i', 0)
    progress_counter = mp.Value('i', 0)
    processes = []

    for worker_id in range(worker_count):
        p = mp.Process(
            target=worker_process,
            args=(worker_id, frame_files, frame_index, progress_counter, len(frame_files))
        )
        p.start()
        processes.append(p)
        time.sleep(1.0)

    for p in processes:
        p.join()

    print("\n✅ ALL WORKERS FINISHED")

if __name__ == "__main__":
    mp.freeze_support()
    mp.set_start_method("spawn", force=True)
    main()
