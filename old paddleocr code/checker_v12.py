import os
import cv2
import numpy as np
import time
import logging
import multiprocessing
from glob import glob
from paddleocr import PaddleOCR

# ==========================================
# CONFIG
# ==========================================
logging.getLogger("ppocr").setLevel(logging.WARNING)

INPUT_FOLDER = "./frames"
LOGO_FOLDER = "./logos"

# OCR settings
CONFIDENCE_THRESHOLD = 0.65
MIN_ASPECT_RATIO = 2.0
PADDING = 20

# Tile settings
TILE_SIZE = 960
OVERLAP = 128
STEP = TILE_SIZE - OVERLAP

# Logo detection
LOGO_THRESHOLD = 0.95

KEYWORDS = [
    'stake.com',
    'stake .com.',
    'stake.com.',
    'https://',
    'Stake',
    'stakecom',
    'stake',
    'Stake Originals',
    'Only on Stake',
    'casino',
    'live casino',
    'slots',
    'blackjack',
    'baccarat',
    'Roulette',
    'ustake'
]

# ==========================================
# HELPERS
# ==========================================
def load_logos(logo_dir):
    logos = []
    if not os.path.exists(logo_dir):
        print(f"⚠️ Logo directory '{logo_dir}' not found.")
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


def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    if interArea <= 0:
        return 0.0

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return interArea / float(areaA + areaB - interArea)


def deduplicate_detections(detections, iou_threshold=0.5):
    detections = sorted(
        detections,
        key=lambda x: x["score"],
        reverse=True
    )

    final = []
    for det in detections:
        keep = True
        for existing in final:
            iou = calculate_iou(det["bbox"], existing["bbox"])
            if iou > iou_threshold:
                keep = False
                break
        if keep:
            final.append(det)
    return final


# ==========================================
# OCR
# ==========================================
def create_ocr():
    return PaddleOCR(
        use_textline_orientation=True,
        lang='en',
        enable_mkldnn=False,
        text_det_limit_side_len=960,
        text_det_box_thresh=0.3,
        text_det_thresh=0.2,
        text_det_unclip_ratio=1.5
    )


def process_detection(pts_off, text, score, detections):
    x, y, w, h = cv2.boundingRect(pts_off)
    if h == 0:
        return

    aspect_ratio = w / float(h)
    if aspect_ratio < MIN_ASPECT_RATIO:
        return

    x1 = x - PADDING
    y1 = y - PADDING
    x2 = x + w + PADDING
    y2 = y + h + PADDING

    detections.append({
        "bbox": (x1, y1, x2, y2),
        "score": float(score),
        "text": text
    })


def _ocr_tile(tile, x_offset, y_offset, keywords, detections, ocr):
    try:
        res = ocr.predict(tile)
    except Exception as e:
        print(f"OCR ERROR: {e}")
        return

    if not res:
        return

    for entry in res:
        # PaddleOCR v3 dict format
        if isinstance(entry, dict):
            boxes = entry.get('dt_polys') or entry.get('boxes') or []
            texts = entry.get('rec_texts') or entry.get('texts') or []
            scores = entry.get('rec_scores') or entry.get('scores') or []

            for box, text, score in zip(boxes, texts, scores):
                if float(score) < CONFIDENCE_THRESHOLD:
                    continue
                if not check_keywords(text, keywords):
                    continue

                pts = np.array(box, dtype=np.int32).copy()
                pts[:, 0] += x_offset
                pts[:, 1] += y_offset

                process_detection(pts, text, score, detections)

        # Older PaddleOCR format
        else:
            try:
                box = np.array(entry[0], dtype=np.int32).copy()
                text = entry[1][0]
                score = entry[1][1]
            except Exception:
                continue

            if float(score) < CONFIDENCE_THRESHOLD:
                continue
            if not check_keywords(text, keywords):
                continue

            box[:, 0] += x_offset
            box[:, 1] += y_offset

            process_detection(box, text, score, detections)


def ocr_collect_detections(img, keywords, ocr):
    detections = []
    h, w = img.shape[:2]

    for y in range(0, h, STEP):
        for x in range(0, w, STEP):
            x2 = min(w, x + TILE_SIZE)
            y2 = min(h, y + TILE_SIZE)
            tile = img[y:y2, x:x2]

            _ocr_tile(tile, x, y, keywords, detections, ocr)

    return detections


# ==========================================
# WORKER
# ==========================================
def worker_process(worker_id, frame_files, total_frames):
    try:
        if os.name == "nt":
            import psutil
            proc = psutil.Process(os.getpid())
            proc.cpu_affinity([worker_id % os.cpu_count()])
    except Exception:
        pass

    worker_log_path = f"./ocr_results_worker_{worker_id}.txt"

    logos = load_logos(LOGO_FOLDER)
    ocr = create_ocr()

    start_time = time.time()

    with open(worker_log_path, "w", encoding="utf-8") as log_file:

        for idx, frame_path in enumerate(frame_files):

            frame_name = os.path.basename(frame_path)

            img = cv2.imread(frame_path)

            if img is None:
                continue

            # ==========================================
            # OCR DETECTIONS (NO DRAWING YET)
            # ==========================================

            detections = ocr_collect_detections(
                img,
                KEYWORDS,
                ocr
            )

            detections = deduplicate_detections(
                detections,
                iou_threshold=0.5
            )

            all_boxes = []

            for det in detections:
                all_boxes.append(det["bbox"])

            # ==========================================
            # LOGO DETECTIONS (ON ORIGINAL IMAGE)
            # ==========================================

            for logo_name, logo_img in logos:

                lh, lw = logo_img.shape[:2]

                if (
                    lh == 0
                    or lw == 0
                    or img.shape[0] < lh
                    or img.shape[1] < lw
                ):
                    continue

                res_match = cv2.matchTemplate(
                    img,
                    logo_img,
                    cv2.TM_CCOEFF_NORMED
                )

                loc = np.where(
                    res_match >= LOGO_THRESHOLD
                )

                for pt in zip(*loc[::-1]):

                    x1 = int(pt[0])
                    y1 = int(pt[1])
                    x2 = x1 + lw
                    y2 = y1 + lh

                    all_boxes.append(
                        (x1, y1, x2, y2)
                    )

            # ==========================================
            # APPLY ALL REDACTIONS ONCE
            # ==========================================

            h, w = img.shape[:2]

            for x1, y1, x2, y2 in all_boxes:

                x1 = max(0, int(x1))
                y1 = max(0, int(y1))
                x2 = min(w, int(x2))
                y2 = min(h, int(y2))

                img[y1:y2, x1:x2] = 0

            # ==========================================
            # SAVE PNG
            # ==========================================

            output_path = frame_path

            cv2.imwrite(
                output_path,
                img,
                [cv2.IMWRITE_PNG_COMPRESSION, 9]
            )

            log_file.write(f"{frame_name}\n")
            log_file.flush()

            elapsed = time.time() - start_time
            sec_per_frame = elapsed / (idx + 1)

            print(
                f"[Worker {worker_id}] "
                f"{idx + 1}/{len(frame_files)} | "
                f"{sec_per_frame:.2f} sec/frame"
            )

    total = time.time() - start_time

    print(
        f"✅ Worker {worker_id} done in "
        f"{total:.2f}s"
    )


# ==========================================
# MAIN
# ==========================================
def main():
    multiprocessing.freeze_support()


    frame_files = sorted(glob(os.path.join(INPUT_FOLDER, "*.png")))
    frame_files.extend(sorted(glob(os.path.join(INPUT_FOLDER, "*.jpg"))))

    if not frame_files:
        print(f"❌ No frames found in '{INPUT_FOLDER}'")
        return

    cpu_count = os.cpu_count()
    print(f"\nDetected CPU cores: {cpu_count}")

    worker_count = input(f"How many workers do you want to run? (1-{cpu_count}): ")
    try:
        worker_count = int(worker_count)
    except Exception:
        worker_count = 1

    worker_count = max(1, min(worker_count, cpu_count))
    print(f"\n🚀 Starting {worker_count} workers...")
    print(f"📦 Total frames: {len(frame_files)}")

    # Standard Python list chunking prevents pickling leaks across processes
    avg = len(frame_files) / float(worker_count)
    split_frames = []
    last = 0.0
    while last < len(frame_files):
        split_frames.append(frame_files[int(last):int(last + avg)])
        last += avg

    processes = []
    global_start = time.time()

    for worker_id in range(worker_count):
        worker_frames = split_frames[worker_id]
        p = multiprocessing.Process(
            target=worker_process,
            args=(worker_id, worker_frames, len(frame_files))
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    total_time = time.time() - global_start
    print(f"\n✅ ALL WORKERS FINISHED")
    print(f"⏱ Total time: {total_time:.2f}s")


if __name__ == "__main__":
    main()