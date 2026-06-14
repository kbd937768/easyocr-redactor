# ==========================================
# GPU BATCHED PADDLEOCR REDACTOR
# WINDOWS + CUDA VERSION
# ==========================================

# ==========================================
# IMPORTS
# ==========================================
import os
import sys

site = os.path.join(
    os.path.dirname(sys.executable),
    "Lib",
    "site-packages"
)

paths = [
    os.path.join(site, "nvidia", "cu13", "bin", "x86_64"),
    os.path.join(site, "nvidia", "cudnn", "bin"),
]

for p in paths:
    if os.path.exists(p):
        os.add_dll_directory(p)

os.environ["PATH"] = (
    ";".join(paths)
    + ";"
    + os.environ["PATH"]
)

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
LOGO_FOLDER = "./logos"

CONFIDENCE_THRESHOLD = 0.65
MIN_ASPECT_RATIO = 2.0
PADDING = 20

# KEEP YOUR ORIGINAL TILE SETTINGS
TILE_SIZE = 960
OVERLAP = 128
STEP = TILE_SIZE - OVERLAP

LOGO_THRESHOLD = 0.95

# GPU BATCH SIZE
# Increase if you have lots of VRAM
BATCH_SIZE = 32

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
# NORMALIZED KEYWORDS
# ==========================================
NORMALIZED_KEYWORDS = set()

for kw in KEYWORDS:
    norm = re.sub(r'[^a-z0-9]', '', kw.lower())
    NORMALIZED_KEYWORDS.add(norm)

# ==========================================
# HELPERS
# ==========================================
def normalize_text(text):
    return re.sub(r'[^a-z0-9]', '', (text or "").lower())


def check_keywords(text):

    normalized = normalize_text(text)

    for kw in NORMALIZED_KEYWORDS:
        if kw in normalized:
            return True

    return False


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


def deduplicate_detections(
    detections,
    iou_threshold=0.5
):

    detections = sorted(
        detections,
        key=lambda x: x["score"],
        reverse=True
    )

    final = []

    for det in detections:

        keep = True

        for existing in final:

            if calculate_iou(
                det["bbox"],
                existing["bbox"]
            ) > iou_threshold:

                keep = False
                break

        if keep:
            final.append(det)

    return final


def format_elapsed_time(seconds: float) -> str:

    if seconds < 0:
        seconds = 0

    weeks = int(seconds // (7 * 24 * 3600))
    seconds %= (7 * 24 * 3600)

    days = int(seconds // (24 * 3600))
    seconds %= (24 * 3600)

    hours = int(seconds // 3600)
    seconds %= 3600

    minutes = int(seconds // 60)
    seconds = int(seconds % 60)

    parts = []

    if weeks > 0:
        parts.append(f"{weeks}w")

    if days > 0:
        parts.append(f"{days}d")

    if hours > 0:
        parts.append(f"{hours}h")

    if minutes > 0:
        parts.append(f"{minutes}m")

    parts.append(f"{seconds}s")

    return " ".join(parts)

# ==========================================
# OCR
# ==========================================
def create_ocr():

    import paddle

    print("Creating GPU OCR Engine...", flush=True)

    paddle.set_device('gpu')

    paddle.set_flags({
        'FLAGS_check_nan_inf': False,
    })

    ocr = PaddleOCR(
        use_textline_orientation=True,
        lang='en',

        # KEEP YOUR ORIGINAL DET SETTINGS
        text_det_limit_side_len=960,
        text_det_box_thresh=0.3,
        text_det_thresh=0.2,
        text_det_unclip_ratio=1.5,

        # SPEED
        use_tensorrt=False,
        precision='fp32'
    )

    print("GPU OCR Engine Ready", flush=True)

    return ocr


def process_detection(
    pts_off,
    text,
    score,
    detections
):

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


# ==========================================
# PROCESS BATCH RESULTS
# ==========================================
def process_batch_results(
    results,
    offsets,
    detections
):

    for res, (x_offset, y_offset) in zip(results, offsets):

        if not res:
            continue

        for entry in res:

            if isinstance(entry, dict):

                boxes = entry.get('dt_polys') or entry.get('boxes') or []
                texts = entry.get('rec_texts') or entry.get('texts') or []
                scores = entry.get('rec_scores') or entry.get('scores') or []

                for box, text, score in zip(
                    boxes,
                    texts,
                    scores
                ):

                    if float(score) < CONFIDENCE_THRESHOLD:
                        continue

                    if not check_keywords(text):
                        continue

                    pts = np.array(
                        box,
                        dtype=np.int32
                    ).copy()

                    pts[:, 0] += x_offset
                    pts[:, 1] += y_offset

                    process_detection(
                        pts,
                        text,
                        score,
                        detections
                    )

            else:

                try:

                    box = np.array(
                        entry[0],
                        dtype=np.int32
                    ).copy()

                    text = entry[1][0]
                    score = entry[1][1]

                except Exception:
                    continue

                if float(score) < CONFIDENCE_THRESHOLD:
                    continue

                if not check_keywords(text):
                    continue

                box[:, 0] += x_offset
                box[:, 1] += y_offset

                process_detection(
                    box,
                    text,
                    score,
                    detections
                )


# ==========================================
# GPU BATCH OCR
# ==========================================
def ocr_collect_detections(
    img,
    ocr
):

    detections = []

    h, w = img.shape[:2]

    tiles = []
    offsets = []

    # ======================================
    # BUILD TILES
    # ======================================
    for y in range(0, h, STEP):

        for x in range(0, w, STEP):

            x2 = min(w, x + TILE_SIZE)
            y2 = min(h, y + TILE_SIZE)

            tile = img[y:y2, x:x2]

            tiles.append(tile)
            offsets.append((x, y))

    # ======================================
    # RUN GPU OCR IN BATCHES
    # ======================================
    for i in range(0, len(tiles), BATCH_SIZE):

        batch_tiles = tiles[i:i + BATCH_SIZE]
        batch_offsets = offsets[i:i + BATCH_SIZE]

        try:
            results = ocr.predict(batch_tiles)

        except Exception as e:
            print(f"OCR ERROR: {e}")
            continue

        process_batch_results(
            results,
            batch_offsets,
            detections
        )

    return detections

# ==========================================
# WORKER
# ==========================================
def worker_process(
    worker_id,
    frame_files,
    frame_index,
    progress_counter,
    total_frames
):

    cv2.setNumThreads(1)

    logos = load_logos(LOGO_FOLDER)

    print(
        f"[Worker {worker_id}] Loading OCR...",
        flush=True
    )

    ocr = create_ocr()

    print(
        f"[Worker {worker_id}] OCR Loaded",
        flush=True
    )

    start_time = time.time()

    while True:

        # ==================================
        # GET NEXT FRAME
        # ==================================
        with frame_index.get_lock():

            idx = frame_index.value
            frame_index.value += 1

        if idx >= total_frames:
            break

        frame_path = frame_files[idx]

        frame_name = os.path.basename(frame_path)

        # ==================================
        # READ
        # ==================================
        t_read_start = time.time()

        img = cv2.imread(frame_path)

        t_read = time.time() - t_read_start

        if img is None:
            continue

        # ==================================
        # OCR
        # ==================================
        t_ocr_start = time.time()

        detections = ocr_collect_detections(
            img,
            ocr
        )

        detections = deduplicate_detections(
            detections,
            iou_threshold=0.5
        )

        all_boxes = [
            det["bbox"]
            for det in detections
        ]

        t_ocr = time.time() - t_ocr_start

        # ==================================
        # LOGO
        # ==================================
        t_logo_start = time.time()

        for _, logo_img in logos:

            lh, lw = logo_img.shape[:2]

            if (
                lh == 0 or
                lw == 0 or
                img.shape[0] < lh or
                img.shape[1] < lw
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

        t_logo = time.time() - t_logo_start

        # ==================================
        # REDACT
        # ==================================
        t_redact_start = time.time()

        h, w = img.shape[:2]

        for x1, y1, x2, y2 in all_boxes:

            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(w, int(x2))
            y2 = min(h, int(y2))

            img[y1:y2, x1:x2] = 0

        t_redact = time.time() - t_redact_start

        # ==================================
        # SAVE
        # ==================================
        t_save_start = time.time()

        _, ext = os.path.splitext(frame_path)

        if ext.lower() in ['.jpg', '.jpeg']:

            cv2.imwrite(
                frame_path,
                img,
                [cv2.IMWRITE_JPEG_QUALITY, 100]
            )

        else:

            cv2.imwrite(
                frame_path,
                img,
                [cv2.IMWRITE_PNG_COMPRESSION, 3]
            )

        t_save = time.time() - t_save_start

        # ==================================
        # TOTAL FRAME TIME
        # ==================================
        elapsed_frame = (
            t_read +
            t_ocr +
            t_logo +
            t_redact +
            t_save
        )

        # ==================================
        # PROGRESS
        # ==================================
        with progress_counter.get_lock():

            progress_counter.value += 1
            done = progress_counter.value

        elapsed_total = time.time() - start_time

        fps = (
            done / elapsed_total
            if elapsed_total > 0 else 0
        )

        eta = (
            (total_frames - done) / fps
            if fps > 0 else 0
        )

        print(
            f"[Worker {worker_id}] "
            f"{done}/{total_frames} | "
            f"Frame: {frame_name} | "
            f"Total: {elapsed_frame:.2f}s | "
            f"Read: {t_read:.2f}s | "
            f"OCR: {t_ocr:.2f}s | "
            f"Logo: {t_logo:.2f}s | "
            f"Redact: {t_redact:.2f}s | "
            f"Save: {t_save:.2f}s | "
            f"FPS: {fps:.2f} | "
            f"Elapsed: {format_elapsed_time(elapsed_total)} | "
            f"ETA: {format_elapsed_time(eta)}",
            flush=True
        )

    total_elapsed = time.time() - start_time

    print(
        f"✅ Worker {worker_id} finished "
        f"in {format_elapsed_time(total_elapsed)}",
        flush=True
    )

# ==========================================
# MAIN
# ==========================================
def main():

    frame_files = sorted(
        glob(os.path.join(INPUT_FOLDER, "*.png"))
    )

    frame_files.extend(sorted(
        glob(os.path.join(INPUT_FOLDER, "*.jpg"))
    ))

    frame_files.extend(sorted(
        glob(os.path.join(INPUT_FOLDER, "*.jpeg"))
    ))

    if not frame_files:
        print(f"❌ No frames found in '{INPUT_FOLDER}'")
        return

    print(f"\n📦 Total frames: {len(frame_files)}")

    # ======================================
    # GPU WORKER COUNT
    # ======================================
    recommended = 1

    worker_input = input(
        f"\nHow many GPU workers? "
        f"(recommended=1): "
    )

    try:

        worker_count = (
            int(worker_input)
            if worker_input.strip()
            else recommended
        )

    except:
        worker_count = recommended

    worker_count = max(1, worker_count)

    print(f"\n🚀 Starting {worker_count} GPU OCR worker(s)...")

    # ======================================
    # SHARED COUNTERS
    # ======================================
    frame_index = mp.Value('i', 0)

    progress_counter = mp.Value('i', 0)

    processes = []

    global_start = time.time()

    # ======================================
    # START WORKERS
    # ======================================
    for worker_id in range(worker_count):

        p = mp.Process(
            target=worker_process,
            args=(
                worker_id,
                frame_files,
                frame_index,
                progress_counter,
                len(frame_files)
            )
        )

        p.start()

        processes.append(p)

        time.sleep(1.0)

    # ======================================
    # WAIT
    # ======================================
    for p in processes:
        p.join()

    total_time = time.time() - global_start

    fps = (
        len(frame_files) / total_time
        if total_time > 0 else 0
    )

    print("\n✅ ALL WORKERS FINISHED")
    print(f"⏱ Total time: {format_elapsed_time(total_time)}")
    print(f"🚀 Average FPS: {fps:.2f}")

# ==========================================
# ENTRY
# ==========================================
if __name__ == "__main__":

    mp.freeze_support()

    # REQUIRED FOR WINDOWS + CUDA
    mp.set_start_method("spawn", force=True)

    main()