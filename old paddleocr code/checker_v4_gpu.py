# ==========================================
# GPU BATCHED PADDLEOCR REDACTOR
# PP-OCRv6 FIXED 4K VERSION
# WINDOWS + CUDA
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
    + os.environ.get("PATH", "")
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
OCR_DEBUG_FOLDER = "./ocr_debug"

CONFIDENCE_THRESHOLD = 0.65
LOGO_THRESHOLD = 0.95

# ==========================================
# FINAL FIXED TILE SETTINGS
# ==========================================
TILE_SIZE = 1024
OVERLAP = 128
STEP = TILE_SIZE - OVERLAP

# FIX: Set to 1 so PaddleOCR stops stitching tiles horizontally
# and exceeding the 4000px max limit.
BATCH_SIZE = 1 

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

    norm = re.sub(
        r'[^a-z0-9]',
        '',
        kw.lower()
    )

    NORMALIZED_KEYWORDS.add(norm)

# ==========================================
# HELPERS
# ==========================================
def normalize_text(text):

    return re.sub(
        r'[^a-z0-9]',
        '',
        (text or "").lower()
    )


def check_keywords(text):

    normalized = normalize_text(text)

    print(f"OCR SAW: {normalized}")

    for kw in NORMALIZED_KEYWORDS:

        if kw in normalized:
            return True

    return False


def load_logos(logo_dir):

    logos = []

    if not os.path.exists(logo_dir):

        print(
            f"⚠️ Logo directory "
            f"'{logo_dir}' not found."
        )

        return logos

    for ext in ('*.png', '*.jpg', '*.jpeg'):

        for path in glob(
            os.path.join(logo_dir, ext)
        ):

            logo = cv2.imread(path)

            if logo is not None:

                logos.append((
                    os.path.basename(path),
                    logo
                ))

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

    areaA = (
        (boxA[2] - boxA[0]) *
        (boxA[3] - boxA[1])
    )

    areaB = (
        (boxB[2] - boxB[0]) *
        (boxB[3] - boxB[1])
    )

    return interArea / float(
        areaA + areaB - interArea
    )


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


def save_ocr_debug(
    frame_name,
    detections
):

    os.makedirs(
        OCR_DEBUG_FOLDER,
        exist_ok=True
    )

    txt_path = os.path.join(
        OCR_DEBUG_FOLDER,
        os.path.splitext(frame_name)[0] + ".txt"
    )

    with open(
        txt_path,
        "w",
        encoding="utf-8"
    ) as f:

        if not detections:

            f.write("NO DETECTIONS\n")
            return

        for det in detections:

            bbox = det["bbox"]
            score = det["score"]
            text = det["text"]

            f.write(
                f"TEXT: {text}\n"
                f"NORMALIZED: "
                f"{normalize_text(text)}\n"
                f"SCORE: {score:.4f}\n"
                f"BBOX: {bbox}\n"
                f"{'-'*60}\n"
            )

# ==========================================
# OCR
# ==========================================
def create_ocr():

    import paddle

    print(
        "Creating GPU OCR Engine...",
        flush=True
    )

    paddle.set_device('gpu')

    paddle.set_flags({
        'FLAGS_check_nan_inf': False,
    })

    ocr = PaddleOCR(

        lang='en',

        # ==================================
        # IMPORTANT: FIXED KWARG NAMES
        # Paddle ignores kwargs that don't match exactly.
        # "text_det_limit_side_len" wasn't a valid setting, 
        # so it was falling back to the 960px default and squishing your 1024px tiles.
        # ==================================
        det_limit_side_len=1024,
        
        det_db_box_thresh=0.25,
        det_db_thresh=0.15,

        # tighter polygons
        det_db_unclip_ratio=1.05,

        use_tensorrt=False,
        precision='fp32'
    )

    print(
        "GPU OCR Engine Ready",
        flush=True
    )

    return ocr

# ==========================================
# FIXED DETECTION
# ==========================================
def process_detection(
    pts,
    text,
    score,
    detections
):

    xs = pts[:, 0]
    ys = pts[:, 1]

    # Extreme bounds for AABB deduplication
    x1 = int(np.min(xs))
    y1 = int(np.min(ys))
    x2 = int(np.max(xs))
    y2 = int(np.max(ys))

    w = x2 - x1
    h = y2 - y1

    if w <= 0 or h <= 0:
        return

    # VERY SMALL PADDING
    pad_x = max(2, int(w * 0.03))
    pad_y = max(2, int(h * 0.10))

    # Polygon padding from centroid
    cx = np.mean(xs)
    cy = np.mean(ys)

    padded_pts = []
    for x, y in pts:
        px = x + pad_x if x > cx else x - pad_x
        py = y + pad_y if y > cy else y - pad_y
        padded_pts.append([px, py])

    padded_poly = np.array(padded_pts, dtype=np.int32)

    # We keep bbox purely for IOU dedupe checks
    bx1 = x1 - pad_x
    by1 = y1 - pad_y
    bx2 = x2 + pad_x
    by2 = y2 + pad_y

    detections.append({
        "poly": padded_poly,
        "bbox": (bx1, by1, bx2, by2),
        "score": float(score),
        "text": text
    })

# ==========================================
# PROCESS BATCH
# ==========================================
def process_batch_results(
    results,
    offsets,
    detections,
    raw_ocr_entries
):

    for res, (
        x_offset,
        y_offset
    ) in zip(results, offsets):

        if res is None:
            continue

        # ==================================
        # LIST FORMAT
        # ==================================
        if isinstance(res, list):

            for entry in res:

                try:

                    box = np.array(
                        entry[0],
                        dtype=np.int32
                    )

                    text = str(entry[1][0])

                    score = float(
                        entry[1][1]
                    )

                except Exception:
                    continue

                box[:, 0] += x_offset
                box[:, 1] += y_offset

                raw_ocr_entries.append({
                    "text": text,
                    "score": score,
                    "bbox": (
                        int(np.min(box[:, 0])),
                        int(np.min(box[:, 1])),
                        int(np.max(box[:, 0])),
                        int(np.max(box[:, 1]))
                    )
                })

                print(
                    f"OCR FOUND: "
                    f"{text} ({score:.3f})"
                )

                if score < CONFIDENCE_THRESHOLD:
                    continue

                if not check_keywords(text):
                    continue

                process_detection(
                    box,
                    text,
                    score,
                    detections
                )

        # ==================================
        # DICT FORMAT
        # ==================================
        elif isinstance(res, dict):

            boxes = (
                res.get("dt_polys")
                or res.get("boxes")
                or []
            )

            texts = (
                res.get("rec_texts")
                or res.get("texts")
                or []
            )

            scores = (
                res.get("rec_scores")
                or res.get("scores")
                or []
            )

            for box, text, score in zip(
                boxes,
                texts,
                scores
            ):

                pts = np.array(
                    box,
                    dtype=np.int32
                )

                pts[:, 0] += x_offset
                pts[:, 1] += y_offset

                raw_ocr_entries.append({
                    "text": text,
                    "score": float(score),
                    "bbox": (
                        int(np.min(pts[:, 0])),
                        int(np.min(pts[:, 1])),
                        int(np.max(pts[:, 0])),
                        int(np.max(pts[:, 1]))
                    )
                })

                print(
                    f"OCR FOUND: "
                    f"{text} ({score:.3f})"
                )

                if float(score) < CONFIDENCE_THRESHOLD:
                    continue

                if not check_keywords(text):
                    continue

                process_detection(
                    pts,
                    text,
                    score,
                    detections
                )

# ==========================================
# GPU OCR
# ==========================================
def ocr_collect_detections(
    img,
    ocr
):

    detections = []
    raw_ocr_entries = []

    h, w = img.shape[:2]

    tiles = []
    offsets = []

    # ======================================
    # BUILD TILES
    # ======================================
    for y in range(0, h, STEP):

        for x in range(0, w, STEP):

            x2 = min(
                w,
                x + TILE_SIZE
            )

            y2 = min(
                h,
                y + TILE_SIZE
            )

            tile = img[y:y2, x:x2]

            tiles.append(tile)

            offsets.append((x, y))

    # ======================================
    # OCR BATCHES
    # ======================================
    for i in range(
        0,
        len(tiles),
        BATCH_SIZE
    ):

        batch_tiles = (
            tiles[i:i + BATCH_SIZE]
        )

        batch_offsets = (
            offsets[i:i + BATCH_SIZE]
        )

        try:

            results = ocr.predict(
                batch_tiles
            )

        except Exception as e:

            print(f"OCR ERROR: {e}")

            continue

        process_batch_results(
            results,
            batch_offsets,
            detections,
            raw_ocr_entries
        )

    return detections, raw_ocr_entries

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

    logos = load_logos(
        LOGO_FOLDER
    )

    print(
        f"[Worker {worker_id}] "
        f"Loading OCR...",
        flush=True
    )

    ocr = create_ocr()

    print(
        f"[Worker {worker_id}] "
        f"OCR Loaded",
        flush=True
    )

    while True:

        with frame_index.get_lock():

            idx = frame_index.value

            frame_index.value += 1

        if idx >= total_frames:
            break

        frame_path = frame_files[idx]

        frame_name = os.path.basename(
            frame_path
        )

        img = cv2.imread(frame_path)

        if img is None:
            continue

        # ==================================
        # OCR
        # ==================================
        detections, raw_ocr_entries = (
            ocr_collect_detections(
                img,
                ocr
            )
        )

        detections = deduplicate_detections(
            detections,
            iou_threshold=0.5
        )

        save_ocr_debug(
            frame_name,
            raw_ocr_entries
        )

        # Build heterogeneous list for poly & rect redaction
        all_redactions = []

        for det in detections:
            all_redactions.append({
                "type": "poly",
                "data": det["poly"]
            })

        # ==================================
        # LOGO MATCH
        # ==================================
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

                all_redactions.append({
                    "type": "rect",
                    "data": (x1, y1, x2, y2)
                })

        # ==================================
        # REDACT (THE FIX)
        # ==================================
        h, w = img.shape[:2]

        for redaction in all_redactions:
            
            # Precisely map OCR polygons (handles slant/rotation)
            if redaction["type"] == "poly":
                cv2.fillPoly(
                    img,
                    [redaction["data"]],
                    (0, 0, 0)
                )
            
            # Map logo matches (Axis-aligned blocks)
            elif redaction["type"] == "rect":
                x1, y1, x2, y2 = redaction["data"]

                x1 = max(0, int(x1))
                y1 = max(0, int(y1))
                x2 = min(w, int(x2))
                y2 = min(h, int(y2))

                cv2.rectangle(
                    img,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 0),
                    thickness=-1
                )

        # ==================================
        # SAVE
        # ==================================
        _, ext = os.path.splitext(
            frame_path
        )

        if ext.lower() in [
            '.jpg',
            '.jpeg'
        ]:

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

        # ==================================
        # PROGRESS
        # ==================================
        with progress_counter.get_lock():

            progress_counter.value += 1

            done = progress_counter.value

        print(
            f"[Worker {worker_id}] "
            f"{done}/{total_frames} | "
            f"{frame_name}",
            flush=True
        )

    print(
        f"✅ Worker {worker_id} finished",
        flush=True
    )

# ==========================================
# MAIN
# ==========================================
def main():

    os.makedirs(
        OCR_DEBUG_FOLDER,
        exist_ok=True
    )

    frame_files = sorted(
        glob(os.path.join(
            INPUT_FOLDER,
            "*.png"
        ))
    )

    frame_files.extend(sorted(
        glob(os.path.join(
            INPUT_FOLDER,
            "*.jpg"
        ))
    ))

    frame_files.extend(sorted(
        glob(os.path.join(
            INPUT_FOLDER,
            "*.jpeg"
        ))
    ))

    if not frame_files:

        print(
            f"❌ No frames found "
            f"in '{INPUT_FOLDER}'"
        )

        return

    print(
        f"\n📦 Total frames: "
        f"{len(frame_files)}"
    )

    worker_input = input(
        "\nHow many GPU workers? "
        "(recommended=1): "
    )

    try:

        worker_count = (
            int(worker_input)
            if worker_input.strip()
            else 1
        )

    except:
        worker_count = 1

    worker_count = max(
        1,
        worker_count
    )

    print(
        f"\n🚀 Starting "
        f"{worker_count} GPU worker(s)..."
    )

    frame_index = mp.Value('i', 0)

    progress_counter = mp.Value('i', 0)

    processes = []

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

    for p in processes:
        p.join()

    print("\n✅ ALL WORKERS FINISHED")

# ==========================================
# ENTRY
# ==========================================
if __name__ == "__main__":

    mp.freeze_support()

    mp.set_start_method(
        "spawn",
        force=True
    )

    main()