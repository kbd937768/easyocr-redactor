import os
import cv2
import numpy as np
import time
import logging
from glob import glob
from paddleocr import PaddleOCR

# ==========================================
# CONFIG
# ==========================================
logging.getLogger("ppocr").setLevel(logging.WARNING)

INPUT_FOLDER = "./frames"
OUTPUT_FOLDER = "./processed"
LOG_FILE_PATH = "./ocr_results.txt"
LOGO_FOLDER = "./logos"

WORKER_ID = 1

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
    'bitcoin'
]

# ==========================================
# OCR INIT
# ==========================================
ocr = PaddleOCR(
    use_textline_orientation=True,
    lang='en',
    enable_mkldnn=False,

    # NEW PARAMETER NAMES
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


def _log_detection(pts_off, text, score, status, log_file):

    xmin = int(pts_off[:, 0].min())
    xmax = int(pts_off[:, 0].max())
    ymin = int(pts_off[:, 1].min())
    ymax = int(pts_off[:, 1].max())

    poly_str = ", ".join(
        [f"({int(x)},{int(y)})" for x, y in pts_off.tolist()]
    )

    log_file.write(f"  - [{float(score):.2f}] {text} ({status})\n")
    log_file.write(f"    polygon: [{poly_str}]\n")
    log_file.write(
        f"    bbox: xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}\n"
    )


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

            iou = calculate_iou(
                det["bbox"],
                existing["bbox"]
            )

            if iou > iou_threshold:
                keep = False
                break

        if keep:
            final.append(det)

    return final


# ==========================================
# DETECTION PROCESSING
# ==========================================
def process_detection(
    pts_off,
    text,
    score,
    detections,
    log_file
):

    x, y, w, h = cv2.boundingRect(pts_off)

    if h == 0:
        return

    aspect_ratio = w / float(h)

    if aspect_ratio < MIN_ASPECT_RATIO:

        _log_detection(
            pts_off,
            text,
            score,
            "Ignored: Failed Aspect Ratio Check",
            log_file
        )

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

    _log_detection(
        pts_off,
        text,
        score,
        "Queued For Redaction",
        log_file
    )


# ==========================================
# OCR TILE
# ==========================================
def _ocr_tile(
    tile,
    x_offset,
    y_offset,
    keywords,
    detections,
    log_file
):

    try:
        res = ocr.predict(tile)

    except Exception as e:
        print(f"OCR ERROR: {e}")
        return

    if not res:
        return

    for entry in res:

        # ----------------------------
        # PaddleOCR v3 dict format
        # ----------------------------
        if isinstance(entry, dict):

            boxes = (
                entry.get('dt_polys')
                or entry.get('boxes')
                or []
            )

            texts = (
                entry.get('rec_texts')
                or entry.get('texts')
                or []
            )

            scores = (
                entry.get('rec_scores')
                or entry.get('scores')
                or []
            )

            for box, text, score in zip(
                boxes,
                texts,
                scores
            ):

                if float(score) < CONFIDENCE_THRESHOLD:
                    continue

                if not check_keywords(text, keywords):
                    continue

                # IMPORTANT:
                # fresh copy
                pts = np.array(
                    box,
                    dtype=np.int32
                ).copy()

                # convert tile coords -> image coords
                pts[:, 0] = pts[:, 0] + x_offset
                pts[:, 1] = pts[:, 1] + y_offset

                process_detection(
                    pts,
                    text,
                    score,
                    detections,
                    log_file
                )

        # ----------------------------
        # Older PaddleOCR format
        # ----------------------------
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

            if not check_keywords(text, keywords):
                continue

            box[:, 0] = box[:, 0] + x_offset
            box[:, 1] = box[:, 1] + y_offset

            process_detection(
                box,
                text,
                score,
                detections,
                log_file
            )


# ==========================================
# OCR WHOLE IMAGE
# ==========================================
def ocr_collect_detections(
    img,
    keywords,
    log_file
):

    detections = []

    h, w = img.shape[:2]

    for y in range(0, h, STEP):

        for x in range(0, w, STEP):

            x2 = min(w, x + TILE_SIZE)
            y2 = min(h, y + TILE_SIZE)

            # IMPORTANT:
            # COPY TILE
            tile = img[y:y2, x:x2].copy()

            _ocr_tile(
                tile,
                x,
                y,
                keywords,
                detections,
                log_file
            )

    return detections


# ==========================================
# DRAW REDACTIONS
# ==========================================
def draw_redactions(img, detections):

    for det in detections:

        x1, y1, x2, y2 = det["bbox"]

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img.shape[1], x2)
        y2 = min(img.shape[0], y2)

        cv2.rectangle(
            img,
            (x1, y1),
            (x2, y2),
            (0, 0, 0),
            -1
        )


# ==========================================
# MAIN
# ==========================================
def main():

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    frame_files = sorted(
        glob(os.path.join(INPUT_FOLDER, "*.png"))
    )

    frame_files.extend(
        sorted(glob(os.path.join(INPUT_FOLDER, "*.jpg")))
    )

    if not frame_files:
        print(f"❌ No frames found in '{INPUT_FOLDER}'")
        return

    logos = load_logos(LOGO_FOLDER)

    print(f"🚀 Starting processing for {len(frame_files)} frames...")

    start_time = time.time()

    with open(LOG_FILE_PATH, "w", encoding="utf-8") as log_file:

        for idx, frame_path in enumerate(frame_files):

            frame_name = os.path.basename(frame_path)

            img = cv2.imread(frame_path)

            if img is None:
                continue

            log_file.write(f"--- Frame: {frame_name} ---\n")

            # =====================================
            # OCR DETECTIONS
            # =====================================
            detections = ocr_collect_detections(
                img,
                KEYWORDS,
                log_file
            )

            # =====================================
            # REMOVE DUPLICATES
            # =====================================
            detections = deduplicate_detections(
                detections,
                iou_threshold=0.5
            )

            # =====================================
            # DRAW BLACK BOXES
            # =====================================
            draw_redactions(
                img,
                detections
            )

            # =====================================
            # LOGO DETECTION
            # =====================================
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

                    cv2.rectangle(
                        img,
                        pt,
                        (pt[0] + lw, pt[1] + lh),
                        (0, 0, 0),
                        -1
                    )

            # =====================================
            # SAVE
            # =====================================
            output_path = os.path.join(
                OUTPUT_FOLDER,
                frame_name
            )

            cv2.imwrite(
                output_path,
                img
            )

            elapsed = time.time() - start_time

            sec_per_frame = elapsed / (idx + 1)

            print(
                f"Processing: {idx + 1}/{len(frame_files)} | "
                f"Sec/frame: {sec_per_frame:.2f}s",
                end="\r"
            )

            log_file.write("\n")
            log_file.flush()

    total = time.time() - start_time

    print(f"\n✅ Done in {total:.2f}s")
    print(f"📝 OCR log saved to: {LOG_FILE_PATH}")


if __name__ == "__main__":
    main()