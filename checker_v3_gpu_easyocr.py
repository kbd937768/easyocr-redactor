import os
import time
import cv2
import numpy as np
import easyocr

FRAMES_DIR = "frames"
READER_DEVICE = 'cuda'  # 'cuda' to use GPU (5060 Ti) or 'cpu' to force CPU
OCR_MIN_CONF = 0.45

KEYWORDS = [
    'stake.com', 'stake .com.', 'stake.com.', 'https://', 'Stake',
    'stakecom', 'stake', 'Stake Originals', 'Only on Stake',
    'casino', 'live casino', 'slots', 'blackjack', 'baccarat',
    'Roulette', 'ustake', 'gamble', 'betting', 'bet', 'gambling', 'online gambling'
]
KEYWORDS_LOWER = [k.lower() for k in KEYWORDS]

reader = easyocr.Reader(['en'], gpu=(READER_DEVICE == 'cuda'))

def normalize_text(t): return (t or "").lower().strip()
def text_matches(t):
    nt = normalize_text(t)
    for kw in KEYWORDS_LOWER:
        if kw in nt: return True
    return False

def expand_box(box, w, h, pad=6):
    xs = [int(round(p[0])) for p in box]
    ys = [int(round(p[1])) for p in box]
    x1 = max(0, min(xs) - pad); y1 = max(0, min(ys) - pad)
    x2 = min(w, max(xs) + pad); y2 = min(h, max(ys) + pad)
    return x1, y1, x2, y2

def merge_rects(rects, iou_thresh=0.15):
    if not rects: return []
    boxes = np.array(rects)
    x1 = boxes[:,0]; y1 = boxes[:,1]; x2 = boxes[:,2]; y2 = boxes[:,3]
    areas = (x2-x1)*(y2-y1)
    order = areas.argsort()[::-1]
    keep = []
    while order.size>0:
        i = order[0]; keep.append((int(x1[i]),int(y1[i]),int(x2[i]),int(y2[i])))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w_ = np.maximum(0.0, xx2-xx1); h_ = np.maximum(0.0, yy2-yy1)
        inter = w_*h_
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]
    return keep

def process_file(path, reader):
    fname = os.path.basename(path)
    t_read_start = time.time()
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    t_read = time.time() - t_read_start
    if img is None:
        print(f"Failed to read {fname}"); return None

    has_alpha = (img.ndim==3 and img.shape[2]==4)
    bgr = img[:,:,:3] if has_alpha else img
    h,w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    t_ocr_start = time.time()
    results = reader.readtext(rgb, detail=1, paragraph=False)
    t_ocr = time.time() - t_ocr_start

    rects=[]
    for bbox, text, prob in results:
        if prob >= OCR_MIN_CONF and text_matches(text):
            rects.append(expand_box(bbox, w, h, pad=6))

    merged = merge_rects(rects, iou_thresh=0.15)

    t_redact_start = time.time()
    redacted = bgr.copy()
    for (x1,y1,x2,y2) in merged:
        x1=max(0,int(x1)); y1=max(0,int(y1)); x2=min(w,int(x2)); y2=min(h,int(y2))
        redacted[y1:y2, x1:x2] = 0
    t_redact = time.time() - t_redact_start

    if has_alpha:
        out = np.dstack([redacted, img[:,:,3]])
    else:
        out = redacted

    t_save_start = time.time()
    _, ext = os.path.splitext(path)
    if ext.lower() in ['.jpg','.jpeg']:
        cv2.imwrite(path, out, [cv2.IMWRITE_JPEG_QUALITY, 100])
    else:
        cv2.imwrite(path, out, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    t_save = time.time() - t_save_start

    elapsed = t_read + t_ocr + t_redact + t_save
    return {
        "name": fname,
        "read": t_read,
        "ocr": t_ocr,
        "redact": t_redact,
        "save": t_save,
        "total": elapsed,
        "n_redacted": len(merged)
    }

def format_elapsed(seconds: float) -> str:
    if seconds < 0: seconds = 0
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h>0: return f"{h}h {m}m {s}s"
    if m>0: return f"{m}m {s}s"
    return f"{s}s"

def main():
    files = sorted([os.path.join(FRAMES_DIR,f) for f in os.listdir(FRAMES_DIR) if f.lower().endswith(('.png','.jpg','.jpeg'))])
    if not files:
        print("No frames found in", FRAMES_DIR); return

    total_frames = len(files)
    print(f"Total frames: {total_frames}")
    start_all = time.time()
    done = 0
    for path in files:
        res = process_file(path, reader)
        if res is None:
            continue
        done += 1
        elapsed_total = time.time() - start_all
        fps = done / elapsed_total if elapsed_total>0 else 0
        eta = (total_frames - done) / fps if fps>0 else 0
        print(
            f"{done}/{total_frames} | {res['name']} | Total: {res['total']:.2f}s | "
            f"Read: {res['read']:.2f}s | OCR: {res['ocr']:.2f}s | Redact: {res['redact']:.2f}s | "
            f"Save: {res['save']:.2f}s | Redacted: {res['n_redacted']} | FPS: {fps:.2f} | "
            f"Elapsed: {format_elapsed(elapsed_total)} | ETA: {format_elapsed(eta)}"
        )

    total_time = time.time() - start_all
    avg_fps = total_frames / total_time if total_time>0 else 0
    print(f"\nDone. Total time: {format_elapsed(total_time)} | Average FPS: {avg_fps:.2f}")

if __name__ == "__main__":
    main()
