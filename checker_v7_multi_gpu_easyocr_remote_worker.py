import os
import re
import cv2
import time
import queue
import signal
import hashlib
import tempfile
import traceback
import numpy as np
import multiprocessing as mp
from multiprocessing import Manager

FRAMES_DIR = "frames_upscaled"
OCR_MIN_CONF = 0.45
BOX_PAD = 6
MERGE_IOU = 0.15

# If True, skip frames that were already processed earlier (tracked by size + mtime).
RESUME_MODE = True

# If True, skip frames that look already blacked-out for matched OCR regions.
SKIP_ALREADY_BLACKED = False

# Log file for failed frames
FAILED_LOG_TXT = "failed_frames.txt"

KEYWORDS = [
    'stake.com', 'stake .com.', 'stake.com.', 'https://', 'Stake',
    'stakecom', 'stake', 'Stake Originals', 'Only on Stake',
    'casino', 'live casino', 'slots', 'blackjack', 'baccarat',
    'Roulette', 'ustake', 'gamble', 'betting', 'bet', 'gambling', 'online gambling'
]
KEYWORDS_LOWER = [k.lower() for k in KEYWORDS]

# ---------------------------
# Globals initialized per worker
# ---------------------------
_worker_reader = None
_worker_device_id = None
_worker_stop_event = None


def get_processed_db_path(worker_id):
    """ Returns the dedicated processed tracking file for a specific worker. """
    return f"processed_frames_worker_{worker_id}.txt"


def normalize_text(t):
    return (t or "").lower().strip()


def text_matches(t):
    nt = normalize_text(t)
    for kw in KEYWORDS_LOWER:
        if kw in nt:
            return True
    return False


def expand_box(box, w, h, pad=6):
    xs = [int(round(p[0])) for p in box]
    ys = [int(round(p[1])) for p in box]
    x1 = max(0, min(xs) - pad)
    y1 = max(0, min(ys) - pad)
    x2 = min(w, max(xs) + pad)
    y2 = min(h, max(ys) + pad)
    return x1, y1, x2, y2


def merge_rects(rects, iou_thresh=0.15):
    if not rects:
        return []

    boxes = np.array(rects)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = areas.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append((int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])))

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w_ = np.maximum(0.0, xx2 - xx1)
        h_ = np.maximum(0.0, yy2 - yy1)
        inter = w_ * h_
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]

    return keep


def format_elapsed(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def frame_signature(path):
    """
    Signature used for resume mode.
    Uses path + file size + mtime_ns.
    """
    try:
        st = os.stat(path)
        return f"{path}|{st.st_size}|{st.st_mtime_ns}"
    except FileNotFoundError:
        return None


def load_processed_db(txt_path):
    if not RESUME_MODE or not os.path.exists(txt_path):
        return set()
    processed = set()
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                processed.add(s)
    return processed


def append_processed_db(txt_path, signature):
    with open(txt_path, "a", encoding="utf-8") as f:
        f.write(signature + "\n")


def append_failed_log(txt_path, path, error_text):
    with open(txt_path, "a", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write(f"FILE: {path}\n")
        f.write(f"TIME: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("ERROR:\n")
        f.write(error_text.rstrip() + "\n\n")


def is_region_already_black(img_region):
    if img_region.size == 0:
        return True
    return np.max(img_region) <= 3


def init_worker(device_id, stop_event):
    """
    Worker initializer: set CUDA device visibility and create EasyOCR reader inside the worker.
    """
    global _worker_reader, _worker_device_id, _worker_stop_event
    _worker_device_id = str(device_id).lower()
    _worker_stop_event = stop_event

    import easyocr  # import inside worker

    if _worker_device_id == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        _worker_reader = easyocr.Reader(['en'], gpu=False)
    else:
        # Must be set before torch/easyocr initialize CUDA in this worker
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
        _worker_reader = easyocr.Reader(['en'], gpu=True)


def process_file(path):
    """
    Uses per-worker global reader.
    Returns a result dict or raises exception.
    """
    global _worker_reader, _worker_device_id, _worker_stop_event

    if _worker_stop_event is not None and _worker_stop_event.is_set():
        return None

    fname = os.path.basename(path)
    t_read_start = time.time()
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    t_read = time.time() - t_read_start

    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")

    has_alpha = (img.ndim == 3 and img.shape[2] == 4)
    bgr = img[:, :, :3] if has_alpha else img
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    t_ocr_start = time.time()
    results = _worker_reader.readtext(rgb, detail=1, paragraph=False)
    t_ocr = time.time() - t_ocr_start

    rects = []
    for bbox, text, prob in results:
        if prob >= OCR_MIN_CONF and text_matches(text):
            rects.append(expand_box(bbox, w, h, pad=BOX_PAD))

    merged = merge_rects(rects, iou_thresh=MERGE_IOU)

    # Optional skip if everything already looks blacked out
    if SKIP_ALREADY_BLACKED and merged:
        all_black = True
        for (x1, y1, x2, y2) in merged:
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(w, int(x2))
            y2 = min(h, int(y2))
            if not is_region_already_black(bgr[y1:y2, x1:x2]):
                all_black = False
                break
        if all_black:
            elapsed = t_read + t_ocr
            return {
                "name": fname,
                "read": t_read,
                "ocr": t_ocr,
                "redact": 0.0,
                "save": 0.0,
                "total": elapsed,
                "n_redacted": len(merged),
                "skipped": True,
                "device_id": _worker_device_id,
            }

    t_redact_start = time.time()
    redacted = bgr.copy()
    for (x1, y1, x2, y2) in merged:
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(w, int(x2))
        y2 = min(h, int(y2))
        redacted[y1:y2, x1:x2] = 0
    t_redact = time.time() - t_redact_start

    if has_alpha:
        out = np.dstack([redacted, img[:, :, 3]])
    else:
        out = redacted

    # Atomic overwrite: write temp file, then replace original
    t_save_start = time.time()
    _, ext = os.path.splitext(path)
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=ext, dir=os.path.dirname(path))
    os.close(tmp_fd)

    try:
        if ext.lower() in ['.jpg', '.jpeg']:
            ok = cv2.imwrite(tmp_path, out, [cv2.IMWRITE_JPEG_QUALITY, 100])
        else:
            ok = cv2.imwrite(tmp_path, out, [cv2.IMWRITE_PNG_COMPRESSION, 2])

        if not ok:
            raise RuntimeError(f"cv2.imwrite failed for temp file: {tmp_path}")

        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    t_save = time.time() - t_save_start
    elapsed = t_read + t_ocr + t_redact + t_save

    return {
        "name": fname,
        "read": t_read,
        "ocr": t_ocr,
        "redact": t_redact,
        "save": t_save,
        "total": elapsed,
        "n_redacted": len(merged),
        "skipped": False,
        "device_id": _worker_device_id,
    }


def list_available_devices():
    """
    Returns a list of tuples: (device_id_string, device_name)
    Always includes CPU. Probes PyTorch for available GPUs.
    """
    devices = [("cpu", "Central Processing Unit")]
    try:
        import torch
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            for i in range(count):
                try:
                    name = torch.cuda.get_device_name(i)
                except Exception:
                    name = f"GPU {i}"
                devices.append((str(i), name))
    except Exception as e:
        print("PyTorch import/GPU check failed:", e)
        print("Defaulting to CPU only.")
    return devices


def parse_device_selection(user_input, available_ids):
    """ Parse device IDs like 'cpu', '0', 'cpu, 0' """
    s = user_input.strip().lower()
    if s == "all":
        return list(available_ids)

    parts = re.split(r"[,\s]+", s)
    picked = []
    for p in parts:
        if not p:
            continue
        if p not in available_ids:
            raise ValueError(f"Device '{p}' is not available.")
        if p not in picked:
            picked.append(p)
    if not picked:
        raise ValueError("No valid selection made.")
    return picked


def parse_worker_selection(user_input, total_workers):
    """ Parse worker integers like '1', '2' up to total_workers """
    s = user_input.strip().lower()
    if s == "all":
        return list(range(1, total_workers + 1))

    parts = re.split(r"[,\s]+", s)
    picked = []
    for p in parts:
        if not p:
            continue
        if not p.isdigit():
            raise ValueError(f"Invalid worker ID: {p}")
        val = int(p)
        if val < 1 or val > total_workers:
            raise ValueError(f"Worker ID {val} is out of bounds (must be 1 to {total_workers}).")
        if val not in picked:
            picked.append(val)
    if not picked:
        raise ValueError("No valid worker IDs selected.")
    return picked


def worker_loop(device_id, tasks, result_queue, stop_event):
    """
    Each worker owns exactly one device (CPU or a specific GPU) and processes its own list of frames.
    """
    try:
        init_worker(device_id, stop_event)

        for global_idx, path in tasks:
            if stop_event.is_set():
                break

            try:
                res = process_file(path)
                if res is None:
                    break
                res["global_idx"] = global_idx
                res["path"] = path
                result_queue.put(("ok", res))
            except Exception:
                err = traceback.format_exc()
                result_queue.put(("err", {
                    "device_id": device_id,
                    "global_idx": global_idx,
                    "path": path,
                    "error": err
                }))
    except Exception:
        err = traceback.format_exc()
        result_queue.put(("fatal", {
            "device_id": device_id,
            "error": err
        }))


def main():
    files = sorted([
        os.path.join(FRAMES_DIR, f)
        for f in os.listdir(FRAMES_DIR)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    if not files:
        print("No frames found in", FRAMES_DIR)
        return

    original_total = len(files)

    devices = list_available_devices()
    print("\nAvailable Devices:")
    for dev_id, dev_name in devices:
        print(f"  [{dev_id}] {dev_name}")

    available_ids = [dev_id for dev_id, _ in devices]

    # 1. Ask for local devices to use
    while True:
        choice = input("\nSelect LOCAL device(s) to use (example: cpu or 0 or cpu,0 or all): ").strip()
        try:
            selected_device_ids = parse_device_selection(choice, available_ids)
            break
        except ValueError as e:
            print("Invalid selection:", e)

    # 2. Ask for total number of global workers across all PCs
    while True:
        try:
            total_workers_input = input("\nEnter TOTAL number of global workers across all machines (e.g., 3): ").strip()
            total_workers = int(total_workers_input)
            if total_workers < 1:
                raise ValueError("Must be at least 1.")
            break
        except ValueError as e:
            print(f"Invalid input: {e}")

    # 3. Ask for the specific global worker IDs for the selected local devices
    while True:
        choice = input(f"Enter the {len(selected_device_ids)} global worker ID(s) for this machine (1 to {total_workers}, example: 1 or 1,2): ").strip()
        try:
            assigned_worker_ids = parse_worker_selection(choice, total_workers)
            
            if len(assigned_worker_ids) != len(selected_device_ids):
                raise ValueError(f"You selected {len(selected_device_ids)} device(s), but provided {len(assigned_worker_ids)} worker IDs.")
            
            break
        except ValueError as e:
            print(f"Invalid selection: {e}")

    # Load only the processed databases for the workers assigned to this machine
    worker_processed_dbs = {}
    if RESUME_MODE:
        for w_id in assigned_worker_ids:
            db_path = get_processed_db_path(w_id)
            worker_processed_dbs[w_id] = load_processed_db(db_path)

    # 4. Distribute files deterministically using modulo logic across the global cluster
    assignments = {did: [] for did in selected_device_ids}
    skipped_resume = 0

    for idx, path in enumerate(files):
        # global_worker_id logic (1-based mapping)
        global_worker_id = (idx % total_workers) + 1
        
        # If this frame belongs to one of our assigned workers
        if global_worker_id in assigned_worker_ids:
            # Map the global worker ID to the correct local device
            local_idx = assigned_worker_ids.index(global_worker_id)
            local_device = selected_device_ids[local_idx]
            
            # Now check resume mode specifically against this worker's DB
            if RESUME_MODE:
                sig = frame_signature(path)
                if sig and sig in worker_processed_dbs[global_worker_id]:
                    skipped_resume += 1
                    continue
            
            assignments[local_device].append((idx, path))

    # Calculate how many frames this machine is processing
    machine_total_tasks = sum(len(assignments[did]) for did in selected_device_ids)

    print("\n=== Startup ===")
    print(f"Frames directory: {FRAMES_DIR}")
    print(f"Total frames globally: {original_total}")
    print(f"Total global workers: {total_workers}")
    print(f"This machine handles worker IDs: {assigned_worker_ids}")
    if RESUME_MODE:
        print(f"Resume mode: skipped {skipped_resume} already-processed frames assigned to this machine.")
    print(f"Frames to process NOW on this machine: {machine_total_tasks}")
    
    for did, w_id in zip(selected_device_ids, assigned_worker_ids):
        print(f"  Local Device [{did}] (Global Worker {w_id}): {len(assignments[did])} frame(s)")
    
    print("Processing mode: in-place overwrite")
    print("Press Ctrl+C to stop safely.\n")

    if machine_total_tasks == 0:
        print("Nothing left to process for this machine.")
        return

    manager = Manager()
    result_queue = manager.Queue()
    stop_event = manager.Event()

    workers = []
    for did in selected_device_ids:
        p = mp.Process(
            target=worker_loop,
            args=(did, assignments[did], result_queue, stop_event),
            daemon=False
        )
        p.start()
        workers.append(p)

    done = 0
    failed = 0
    start_all = time.time()

    # Per-device counters
    device_done = {did: 0 for did in selected_device_ids}
    device_total = {did: len(assignments[did]) for did in selected_device_ids}

    # Count of messages expected from workers running on this machine
    expected = machine_total_tasks
    received = 0

    try:
        while received < expected:
            try:
                kind, payload = result_queue.get(timeout=0.5)
            except queue.Empty:
                # Check if all workers died early
                if not any(p.is_alive() for p in workers):
                    break
                continue

            received += 1

            if kind == "ok":
                res = payload
                did = res["device_id"]
                global_idx = res["global_idx"]
                
                done += 1
                device_done[did] += 1

                # Calculate which global worker this was to save to the correct TXT file
                completed_worker_id = (global_idx % total_workers) + 1

                # save resume signature after successful processing / skip
                sig = frame_signature(res["path"])
                if sig:
                    db_path = get_processed_db_path(completed_worker_id)
                    append_processed_db(db_path, sig)

                elapsed_total = time.time() - start_all
                fps = done / elapsed_total if elapsed_total > 0 else 0
                eta = (machine_total_tasks - done) / fps if fps > 0 else 0

                global_index_1based = global_idx + 1
                tag = "SKIPPED" if res.get("skipped") else f"Redacted: {res['n_redacted']}"

                print(
                    f"device[{did}]: GlobalFrame {global_index_1based}/{original_total} | {res['name']} | "
                    f"Total: {res['total']:.2f}s | Read: {res['read']:.2f}s | OCR: {res['ocr']:.2f}s | "
                    f"Redact: {res['redact']:.2f}s | Save: {res['save']:.2f}s | {tag} | "
                    f"FPS: {fps:.2f} | Elapsed: {format_elapsed(elapsed_total)} | ETA: {format_elapsed(eta)}"
                )

            elif kind == "err":
                failed += 1
                did = payload["device_id"]
                path = payload["path"]
                err = payload["error"]

                append_failed_log(FAILED_LOG_TXT, path, err)
                print(f"device[{did}]: ERROR processing {os.path.basename(path)} (logged to {FAILED_LOG_TXT})")

            elif kind == "fatal":
                did = payload["device_id"]
                err = payload["error"]
                append_failed_log(FAILED_LOG_TXT, f"[WORKER DEVICE {did} FATAL]", err)
                print(f"device[{did}]: FATAL worker error (logged to {FAILED_LOG_TXT})")
                stop_event.set()

        total_time = time.time() - start_all
        avg_fps = done / total_time if total_time > 0 else 0

        print("\nDone.")
        print(f"Processed successfully: {done}")
        print(f"Failed: {failed}")
        print(f"Total time: {format_elapsed(total_time)} | Average FPS: {avg_fps:.2f}")

        print("\nPer-Device summary:")
        for did in selected_device_ids:
            print(f"  Device [{did}]: {device_done[did]}/{device_total[did]} frames completed")

        if failed > 0:
            print(f"\nFailed files were written to: {FAILED_LOG_TXT}")

    except KeyboardInterrupt:
        print("\nCtrl+C received. Stopping workers safely...")
        stop_event.set()

    finally:
        for p in workers:
            p.join(timeout=10)

        # Force terminate if any are still hanging
        for p in workers:
            if p.is_alive():
                p.terminate()
                p.join(timeout=2)

        print("Shutdown complete.")


if __name__ == "__main__":
    mp.freeze_support()  # important for Windows
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()