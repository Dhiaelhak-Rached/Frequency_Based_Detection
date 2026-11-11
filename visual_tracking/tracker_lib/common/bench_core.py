"""
bench_core.py - shared code for all tracker runs.

  * sequences: a standard description of one test video (sequence.json + gt.txt), see load_sequence()
  * frame reading, streamed one frame at a time (AOT frames are 2448x2048: never load a whole sequence)
  * the run loop: initial box -> update on every frame -> optional re-initialisation (simulated detector)
  * the live window: initial / ground-truth box in GREEN, tracker output in RED, re-initialisation in YELLOW
  * metrics and result files

Box convention everywhere: (x, y, w, h) in pixels, (x, y) = top-left corner.
Works on Windows, Linux and macOS.
"""
import csv
import glob
import json
import math
import os
import statistics
import sys
import time

import numpy as np

try:
    import cv2
    if not hasattr(cv2, "imread"):  # half-installed package
        cv2 = None
except ImportError:
    cv2 = None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


# Big datasets can live on another drive (e.g. a USB flash drive). Paths into it are stored as "{data_drive}/..." and
# "data_drive" in config.json says where it is, so a new drive letter (or a Jetson mount point) is a one-line change.
DRIVE_TAG = "{data_drive}"
_DATA_DRIVE = []


def data_drive():
    """The data drive folder from config.json ("data_drive"), or None."""
    if not _DATA_DRIVE:
        d = load_config().get("data_drive")
        _DATA_DRIVE.append(os.path.normpath(os.path.expanduser(d)) if d else None)
    return _DATA_DRIVE[0]


def rootpath(p):
    """Resolve a path from config.json / sequence.json: "{data_drive}/..." is on the data drive, absolute stays,
    relative is taken from the project root."""
    if p is None:
        return None
    if p.startswith(DRIVE_TAG):
        base = data_drive()
        if not base:
            sys.exit(f'{p} is on the data drive, but "data_drive" is not set in config.json')
        return os.path.join(base, *[s for s in p[len(DRIVE_TAG):].replace("\\", "/").split("/") if s])
    p = os.path.expanduser(p)
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(ROOT, p))


def load_config():
    path = os.path.join(ROOT, "config.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


# ============================================================ tracker interface
class BaseTracker:
    """Every tracker implements these two methods."""
    name = "base"
    device = "cpu"  # where it runs, copied into every result summary (GPU trackers set e.g. "cuda (<GPU name>)")

    def init(self, frame, box):
        """frame: BGR uint8 image; box: (x, y, w, h) of the target in this frame."""
        raise NotImplementedError

    def update(self, frame):
        """Return (ok, (x, y, w, h), score_or_None) for the new frame."""
        raise NotImplementedError


# ============================================================ sequences
def read_boxes(path):
    """Ground truth: one 'x,y,w,h' line per frame; NaN, empty or zero size = target not visible."""
    boxes = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                boxes.append(None)
                continue
            parts = line.replace(",", " ").replace("\t", " ").split()
            try:
                vals = [float(p) for p in parts[:4]]
            except ValueError:
                boxes.append(None)
                continue
            if len(vals) < 4 or any(math.isnan(v) for v in vals) or vals[2] <= 0 or vals[3] <= 0:
                boxes.append(None)
            else:
                boxes.append(tuple(vals))
    return boxes


def write_boxes(path, boxes):
    with open(path, "w") as f:
        for b in boxes:
            f.write("nan,nan,nan,nan\n" if b is None else f"{b[0]:.2f},{b[1]:.2f},{b[2]:.2f},{b[3]:.2f}\n")


def portable(p):
    """Store paths on the data drive as "{data_drive}/...", paths inside the project relative to it (so the folder can
    be moved), others absolute."""
    if p is None:
        return None
    ap = os.path.abspath(p)
    dd = data_drive()
    if dd and os.path.normcase(ap).startswith(os.path.normcase(dd) + os.sep):
        return DRIVE_TAG + "/" + os.path.relpath(ap, dd).replace(os.sep, "/")
    try:
        rel = os.path.relpath(ap, ROOT)
    except ValueError:  # different drive on Windows
        return ap
    return ap if rel.startswith("..") else rel.replace(os.sep, "/")


def save_sequence(folder, name, boxes, video=None, images=None, frames_dir=None, extra=None, zip_path=None):
    """Write a standard sequence: folder/sequence.json + folder/gt.txt (one x,y,w,h line per frame).
    With zip_path, "video" / "images" are member names inside that zip (the dataset is used without extracting it)."""
    frames_dir = portable(frames_dir)
    if zip_path:
        images = list(images) if images else None
    else:
        video = portable(video)
        images = [portable(p) for p in images] if images else None
    os.makedirs(folder, exist_ok=True)
    write_boxes(os.path.join(folder, "gt.txt"), boxes)
    meta = {"name": name, "video": video, "frames_dir": frames_dir, "images": images, "gt": "gt.txt"}
    if zip_path:
        meta["zip"] = portable(zip_path)
    if extra:
        meta.update(extra)
    with open(os.path.join(folder, "sequence.json"), "w") as f:
        json.dump(meta, f, indent=1)


def load_sequence(folder):
    with open(os.path.join(folder, "sequence.json")) as f:
        meta = json.load(f)
    meta["folder"] = folder
    # stored paths may be relative to the project root (portable) or absolute
    meta["zip"] = rootpath(meta.get("zip"))
    if not meta["zip"]:  # inside a zip, "video" / "images" are member names: keep them as they are
        meta["video"] = rootpath(meta.get("video"))
        if meta.get("images"):
            meta["images"] = [rootpath(p) for p in meta["images"]]
    meta["frames_dir"] = rootpath(meta.get("frames_dir"))
    gt_path = os.path.join(folder, meta.get("gt") or "gt.txt")
    meta["boxes"] = read_boxes(gt_path) if os.path.exists(gt_path) else []
    src = meta["zip"] or meta["video"] or meta["frames_dir"] or (meta["images"][0] if meta.get("images") else None)
    if src and not os.path.exists(src):
        dd = data_drive()
        if dd and os.path.normcase(src).startswith(os.path.normcase(dd)):
            sys.exit(f"{meta.get('name')}: its images are on the data drive ({dd}), which is not available.\n"
                     f'Plug in the flash drive. If Windows gave it another letter, change "data_drive" in config.json.')
        sys.exit(f"{meta.get('name')}: {src} not found")
    return meta


def list_sequences(dataset_dir):
    return sorted(os.path.dirname(p) for p in glob.glob(os.path.join(dataset_dir, "**", "sequence.json"),
                                                        recursive=True))


def iter_frames(video=None, frames_dir=None, images=None, zip_path=None):
    """Yield BGR uint8 frames one by one from a video file, an image folder, an explicit image list, or from a zip
    file (zip_path + member names: big datasets can be used without extracting them). A video inside a zip is copied
    to a temporary file first (OpenCV can't play from a zip) and deleted afterwards; that copy isn't timed."""
    if zip_path and video:
        import shutil
        import tempfile
        import zipfile
        tmp = tempfile.NamedTemporaryFile(suffix=os.path.splitext(video)[1] or ".mp4", delete=False)
        try:
            with zipfile.ZipFile(zip_path) as z, z.open(video) as src:
                shutil.copyfileobj(src, tmp, 16 << 20)
            tmp.close()
            yield from iter_frames(video=tmp.name)
        finally:
            tmp.close()
            try:
                os.remove(tmp.name)
            except OSError:
                pass
        return
    if zip_path:
        import zipfile
        with zipfile.ZipFile(zip_path) as z:
            for name in images:
                data = np.frombuffer(z.read(name), np.uint8)
                if cv2 is not None:
                    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
                else:
                    import io
                    from PIL import Image
                    img = np.asarray(Image.open(io.BytesIO(data.tobytes())).convert("RGB"))[:, :, ::-1].copy()
                if img is None:
                    sys.exit(f"Cannot read {name} in {zip_path}")
                yield img
        return
    if video:
        if cv2 is None:
            sys.exit("Reading a video needs OpenCV: pip install opencv-contrib-python")
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            sys.exit(f"Cannot open video {video}")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
        cap.release()
        return
    if images is None:
        images = sorted(f for f in glob.glob(os.path.join(frames_dir, "*")) if f.lower().endswith(IMG_EXT))
        if not images:
            sys.exit(f"No images found in {frames_dir}")
    for f in images:
        if cv2 is not None:
            img = cv2.imread(f, cv2.IMREAD_COLOR)  # grayscale files (AOT) are expanded to 3 channels
        else:
            from PIL import Image
            img = np.asarray(Image.open(f).convert("RGB"))[:, :, ::-1].copy()
        if img is None:
            sys.exit(f"Cannot read image {f}")
        yield img


def sequence_frames(seq):
    return iter_frames(video=seq.get("video"), frames_dir=seq.get("frames_dir"), images=seq.get("images"),
                       zip_path=seq.get("zip"))


def first_frame(seq):
    for fr in sequence_frames(seq):
        return fr
    sys.exit("Empty sequence")


# ============================================================ metrics
def iou(a, b):
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def centre_err(a, b):
    return math.hypot(a[0] + a[2] / 2 - b[0] - b[2] / 2, a[1] + a[3] / 2 - b[1] - b[3] / 2)


def is_lost(row, lost_score):
    low = row["score"] is not None and lost_score is not None and row["score"] < lost_score
    return (not row["ok"]) or low


def summarise(name, rows, gt, lost_score, warmup=0):
    """rows: dicts with frame, x, y, w, h, ok, score, ms, reinit. The start frame (ms None) is excluded."""
    track_rows = [r for r in rows if r["ms"] is not None and not r.get("reinit")]
    start = rows[0]["frame"] if rows else 0
    ms = [r["ms"] for r in track_rows if r["frame"] - start > warmup]
    out = {
        "tracker": name,
        "frames": len(track_rows),
        "fps": round(1000.0 / statistics.mean(ms), 1) if ms else "",
        "mean_ms": round(statistics.mean(ms), 2) if ms else "",
        "median_ms": round(statistics.median(ms), 2) if ms else "",
        "p95_ms": round(float(np.percentile(ms, 95)), 2) if ms else "",
        "max_ms": round(max(ms), 2) if ms else "",
        "lost_frames": sum(1 for r in track_rows if is_lost(r, lost_score)),
        "reinits": sum(1 for r in rows if r.get("reinit")),
    }
    if gt:
        ious, errs, fails, prev_ok = [], [], 0, True
        for r in track_rows:
            i = r["frame"]
            if i >= len(gt) or gt[i] is None:
                continue
            pred = (r["x"], r["y"], r["w"], r["h"])
            v = iou(pred, gt[i]) if r["ok"] else 0.0
            ious.append(v)
            errs.append(centre_err(pred, gt[i]) if r["ok"] else float("inf"))
            if prev_ok and v == 0.0:
                fails += 1
            prev_ok = v > 0.0
        if ious:
            thr = np.linspace(0, 1, 21)
            arr = np.array(ious)
            out["gt_frames"] = len(ious)
            out["mean_iou"] = round(float(arr.mean()), 3)
            out["success_auc"] = round(float(np.mean([np.mean(arr > t) for t in thr])), 3)
            out["precision_20px"] = round(float(np.mean(np.array(errs) <= 20)), 3)
            out["failures"] = fails
    return out


SUMMARY_KEYS = ["tracker", "device", "frames", "fps", "mean_ms", "median_ms", "p95_ms", "max_ms", "init_ms",
                "lost_frames", "reinits", "mean_iou", "success_auc", "precision_20px", "failures"]


def md_table(summaries, keys=None):
    keys = keys or SUMMARY_KEYS
    lines = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    for s in summaries:
        lines.append("| " + " | ".join(str(s.get(k, "")) for k in keys) + " |")
    return "\n".join(lines)


# ============================================================ result files
def write_rows(path, rows):
    with open(path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["frame", "x", "y", "w", "h", "ok", "score", "ms", "reinit"])
        for r in rows:
            wr.writerow([r["frame"], f"{r['x']:.1f}", f"{r['y']:.1f}", f"{r['w']:.1f}", f"{r['h']:.1f}",
                         int(r["ok"]), "" if r["score"] is None else f"{r['score']:.3f}",
                         "" if r["ms"] is None else f"{r['ms']:.3f}", int(bool(r.get("reinit")))])


def read_rows(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            def g(k):
                v = r.get(k, "")
                return float(v) if v not in ("", None) else None
            x, y, w, h = g("x"), g("y"), g("w"), g("h")
            ok = (r.get("ok", "") != "0") and not (x is None or w is None or w <= 0 or h is None or h <= 0)
            rows.append(dict(frame=int(float(r["frame"])), x=x or 0.0, y=y or 0.0, w=w or 0.0, h=h or 0.0,
                             ok=ok, score=g("score"), ms=g("ms"), reinit=r.get("reinit", "0") == "1"))
    return rows


def write_results(out_dir, summary, rows):
    os.makedirs(out_dir, exist_ok=True)
    write_rows(os.path.join(out_dir, "per_frame.csv"), rows)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write(md_table([summary]) + "\n")


# ============================================================ drawing and live window
GREEN, RED, YELLOW, WHITE = (0, 200, 0), (0, 0, 255), (0, 220, 255), (255, 255, 255)
# Right-arrow codes from cv2.waitKeyEx: Windows, Linux GTK, Linux Qt, macOS
RIGHT_ARROW = (0x270000, 0xFF53, 0x1000014, 0xF703)


def _rect(img, box, col, t):
    x, y, w, h = [int(round(v)) for v in box]
    cv2.rectangle(img, (x, y), (x + w, y + h), col, t)


def annotate(frame, idx, gt_box, row, tracker_name, ms, initial=False, reinit=False, lost=False, status=""):
    """Return a copy of the frame with: GREEN = initial box / ground truth, RED = tracker output."""
    img = frame.copy()
    th = max(1, img.shape[1] // 640)
    if gt_box is not None:
        _rect(img, gt_box, GREEN, th + (1 if initial else 0))
    if row is not None and row["ok"] and not initial:
        _rect(img, (row["x"], row["y"], row["w"], row["h"]), YELLOW if reinit else RED, th + 1)
    s = 0.5 * th
    y0 = int(22 * th)
    if initial:
        cv2.putText(img, "GREEN = initial box given to the tracker", (10, y0), cv2.FONT_HERSHEY_SIMPLEX, s, GREEN, th)
    else:
        cv2.putText(img, f"RED = {tracker_name}   GREEN = ground truth", (10, y0), cv2.FONT_HERSHEY_SIMPLEX, s,
                    WHITE, th)
        info = f"frame {idx}   {ms:.1f} ms" if ms is not None else f"frame {idx}"
        if row is not None and row.get("score") is not None:
            info += f"   score {row['score']:.2f}"
        if lost:
            info += "   LOST"
        if reinit:
            info += "   RE-INITIALISED (simulated detector)"
        cv2.putText(img, info, (10, 2 * y0), cv2.FONT_HERSHEY_SIMPLEX, s, RED if lost else WHITE, th)
    if status:
        cv2.putText(img, status, (10, 3 * y0), cv2.FONT_HERSHEY_SIMPLEX, s, WHITE, th)
    return img


class LiveWindow:
    """Shows the test while it runs. Keys: SPACE pause/resume, N or RIGHT ARROW next sequence, Q quit all,
    H hide window."""

    def __init__(self, enabled, title, max_width=1280):
        self.enabled = enabled and cv2 is not None
        if enabled and cv2 is None:
            print("(no live window: OpenCV missing)")
        self.title = title
        self.max_width = max_width
        self.paused = False

    def show(self, img, wait_ms=1):
        """Return 'next', 'quit' or None."""
        if not self.enabled:
            return None
        if img.shape[1] > self.max_width:
            f = self.max_width / img.shape[1]
            img = cv2.resize(img, (int(img.shape[1] * f), int(img.shape[0] * f)))
        cv2.imshow(self.title, img)
        while True:
            key = cv2.waitKeyEx(0 if self.paused else wait_ms)  # full code: arrows are lost with & 0xFF
            if key in RIGHT_ARROW:
                return "next"
            k = key & 0xFF if key != -1 else 0xFF
            if k in (ord("q"), ord("Q"), 27):
                return "quit"
            if k in (ord("n"), ord("N")):
                return "next"
            if k in (ord("h"), ord("H")):
                self.close()
                self.enabled = False
                return None
            if k == ord(" "):
                self.paused = not self.paused
                if self.paused:
                    continue
            if not self.paused:
                return None

    def close(self):
        if cv2 is not None:
            try:
                cv2.destroyWindow(self.title)
            except Exception:
                pass


# ============================================================ the run loop
def run_sequence(tracker, frames, gt, init_box=None, lost_score=None, reinit_after=0, max_frames=0,
                 window=None, video_path=None, fps=30, detector_every=0, detector_iou=0.3, playback_speed=1.0):
    """Run one tracker on one sequence (frames = iterator of BGR images).

    Starts at the first frame that has a ground-truth box (or frame 0 with init_box).
    Two simulated detectors can restart the tracker from the true box (YELLOW in the window):
      * detector_every = N: a detector checks the tracker's box every N frames. If the box overlaps the true box
        by less than detector_iou (IoU), the tracker is restarted. The detector is perfect (it uses the ground truth:
        no false alarms, no misses) and instant. N = 0: no periodic detector.
      * reinit_after = K: an oracle that restarts the tracker after K bad frames in a row (watches every frame).
    playback_speed: how fast the live window plays: 1 = real time (the video's fps), 2 = twice as fast, 0 = as fast as
    the computer can. It only waits, so a tracker slower than the video still plays slower than real time.
    Returns (rows, init_ms, stop) where stop is None, 'next' or 'quit'.
    Only tracker.init/update are timed; drawing and file writing are not.
    """
    writer = None
    if cv2 is None:  # no drawing possible: still run and measure
        video_path = None
        if window is not None:
            window.enabled = False
    rows, init_ms, stop = [], None, None
    start = None
    bad = 0
    has_gt = bool(gt) and any(b is not None for b in gt)
    if not has_gt:
        detector_every = reinit_after = 0  # nothing to simulate a detector with
    since = checks = restarts = 0
    frame_dt = 1.0 / (fps * playback_speed) if (fps and playback_speed and playback_speed > 0) else 0.0
    next_t = None  # when the next frame should appear in the window

    def pace_ms():
        """Milliseconds to wait before the next frame so the window plays at the video's own speed."""
        nonlocal next_t
        if not frame_dt:
            return 1
        now = time.perf_counter()
        if next_t is None or now - next_t > 0.25:  # first frame, after a pause, or too slow to keep up: restart the clock
            next_t = now
        next_t += frame_dt
        return max(1, int((next_t - now) * 1000))

    def status_text():
        if detector_every:
            return f"simulated detector every {detector_every} frames | checks {checks}, restarts {restarts}"
        if reinit_after:
            return f"simulated detector: restart after {reinit_after} bad frames | restarts {restarts}"
        return "no detector help: a lost tracker stays lost" if has_gt else "no ground truth: no detector, no scoring"

    for i, frame in enumerate(frames):
        if max_frames and start is not None and i - start >= max_frames:
            break
        g = gt[i] if gt and i < len(gt) else None
        if start is None:  # waiting for the first frame with a box
            box0 = init_box if (init_box is not None and i == 0) else g
            if box0 is None:
                continue
            start = i
            t0 = time.perf_counter()
            tracker.init(frame, box0)
            init_ms = (time.perf_counter() - t0) * 1000
            rows.append(dict(frame=i, x=box0[0], y=box0[1], w=box0[2], h=box0[3], ok=True, score=None, ms=None,
                             reinit=False))
            img = None
            if window is not None and window.enabled or video_path:
                img = annotate(frame, i, box0, None, tracker.name, None, initial=True, status=status_text())
            if video_path and cv2 is not None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                writer.write(img)
            if window is not None:
                stop = window.show(img, wait_ms=700)  # hold the initial box on screen briefly
                next_t = None
                if stop:
                    break
            continue
        t0 = time.perf_counter()
        ok, box, score = tracker.update(frame)
        ms = (time.perf_counter() - t0) * 1000
        row = dict(frame=i, x=box[0], y=box[1], w=box[2], h=box[3], ok=bool(ok), score=score, ms=ms, reinit=False)
        rows.append(row)
        lost = is_lost(row, lost_score)
        did_reinit = False
        if reinit_after and g is not None:
            bad = bad + 1 if (lost or iou(box, g) == 0.0) else 0
            if bad >= reinit_after:
                tracker.init(frame, g)
                rows.append(dict(frame=i, x=g[0], y=g[1], w=g[2], h=g[3], ok=True, score=None, ms=None, reinit=True))
                bad, did_reinit = 0, True
                restarts += 1
        since += 1
        if detector_every and not did_reinit and since % detector_every == 0:
            checks += 1
            if g is not None and iou(box, g) < detector_iou:  # the detector sees the drone elsewhere: restart on it
                tracker.init(frame, g)
                rows.append(dict(frame=i, x=g[0], y=g[1], w=g[2], h=g[3], ok=True, score=None, ms=None, reinit=True))
                bad, did_reinit = 0, True
                restarts += 1
        if (window is not None and window.enabled) or writer is not None:
            img = annotate(frame, i, g, row, tracker.name, ms, reinit=did_reinit, lost=lost, status=status_text())
            if did_reinit:
                _rect(img, g, YELLOW, max(1, img.shape[1] // 640) + 1)
            if writer is not None:
                writer.write(img)
            if window is not None:
                stop = window.show(img, wait_ms=pace_ms() if window.enabled else 1)
                if stop:
                    break
    if writer is not None:
        writer.release()
    return rows, init_ms, stop
