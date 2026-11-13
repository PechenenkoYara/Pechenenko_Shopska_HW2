# %% [markdown]
# 06 — Inference-only experiments on trained YOLO weights: NMS IoU, TTA, SAHI-style tiling.
# Weights (TAG): YOLO11s baseline (03) -> SAHI public 0.16733 vs 0.17329; yolo11s_p2 (03b) -> the table in RUN_BASE,
# SAHI public 0.20267 vs 0.21219. Default SUBMIT='nms_0.6' on yolo11s_p2 = plain P2 predictions exported for 07_wbf.py.
# No training, so this costs little GPU quota. Each experiment changes one thing against `nms_0.6`.
# Kaggle: Accelerator = GPU T4, Internet = ON, Input = competition data + dataset with best.pt.
# Run common.py as the first cell. Set DEBUG = False for the real run.
# Tiling idea: Akyon et al., "Slicing Aided Hyper Inference", ICIP 2022 (github.com/obss/sahi); minimal own version.
# TTA: Ultralytics `augment=True`.

# %%
!pip -q install ultralytics pycocotools
import pickle, shutil, time, zipfile
import cv2
from ultralytics import YOLO

DEBUG = True          # smoke test: few val frames, 8 test frames
IMGSZ = 1280
TILE_FRAC = 0.6       # tile side = 60% of the frame side, 2x2 grid -> 20% overlap, about 1.67x zoom
NMS_IOU = 0.6
MAX_DET = 500
VAL_STRIDE = 50 if DEBUG else 1
TAG = 'yolo11s_p2'    # weights used in this run: goes into the names of saved predictions
RUN_BASE = False      # True: full val table (done for yolo11s_p2: sahi_2x2 0.2104, nms_0.6 0.2046)
SUBMIT = 'nms_0.6'   # name from the val table -> val + test predictions + submission.csv. Never automatic: val misled us on SAHI
                      # 'nms_0.6' = plain yolo11s_p2 (val 0.2046) for 07_wbf.py. SAHI lost on public twice: P2 0.20267 vs 0.21219 without SAHI,
                      # baseline 0.16733 vs 0.17329, although it won on val (0.2104 vs 0.2046)
RUN_SIM = False       # True: resolution check (done: SAHI wins mAP at all four sides)
RUN_EXPORT = False    # True: val + test predictions of each method for WBF, FPS
SIM_STRIDE = 100 if DEBUG else 5   # val frames for the resolution check
SIM_SIDES = [680, 960, 1360, 1920] # long sides that exist in test; val has none below 1344
SAHI_MIN_SIDE = None  # set after the resolution table: SAHI only for test frames with long side >= this

def find_weights():
    """Path to best.pt. A .pt file is a zip archive and Kaggle unpacks it on upload -> pack the folder back."""
    hits = glob.glob('/kaggle/input/**/best.pt', recursive=True)
    if hits:
        return hits[0]
    pkl = glob.glob('/kaggle/input/**/best/data.pkl', recursive=True)
    assert pkl, 'best.pt not found: add the dataset with weights via "Add Input"'
    src, out = os.path.dirname(pkl[0]), f'{WORK}/best.pt'
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_STORED) as z:
        for folder, _, files in os.walk(src):
            for f in files:
                p = os.path.join(folder, f)
                z.write(p, os.path.join('best', os.path.relpath(p, src)))
    return out

model = YOLO(find_weights())
# A wrong Input version would silently mislabel predictions as TAG. P2 head -> strides [4, 8, 16, 32] (plain: [8, 16, 32]);
# yolo11s ~10M parameters, yolo11m ~20M+.
print('weights:', find_weights(), '| strides', model.model.stride.int().tolist(), '|',
      round(sum(p.numel() for p in model.model.parameters()) / 1e6, 1), 'M params')
assert ('p2' not in TAG) == (model.model.stride.min() >= 8), f'TAG={TAG} does not match the loaded weights'
half = torch.cuda.is_available()

# %% TTA (Ultralytics): flips and scales inside model.predict
def predict_tta(paths, bs=32):
    out = {}
    for i in range(0, len(paths), bs):
        chunk = paths[i:i + bs]
        res = model.predict(chunk, imgsz=IMGSZ, conf=0.001, iou=NMS_IOU, max_det=MAX_DET, augment=True,
                            verbose=False, half=half)
        for p, r in zip(chunk, res):
            out[p] = [(c, s, x1, y1, x2 - x1, y2 - y1) for c, s, x1, y1, x2, y2 in boxes(r)]
    return out


def boxes(r):
    """Ultralytics result -> [(cls0, score, x1, y1, x2, y2), ...] in the coordinates of the model input."""
    b = r.boxes
    return [(int(c), float(s), float(x1), float(y1), float(x2), float(y2))
            for (x1, y1, x2, y2), s, c in zip(b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy())]


# %% SAHI-style tiling: full frame + 2x2 overlapping tiles, merged with class-wise NMS
def tile_rects(W, H, frac=TILE_FRAC):
    tw, th = int(W * frac), int(H * frac)
    return [(x, y, x + tw, y + th) for y in (0, H - th) for x in (0, W - tw)]


def nms(dets, iou_thr=NMS_IOU, max_det=MAX_DET):
    """Class-wise greedy NMS on [(cls0, score, x, y, w, h), ...]."""
    if not dets:
        return []
    d = np.array(dets, dtype=np.float64)
    keep = []
    for c in np.unique(d[:, 0]):
        idx = np.where(d[:, 0] == c)[0]
        idx = idx[np.argsort(-d[idx, 1])]
        x1, y1, w, h = d[idx, 2], d[idx, 3], d[idx, 4], d[idx, 5]
        x2, y2, area = x1 + w, y1 + h, w * h
        alive = np.ones(len(idx), bool)
        for i in range(len(idx)):
            if not alive[i]:
                continue
            keep.append(idx[i])
            inter = (np.minimum(x2[i], x2) - np.maximum(x1[i], x1)).clip(0) * \
                    (np.minimum(y2[i], y2) - np.maximum(y1[i], y1)).clip(0)
            over = inter / (area[i] + area - inter + 1e-9) > iou_thr
            over[:i + 1] = False
            alive[over] = False
    keep = sorted(keep, key=lambda j: -d[j, 1])[:max_det]
    return [(int(d[j, 0]), *map(float, d[j, 1:])) for j in keep]


def tile_dets(r, rect, W, H, margin=2):
    """Tile detections in frame pixels. Boxes cut by an inner tile edge are dropped: full frame or a neighbour covers them."""
    x0, y0, x1, y1 = rect
    tw, th = x1 - x0, y1 - y0
    out = []
    for c, s, a, b, e, f in boxes(r):
        cut = (a <= margin and x0 > 0) or (b <= margin and y0 > 0) or \
              (e >= tw - margin and x1 < W) or (f >= th - margin and y1 < H)
        if not cut:
            out.append((c, s, a + x0, b + y0, e - a, f - b))
    return out


def predict_sahi(paths):
    kw = dict(imgsz=IMGSZ, conf=0.001, iou=NMS_IOU, max_det=MAX_DET, verbose=False, half=half)
    out = {}
    for p in paths:
        im = cv2.imread(p)                          # BGR, as Ultralytics expects for numpy input
        H, W = im.shape[:2]
        dets = [(c, s, x1, y1, x2 - x1, y2 - y1) for c, s, x1, y1, x2, y2 in boxes(model.predict(im, **kw)[0])]
        rects = tile_rects(W, H)
        for rect, r in zip(rects, model.predict([im[y0:y1, x0:x1] for x0, y0, x1, y1 in rects], **kw)):
            dets += tile_dets(r, rect, W, H)
        out[p] = nms(dets)
    return out


# %% Experiments: `nms_0.6` must reproduce the baseline mAP from 03_yolo_baseline.py
EXPERIMENTS = {
    'nms_0.6': lambda paths: predict_ultralytics(model, paths, imgsz=IMGSZ, iou=0.6),
    'nms_0.7': lambda paths: predict_ultralytics(model, paths, imgsz=IMGSZ, iou=0.7),
    'tta': predict_tta,
    'sahi_2x2': predict_sahi,
    'imgsz_1600': lambda paths: predict_ultralytics(model, paths, imgsz=1600, iou=0.6, bs=8),
    'imgsz_1920': lambda paths: predict_ultralytics(model, paths, imgsz=1920, iou=0.6, bs=4),
}
if RUN_BASE:
    val_frames = frame_paths('val', VAL_STRIDE)
    rows = []
    for name, fn in EXPERIMENTS.items():
        t = time.time()
        preds = fn(val_frames)
        ms = (time.time() - t) / len(val_frames) * 1000
        pickle.dump(preds, open(f'{WORK}/val_preds_{TAG}_{name}.pkl', 'wb'))     # kept for WBF
        rows.append({'exp': name, **coco_eval(preds, verbose=False), 'ms_per_frame': round(ms, 1)})
        table = pd.DataFrame(rows)
        table.to_csv(f'{WORK}/inference_experiments.csv', index=False)     # saved after each one
    display(table)

# %% Val + test predictions for the experiment named in SUBMIT: both files land in one version for 07_wbf.py
if SUBMIT:
    val_preds = EXPERIMENTS[SUBMIT](frame_paths('val', VAL_STRIDE))
    pickle.dump(val_preds, open(f'{WORK}/val_preds_{TAG}_{SUBMIT}.pkl', 'wb'))
    print(SUBMIT, coco_eval(val_preds, verbose=False))
    tp = test_paths()
    test_preds = EXPERIMENTS[SUBMIT](tp[:8] if DEBUG else tp)
    pickle.dump(test_preds, open(f'{WORK}/test_preds_{TAG}_{SUBMIT}.pkl', 'wb'))
    write_submission(test_preds)


# %% Resolution check: val has 3 of 7 clips with long side >= 2688, test has none and has frames down to 680x382.
# Downscale val frames to the test long sides, predict, map boxes back to the original val coordinates.
def downscaled_val(side, stride=SIM_STRIDE):
    """Copies of val frames with long side <= `side`. Returns {copy_path: factor back to original pixels}."""
    out = {}
    for p in frame_paths('val', stride):
        im = cv2.imread(p)
        s = min(1.0, side / max(im.shape[:2]))
        q = f'{WORK}/val_sim/{side}/{key(p)}'
        os.makedirs(os.path.dirname(q), exist_ok=True)
        if s < 1:
            im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        cv2.imwrite(q, im, [cv2.IMWRITE_JPEG_QUALITY, 95])
        out[q] = 1 / s
    return out


if RUN_SIM:
    rows = []
    for side in SIM_SIDES:
        factors = downscaled_val(side)
        for name in ['nms_0.6', 'sahi_2x2']:
            preds = EXPERIMENTS[name](list(factors))
            preds = {q: [(c, s, x * factors[q], y * factors[q], w * factors[q], h * factors[q]) for c, s, x, y, w, h in dets]
                     for q, dets in preds.items()}
            rows.append({'side': side, 'exp': name, **coco_eval(preds, verbose=False)})
            sim_table = pd.DataFrame(rows)
            sim_table.to_csv(f'{WORK}/resolution_check.csv', index=False)
        shutil.rmtree(f'{WORK}/val_sim/{side}')
    shutil.rmtree(f'{WORK}/val_sim')
    display(sim_table.pivot(index='side', columns='exp', values=['mAP', 'AP50', 'AP_small']))

# %% Submission with SAHI only on large frames (test_images.csv has width/height)
if SAHI_MIN_SIDE:
    test = _test_table()
    test = test.head(8) if DEBUG else test
    big = test.width.clip(lower=test.height) >= SAHI_MIN_SIDE
    print('SAHI frames:', int(big.sum()), '| plain frames:', int((~big).sum()))
    test_preds = {**EXPERIMENTS['nms_0.6'](test.path[~big].tolist()), **EXPERIMENTS['sahi_2x2'](test.path[big].tolist())}
    pickle.dump(test_preds, open(f'{WORK}/test_preds_sahi_min{SAHI_MIN_SIDE}.pkl', 'wb'))
    write_submission(test_preds)


# %% Export for WBF (files are read by Anastasiia's ensemble notebook), FPS as in 02_faster_rcnn.py
def measure_fps(paths, warmup=10):
    ims = [cv2.imread(p) for p in paths]
    kw = dict(imgsz=IMGSZ, conf=0.001, iou=NMS_IOU, max_det=MAX_DET, half=False, verbose=False)
    for im in ims[:warmup]:
        model.predict(im, **kw)
    t = time.time()
    for im in ims:
        model.predict(im, **kw)
    return round(len(ims) / (time.time() - t), 1)


if RUN_EXPORT:
    val_frames, tp = frame_paths('val', VAL_STRIDE), test_paths()
    tp = tp[:8] if DEBUG else tp
    for name in ['nms_0.6', 'tta', 'sahi_2x2']:
        val_preds = EXPERIMENTS[name](val_frames)
        pickle.dump(val_preds, open(f'{WORK}/val_preds_{TAG}_{name}.pkl', 'wb'))
        pickle.dump(EXPERIMENTS[name](tp), open(f'{WORK}/test_preds_{TAG}_{name}.pkl', 'wb'))
        if name == 'nms_0.6':
            print(name, coco_eval(val_preds, verbose=False))
    print('FPS (batch 1, fp32):', measure_fps(frame_paths('val', 20)[:200]))
