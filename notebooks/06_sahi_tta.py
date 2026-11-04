# %% [markdown]
# 06 — Inference-only experiments on the trained YOLO11s baseline: NMS IoU, TTA, SAHI-style tiling.
# No training, so this costs little GPU quota. Each experiment changes one thing against `nms_0.6`.
# Kaggle: Accelerator = GPU T4, Internet = ON, Input = competition data + dataset with best.pt.
# Run common.py as the first cell. Set DEBUG = False for the real run.
# Tiling idea: Akyon et al., "Slicing Aided Hyper Inference", ICIP 2022 (github.com/obss/sahi); minimal own version.
# TTA: Ultralytics `augment=True`.

# %%
!pip -q install ultralytics pycocotools
import pickle, time
import cv2
from ultralytics import YOLO

DEBUG = True          # smoke test: few val frames, 8 test frames
IMGSZ = 1280
TILE_FRAC = 0.6       # tile side = 60% of the frame side, 2x2 grid -> 20% overlap, about 1.67x zoom
NMS_IOU = 0.6
MAX_DET = 500
VAL_STRIDE = 50 if DEBUG else 1

ckpt = glob.glob('/kaggle/input/**/best.pt', recursive=True)
assert ckpt, 'best.pt not found: add the dataset with weights via "Add Input"'
model = YOLO(ckpt[0])
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
}
val_frames = frame_paths('val', VAL_STRIDE)
rows = []
for name, fn in EXPERIMENTS.items():
    t = time.time()
    preds = fn(val_frames)
    ms = (time.time() - t) / len(val_frames) * 1000
    pickle.dump(preds, open(f'{WORK}/val_preds_{name}.pkl', 'wb'))     # kept for WBF
    rows.append({'exp': name, **coco_eval(preds, verbose=False), 'ms_per_frame': round(ms, 1)})
    table = pd.DataFrame(rows)
    table.to_csv(f'{WORK}/inference_experiments.csv', index=False)     # saved after each one
display(table)

# %% Test predictions for the experiment with the best val mAP
SUBMIT = table.sort_values('mAP').exp.iloc[-1]
print('submitting:', SUBMIT)
tp = test_paths()
test_preds = EXPERIMENTS[SUBMIT](tp[:8] if DEBUG else tp)
pickle.dump(test_preds, open(f'{WORK}/test_preds_{SUBMIT}.pkl', 'wb'))
write_submission(test_preds)
