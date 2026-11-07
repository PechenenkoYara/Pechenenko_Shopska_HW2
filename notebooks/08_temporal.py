# %% [markdown]
# 08 — Temporal bonus: rescoring detections with their neighbours in adjacent frames (CPU only, no training).
# A real object is usually detected again at almost the same place in frames t-1, t+1, ...; a false positive often is not.
# New score = (1 - ALPHA) * score + ALPHA * mean over neighbour frames of the best same-class match score (0 if no match).
# Idea: seq-NMS rescoring, Han et al., "Seq-NMS for Video Object Detection", arXiv 1602.08465 (simplified, no linking).
# Kaggle: Accelerator = None. Input = competition data + notebook/dataset with val_preds_<name>.pkl + test_preds_<name>.pkl.
# Run common.py as the first cell.

# %%
import pickle, time
from itertools import product

DEBUG = True          # smoke test: 2 val clips, 2 test clips
SRC = 'yolo11s_p2'    # model to rescore: val_preds_<SRC>.pkl, or <SRC>/val_preds.pkl (Anastasiia's format, e.g. 'wbf')
KS = [1, 2]           # neighbour frames on each side
IOU_THRS = [0.3, 0.5] # same-class IoU to count as "the same object"; tiny boxes shift a lot, so 0.3 is included
ALPHAS = [0.2, 0.4, 0.6]
TOP_K = 300           # boxes kept per frame (max 180 GT objects per frame), lowers the cost of IoU matrices


def find_preds(name):
    """Paths of val/test predictions for `name`, both file layouts used in 07_wbf.py."""
    for pattern in [f'/kaggle/input/**/val_preds_{name}.pkl', f'/kaggle/input/**/{name}/val_preds.pkl']:
        hits = glob.glob(pattern, recursive=True)
        if hits:
            test = hits[0].replace('val_preds', 'test_preds')
            assert os.path.exists(test), f'{test} missing: val and test must come from the same run'
            return hits[0], test
    raise FileNotFoundError(f'no val predictions for {name} in /kaggle/input')


def load_preds(path, top_k=TOP_K):
    preds = pickle.load(open(path, 'rb'))
    return {p: sorted(d, key=lambda t: -t[1])[:top_k] for p, d in preds.items()}


val_path, test_path = find_preds(SRC)
VAL, TEST = load_preds(val_path), load_preds(test_path)
if DEBUG:
    keep = lambda preds: {p: d for p, d in preds.items() if key(p).split('/')[0] in sorted({key(q).split('/')[0] for q in preds})[:2]}
    VAL, TEST = keep(VAL), keep(TEST)
print(SRC, '| val frames', len(VAL), '| test frames', len(TEST))

# %% Neighbour support: for every box, the best same-class match score in frames t-k..t+k
def by_clip(preds):
    """{clip: [paths sorted by frame number]}"""
    clips = {}
    for p in preds:
        clip, frame = key(p).split('/')
        clips.setdefault(clip, []).append((int(os.path.splitext(frame)[0]), p))
    return {c: [p for _, p in sorted(v)] for c, v in clips.items()}


def iou_matrix(a, b):
    """IoU between xywh boxes a (N, 4) and b (M, 4)."""
    ax2, ay2, bx2, by2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = (np.minimum(ax2[:, None], bx2[None]) - np.maximum(a[:, None, 0], b[None, :, 0])).clip(0)
    ih = (np.minimum(ay2[:, None], by2[None]) - np.maximum(a[:, None, 1], b[None, :, 1])).clip(0)
    inter = iw * ih
    return inter / (a[:, 2:].prod(1)[:, None] + b[:, 2:].prod(1)[None] - inter + 1e-9)


def support(preds, k, iou_thr):
    """{path: (scores, mean neighbour support)}; neighbours outside the clip are not counted."""
    arr = {p: np.array(d, dtype=np.float64).reshape(-1, 6) for p, d in preds.items()}
    out = {}
    for frames in by_clip(preds).values():
        for i, p in enumerate(frames):
            a = arr[p]
            sup, n = np.zeros(len(a)), 0
            for j in range(max(0, i - k), min(len(frames), i + k + 1)):
                if j == i:
                    continue
                b = arr[frames[j]]
                n += 1
                if len(a) and len(b):
                    m = iou_matrix(a[:, 2:], b[:, 2:]) * (a[:, None, 0] == b[None, :, 0])
                    sup += np.where(m >= iou_thr, b[None, :, 1], 0).max(1)
            out[p] = (a[:, 1], sup / max(n, 1))
    return out


def rescore(preds, sup, alpha):
    return {p: [(c, (1 - alpha) * s0 + alpha * s1, x, y, w, h)
                for (c, _, x, y, w, h), s0, s1 in zip(d, *sup[p])] for p, d in preds.items()}


# %% Grid search on val: one row per (k, IoU, alpha), plus the untouched predictions as reference
rows = [{'k': 0, 'iou': None, 'alpha': 0.0, **coco_eval(VAL, verbose=False)}]
for k, thr in product(KS, IOU_THRS):
    t = time.time()
    sup = support(VAL, k, thr)
    for alpha in ALPHAS:
        rows.append({'k': k, 'iou': thr, 'alpha': alpha, **coco_eval(rescore(VAL, sup, alpha), verbose=False),
                     'support_s': round(time.time() - t, 1)})
    table = pd.DataFrame(rows)
    table.to_csv(f'{WORK}/temporal_{SRC}.csv', index=False)      # saved after each step
display(table.sort_values('mAP', ascending=False))

# %% Best setting on val -> val + test predictions (for WBF / submission) in one output
best = table.iloc[1:].sort_values('mAP').iloc[-1]
k, thr, alpha = int(best.k), float(best.iou), float(best.alpha)
print(f'best: k={k}, iou={thr}, alpha={alpha} | mAP {best.mAP} vs {table.mAP.iloc[0]} without rescoring')
val_t = rescore(VAL, support(VAL, k, thr), alpha)
test_t = rescore(TEST, support(TEST, k, thr), alpha)
pickle.dump(val_t, open(f'{WORK}/val_preds_{SRC}_temporal.pkl', 'wb'))
pickle.dump(test_t, open(f'{WORK}/test_preds_{SRC}_temporal.pkl', 'wb'))
if not DEBUG:
    write_submission(test_t)
