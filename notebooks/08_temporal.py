# %% [markdown]
# 08 — Temporal bonus: use neighbouring frames to improve detections on the current one (CPU only, no training).
# A real object is usually detected again at almost the same place in frames t-1, t+1, ...; a false positive often is not.
# Three steps, each switched on only if it helps on val:
#   1. Rescoring: new score = (1 - ALPHA) * score + ALPHA * mean over neighbour frames of the best same-class match score.
#      Idea: Seq-NMS rescoring, Han et al., "Seq-NMS for Video Object Detection", arXiv 1602.08465 (simplified, no linking).
#   2. Camera-motion compensation before matching: the drone moves, so a 15 px car shifts by several px between frames and
#      loses IoU with itself. The global shift between two frames = median displacement of confident same-class detections
#      (estimated from the predictions themselves, no images or optical flow).
#   3. Filling misses: a confident object matched in t-1 and t+1 but absent in t is inserted in t (averaged box, lower score).
#      Idea: propagating detections to neighbouring frames, Kang et al., "T-CNN", arXiv 1604.02532 (simplified).
# Kaggle: Accelerator = None. Input = competition data + notebook/dataset with val_preds_<name>.pkl + test_preds_<name>.pkl.
# Run common.py as the first cell.

# %%
import pickle, time, contextlib, io
from itertools import product

DEBUG = True          # smoke test: 2 val clips, 2 test clips
SRC = 'detr2_frcnn_p2'  # final WBF (07_wbf.py): full val 0.3095 -> 0.3128 with temporal (motion, k=3, iou 0.5, alpha 0.2)
                        # val_preds_<SRC>.pkl, or <SRC>/val_preds.pkl (Anastasiia's format, e.g. a WBF tag)
KS = [2, 3]           # neighbour frames on each side (yolo11m_p2 grid: k=2 > k=1; WBF grid: k=3 >= k=2)
IOU_THRS = [0.5]      # same-class IoU to count as "the same object" (yolo11m_p2 grid: 0.5 > 0.3)
ALPHAS = [0.1, 0.2, 0.3]  # yolo11m_p2 grid: 0.2 best, 0.4+ hurts; WBF grid: 0.2 best
MOTIONS = [False, True]   # camera-motion compensation off / on (ablation). WBF grid: 0.3138 -> 0.3143, within noise
MOTION_CONF = 0.5     # only confident boxes estimate the camera shift
MOTION_RADIUS = 150   # px: largest displacement of an object between two compared frames
MIN_PAIRS = 5         # fewer matched pairs -> assume no camera shift
FILL_CONF = 0.5       # both neighbours must be at least this confident to insert a missed box
FILL_DECAYS = [0.3, 0.6]  # score of an inserted box = decay * mean(neighbour scores); searched after rescoring.
# WBF grid: filling changed nothing. With conf 0.001 and up to 500 boxes per frame there is almost always a weak box
# at the missed object's place already, so 'absent in t' never holds: misses are low scores, which rescoring fixes.
TOP_K = 500           # boxes kept per frame = leaderboard limit (300 cost 0.0013 mAP on yolo11m_p2)
EVAL_STRIDE = 5       # grid search scores every 5th val frame (neighbours still use all frames); final = full val


def find_preds(name):
    """Paths of val/test predictions for `name`, both file layouts used in 07_wbf.py."""
    for pattern in [f'/kaggle/input/**/val_preds_{name}.pkl', f'/kaggle/input/**/{name}/val_preds.pkl']:
        hits = glob.glob(pattern, recursive=True)
        if hits:
            test = hits[0].replace('val_preds', 'test_preds')
            assert os.path.exists(test), f'{test} missing: val and test must come from the same run'
            return hits[0], test
    found = sorted(glob.glob('/kaggle/input/**/val_preds*.pkl', recursive=True))
    raise FileNotFoundError(f'no val predictions for {name} in /kaggle/input; found: {found or "none (add the notebook via Add Input)"}')


def load_preds(path, top_k=TOP_K):
    preds = pickle.load(open(path, 'rb'))
    return {p: sorted(d, key=lambda t: -t[1])[:top_k] for p, d in preds.items()}


val_path, test_path = find_preds(SRC)
VAL, TEST = load_preds(val_path), load_preds(test_path)
if DEBUG:
    keep = lambda preds: {p: d for p, d in preds.items() if key(p).split('/')[0] in sorted({key(q).split('/')[0] for q in preds})[:2]}
    VAL, TEST = keep(VAL), keep(TEST)
print(SRC, '| val frames', len(VAL), '| test frames', len(TEST))


def evaluate(preds, stride=EVAL_STRIDE):
    """coco_eval on every `stride`-th frame (stride 1 = full val), without pycocotools printouts."""
    sub = {p: d for p, d in preds.items() if int(os.path.splitext(os.path.basename(p))[0]) % stride == 0}
    with contextlib.redirect_stdout(io.StringIO()):
        return coco_eval(sub, verbose=False)

# %% Helpers: frames of each clip in order, IoU, camera shift between two frames
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


def camera_shift(a, b):
    """(dx, dy) to subtract from frame b's boxes to align them with frame a.
    Each confident box in a is paired with the nearest same-class confident box in b within MOTION_RADIUS;
    the median of the displacements is robust to wrong pairs and to the few objects that really move.
    A second pass keeps only pairs close to the first estimate."""
    a, b = a[a[:, 1] >= MOTION_CONF], b[b[:, 1] >= MOTION_CONF]
    if len(a) < MIN_PAIRS or len(b) < MIN_PAIRS:
        return np.zeros(2)
    ca, cb = a[:, 2:4] + a[:, 4:6] / 2, b[:, 2:4] + b[:, 4:6] / 2
    disp = cb[None] - ca[:, None]                                     # (Na, Nb, 2)
    same = a[:, None, 0] == b[None, :, 0]
    est = np.zeros(2)
    for radius in (MOTION_RADIUS, MOTION_RADIUS / 4):
        dist = np.linalg.norm(disp - est, axis=2)
        dist[~same] = np.inf
        j = dist.argmin(1)
        ok = dist[np.arange(len(a)), j] < radius
        if ok.sum() < MIN_PAIRS:
            return est
        est = np.median(disp[np.arange(len(a))[ok], j[ok]], axis=0)
    return est


def to_arrays(preds):
    return {p: np.array(d, dtype=np.float64).reshape(-1, 6) for p, d in preds.items()}


SHIFTS = {}
def shift(arr, p_a, p_b, motion):
    """Cached camera shift of frame p_b relative to p_a (zero when motion compensation is off)."""
    if not motion:
        return np.zeros(2)
    if (p_a, p_b) not in SHIFTS:
        SHIFTS[(p_a, p_b)] = camera_shift(arr[p_a], arr[p_b])
    return SHIFTS[(p_a, p_b)]


def aligned(b, d):
    """Boxes b moved by -d (frame b coordinates -> frame a coordinates)."""
    b = b.copy()
    b[:, 2:4] -= d
    return b

# %% Step 1-2: neighbour support for every box (best same-class match in frames t-k..t+k), with or without motion
def support(preds, k, iou_thr, motion):
    """{path: (scores, mean neighbour support)}; neighbours outside the clip are not counted."""
    arr = to_arrays(preds)
    out = {}
    for frames in by_clip(preds).values():
        for i, p in enumerate(frames):
            a = arr[p]
            sup, n = np.zeros(len(a)), 0
            for j in range(max(0, i - k), min(len(frames), i + k + 1)):
                if j == i:
                    continue
                b = aligned(arr[frames[j]], shift(arr, p, frames[j], motion))
                n += 1
                if len(a) and len(b):
                    m = iou_matrix(a[:, 2:], b[:, 2:]) * (a[:, None, 0] == b[None, :, 0])
                    sup += np.where(m >= iou_thr, b[None, :, 1], 0).max(1)
            out[p] = (a[:, 1], sup / max(n, 1))
    return out


def rescore(preds, sup, alpha):
    return {p: [(c, (1 - alpha) * s0 + alpha * s1, x, y, w, h)
                for (c, _, x, y, w, h), s0, s1 in zip(d, *sup[p])] for p, d in preds.items()}

# %% Step 3: insert objects that are confident in t-1 and t+1 but missing in t (scores taken before rescoring)
def fill(base, rescored, iou_thr, motion, decay):
    """Adds the averaged, camera-aligned box of matched t-1 / t+1 pairs to frame t when t has no same-class box there."""
    if decay <= 0:
        return rescored
    arr = to_arrays(base)
    out = dict(rescored)
    for frames in by_clip(base).values():
        for i in range(1, len(frames) - 1):
            p, prev, nxt = frames[i], arr[frames[i - 1]], arr[frames[i + 1]]
            P = aligned(prev, shift(arr, p, frames[i - 1], motion))
            N = aligned(nxt, shift(arr, p, frames[i + 1], motion))
            P, N = P[P[:, 1] >= FILL_CONF], N[N[:, 1] >= FILL_CONF]
            if not len(P) or not len(N):
                continue
            m = iou_matrix(P[:, 2:], N[:, 2:]) * (P[:, None, 0] == N[None, :, 0])
            j = m.argmax(1)
            pairs = np.flatnonzero(m[np.arange(len(P)), j] >= iou_thr)
            if not len(pairs):
                continue
            new = (P[pairs] + N[j[pairs]]) / 2                         # averaged box; class is the same
            new[:, 0] = P[pairs, 0]
            new[:, 1] = decay * new[:, 1]
            cur = arr[p]
            if len(cur):
                mc = iou_matrix(new[:, 2:], cur[:, 2:]) * (new[:, None, 0] == cur[None, :, 0])
                new = new[mc.max(1) < iou_thr]                         # already detected in t -> nothing to fill
            if len(new):
                dets = out[p] + [(int(c), float(s), x, y, w, h) for c, s, x, y, w, h in new]
                out[p] = sorted(dets, key=lambda t: -t[1])[:TOP_K]
    return out

# %% Grid search on val (every EVAL_STRIDE-th frame): rescoring with and without motion compensation
rows = [{'step': 'none', 'motion': None, 'k': 0, 'iou': None, 'alpha': 0.0, 'fill': 0.0, **evaluate(VAL)}]
SUP = {}
for motion, k, thr in product(MOTIONS, KS, IOU_THRS):
    t = time.time()
    SUP[(motion, k, thr)] = support(VAL, k, thr, motion)
    for alpha in ALPHAS:
        rows.append({'step': 'rescore', 'motion': motion, 'k': k, 'iou': thr, 'alpha': alpha, 'fill': 0.0,
                     **evaluate(rescore(VAL, SUP[(motion, k, thr)], alpha)), 'support_s': round(time.time() - t, 1)})
    table = pd.DataFrame(rows)
    table.to_csv(f'{WORK}/temporal_{SRC}.csv', index=False)      # saved after each step

# %% Fill missed objects on top of the best rescoring setting
best = table[table.step == 'rescore'].sort_values('mAP').iloc[-1]
motion, k, thr, alpha = bool(best.motion), int(best.k), float(best.iou), float(best.alpha)
val_r = rescore(VAL, SUP[(motion, k, thr)], alpha)
for decay in FILL_DECAYS:
    rows.append({'step': 'rescore+fill', 'motion': motion, 'k': k, 'iou': thr, 'alpha': alpha, 'fill': decay,
                 **evaluate(fill(VAL, val_r, thr, motion, decay))})
table = pd.DataFrame(rows)
table.to_csv(f'{WORK}/temporal_{SRC}.csv', index=False)
display(table.sort_values('mAP', ascending=False))

# %% Best setting -> full val metrics, val + test predictions (for WBF / submission) in one output
best = table.iloc[1:].sort_values('mAP').iloc[-1]
decay = float(best.fill)
print(f'best: motion={motion}, k={k}, iou={thr}, alpha={alpha}, fill={decay} '
      f'| mAP {best.mAP} vs {table.mAP.iloc[0]} without temporal (every {EVAL_STRIDE}th val frame)')
if best.mAP <= table.mAP.iloc[0]:
    print('WARNING: no temporal setting beats the untouched predictions on val')
val_t = fill(VAL, val_r, thr, motion, decay)
test_t = fill(TEST, rescore(TEST, support(TEST, k, thr, motion), alpha), thr, motion, decay)
pickle.dump(val_t, open(f'{WORK}/val_preds_{SRC}_temporal.pkl', 'wb'))
pickle.dump(test_t, open(f'{WORK}/test_preds_{SRC}_temporal.pkl', 'wb'))
if not DEBUG:
    write_submission(test_t)
full = pd.DataFrame([{'preds': SRC, **evaluate(VAL, 1)}, {'preds': f'{SRC}_temporal', **evaluate(val_t, 1)}])
full.to_csv(f'{WORK}/temporal_{SRC}_full_val.csv', index=False)
full
