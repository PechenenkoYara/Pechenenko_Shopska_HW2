# %% [markdown]
# 07 — Ensemble of the team's detectors with Weighted Boxes Fusion (WBF).
# CPU only: Accelerator = None, Internet = ON (pip). Run common.py as the first cell.
# Inputs: competition data + predictions of each model in the shared format {path: [(cls0, score, x, y, w, h)]}:
#   - <run>/val_preds.pkl + <run>/test_preds.pkl   (02_faster_rcnn, 04_retinanet: notebook output or uploaded dataset)
#   - val_preds_<name>.pkl + test_preds_<name>.pkl  (03b_yolo_p2, 05_rtdetr, 06_sahi_tta)
# WBF: Solovyev et al., "Weighted boxes fusion", Image and Vision Computing 2021; code: github.com/ZFTurbo/Weighted-Boxes-Fusion.
# Unlike NMS, WBF averages the coordinates of overlapping boxes from different models instead of keeping one of them.
# Search on quick val (every 10th val frame), then the chosen setting is evaluated on the full val/.

# %%
!pip -q install ensemble-boxes pycocotools
import pickle, itertools, time, contextlib, io
from ensemble_boxes import weighted_boxes_fusion

DEBUG = True              # smoke test: few val frames, 16 test frames
USE = None                # models to ensemble, e.g. ['frcnn_short', 'retina_focal', 'yolo11s_sahi_2x2']; None = all found
IOU_THRS = [0.5, 0.6, 0.7]  # WBF matching IoU, searched for the best subset
SKIP_THR = 0.01           # drop boxes below this score before fusion (speed; they barely affect mAP)
TOP_K = 300               # keep the top-K boxes per model and frame before fusion (max is 180 GT objects per frame)
CONF_TYPE = 'avg'         # 'avg': a box found by only one of N models gets its score scaled down
QUICK_STRIDE = 50 if DEBUG else 10
OUT = f'{WORK}/wbf'
os.makedirs(OUT, exist_ok=True)

# %% Find prediction files in /kaggle/input and load them keyed by '<clip>/<frame>.jpg'
def find_models():
    found = {}
    for vp in sorted(glob.glob('/kaggle/input/**/val_preds*.pkl', recursive=True)):
        base, folder = os.path.basename(vp), os.path.dirname(vp)
        name = os.path.basename(folder) if base == 'val_preds.pkl' else base[len('val_preds_'):-len('.pkl')]
        name = name.replace('-', '_')
        tp = os.path.join(folder, base.replace('val_preds', 'test_preds'))
        if name in found:
            print('duplicate name, skipped:', vp); continue
        found[name] = (vp, tp if os.path.exists(tp) else None)
    return found

def load_preds(path):
    return {key(p): d for p, d in pickle.load(open(path, 'rb')).items()}

found = find_models()
for n, (vp, tp) in found.items():
    print(f'{n:28s} val: {vp}\n{"":28s} test: {tp}')
names = [n for n in (USE or found) if n in found]
assert names, 'no val_preds*.pkl found in /kaggle/input'
missing_test = [n for n in names if found[n][1] is None]
assert not missing_test, f'test_preds missing for {missing_test}: upload them next to val_preds'
VAL = {n: load_preds(found[n][0]) for n in names}

# %% Image sizes (WBF works with coordinates normalized to [0, 1])
ds_val = json.load(open(f'{ROOT}/val/annotations_coco.json'))
SIZE = {key(im['file_name']): (im['width'], im['height']) for im in ds_val['images']}
test_tab = pd.read_csv(f'{ROOT}/test_images.csv')
SIZE.update({key(fn): (w, h) for fn, w, h in zip(test_tab.file_name, test_tab.width, test_tab.height)})

# %% WBF for one frame
def fuse_frame(k, per_model, weights, iou_thr):
    W, H = SIZE[k]
    boxes, scores, labels = [], [], []
    for dets in per_model:
        d = np.array(sorted(dets, key=lambda t: -t[1])[:TOP_K], dtype=float).reshape(-1, 6)
        d = d[(d[:, 4] >= 1) & (d[:, 5] >= 1)]      # drop degenerate boxes (RetinaNet keeps zero-height boxes at the border)
        b = np.c_[d[:, 2] / W, d[:, 3] / H, (d[:, 2] + d[:, 4]) / W, (d[:, 3] + d[:, 5]) / H].clip(0, 1)
        boxes.append(b.tolist()); scores.append(d[:, 1].tolist()); labels.append(d[:, 0].astype(int).tolist())
    b, s, l = weighted_boxes_fusion(boxes, scores, labels, weights=weights, iou_thr=iou_thr,
                                    skip_box_thr=SKIP_THR, conf_type=CONF_TYPE)
    return [(int(c), float(sc), x1 * W, y1 * H, (x2 - x1) * W, (y2 - y1) * H)
            for (x1, y1, x2, y2), sc, c in zip(b, s, l)]

def fuse(preds_by_model, frames, weights=None, iou_thr=0.6):
    return {k: fuse_frame(k, [preds_by_model[n].get(k, []) for n in preds_by_model], weights, iou_thr) for k in frames}

# %% Quick-val frames and single-model reference on them
quick = [k for k in sorted(SIZE) if k in VAL[names[0]] and int(k.split('/')[-1][:-4]) % QUICK_STRIDE == 0]
rows = []
def evaluate(tag, members, weights, iou_thr, preds):
    with contextlib.redirect_stdout(io.StringIO()):      # pycocotools prints index/progress lines
        m = coco_eval(preds, verbose=False)
    rows.append({'setting': tag, 'models': '+'.join(members), 'weights': weights, 'iou_thr': iou_thr, **m})
    print(f"{tag:10s} {'+'.join(members):60s} w={weights} iou={iou_thr}  mAP {m['mAP']:.4f}  AP50 {m['AP50']:.4f}  AP_small {m['AP_small']:.4f}")
    return m['mAP']

for n in names:
    evaluate('single', [n], None, None, {k: VAL[n].get(k, []) for k in quick})

# %% Step 1: which models to combine (equal weights, iou 0.6)
t0 = time.time()
best = (-1, None)
for r in range(2, len(names) + 1):
    for sub in itertools.combinations(names, r):
        score = evaluate('subset', sub, None, 0.6, fuse({n: VAL[n] for n in sub}, quick, None, 0.6))
        best = max(best, (score, sub))
best_sub = best[1]
print(f'best subset: {best_sub} ({(time.time() - t0) / 60:.1f} min)')

# %% Step 2: WBF IoU threshold, then weights (2x for each member in turn) for the best subset
best_iou, best_score = 0.6, best[0]
for thr in IOU_THRS:
    if thr != 0.6:
        s = evaluate('iou', best_sub, None, thr, fuse({n: VAL[n] for n in best_sub}, quick, None, thr))
        if s > best_score: best_iou, best_score = thr, s
best_w = None
for i in range(len(best_sub)):
    w = [2 if j == i else 1 for j in range(len(best_sub))]
    s = evaluate('weights', best_sub, w, best_iou, fuse({n: VAL[n] for n in best_sub}, quick, w, best_iou))
    if s > best_score: best_w, best_score = w, s
search = pd.DataFrame(rows)
search.to_csv(f'{OUT}/search_quick_val.csv', index=False)
best_single = search[search.setting == 'single'].sort_values('mAP').iloc[-1]
print('chosen:', best_sub, 'weights', best_w, 'iou', best_iou, f'| quick val mAP {best_score:.4f}',
      f'vs best single {best_single.models} {best_single.mAP:.4f}')
if best_score <= best_single.mAP:
    print('WARNING: the ensemble does not beat the best single model on quick val')

# %% Full val of the chosen ensemble (quick val was used for the choice, so this is slightly optimistic)
full_frames = [k for k in sorted(SIZE) if k in VAL[best_sub[0]]]
val_wbf = fuse({n: VAL[n] for n in best_sub}, full_frames, best_w, best_iou)
metrics = coco_eval(val_wbf)
print('WBF full val:', metrics)
cfg = dict(models=list(best_sub), weights=best_w, iou_thr=best_iou, skip_box_thr=SKIP_THR, top_k=TOP_K, conf_type=CONF_TYPE)
json.dump({'config': cfg, 'metrics': metrics}, open(f'{OUT}/results.json', 'w'), indent=1)
pickle.dump(val_wbf, open(f'{OUT}/val_preds.pkl', 'wb'))

# %% Test: fuse the same models with the same settings -> submission.csv
TEST = {n: load_preds(found[n][1]) for n in best_sub}
test_frames = [key(p) for p in test_paths()]
test_wbf = fuse(TEST, test_frames[:16] if DEBUG else test_frames, best_w, best_iou)
pickle.dump(test_wbf, open(f'{OUT}/test_preds.pkl', 'wb'))
write_submission(test_wbf, out=f'{OUT}/submission.csv')
