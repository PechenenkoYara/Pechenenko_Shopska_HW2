# %% [markdown]
# 07 — Ensemble of the team's detectors with Weighted Boxes Fusion (WBF).
# CPU only: Accelerator = None, Internet = ON (pip). Run common.py as the first cell.
# Inputs: competition data + predictions of each model in the shared format {path: [(cls0, score, x, y, w, h)]}:
#   - <run>/val_preds.pkl + <run>/test_preds.pkl   (02_faster_rcnn, 04_retinanet: notebook output or uploaded dataset)
#   - val_preds_<name>.pkl + test_preds_<name>.pkl  (03b_yolo_p2, 05_rtdetr, 06_sahi_tta)
# WBF: Solovyev et al., "Weighted boxes fusion", Image and Vision Computing 2021; code: github.com/ZFTurbo/Weighted-Boxes-Fusion.
# Unlike NMS, WBF averages the coordinates of overlapping boxes from different models instead of keeping one of them.
# Two modes: SEARCH picks models/IoU/weights on quick val (every 10th val frame); MANUAL lists settings to export as they are.
# MANUAL exists because val ranks model families unreliably (frcnn_short >> yolo11s_p2 on val, equal on public),
# so weights chosen only on val can underweight a model -> each MANUAL setting gets its own submission to check on public.

# %%
!pip -q install ensemble-boxes pycocotools
import pickle, itertools, time, contextlib, io
from ensemble_boxes import weighted_boxes_fusion

DEBUG = True              # smoke test: few val frames, 16 test frames
USE = ['rtdetr_l_fixwarmup', 'frcnn_short', 'yolo11s_p2', 'retina_focal']   # SEARCH tries all subsets of these
SEARCH = True             # search subsets / IoU / weights on quick val and export the best one as 'search_best'
MANUAL = [                # extra settings: full val metrics + own submission for each (val ranks model families unreliably)
    dict(tag='frcnn_retina_p2_eq', models=['frcnn_short', 'retina_focal', 'yolo11s_p2'], weights=[1, 1, 1], iou=0.7),
    dict(tag='frcnn_retina_p2x2',  models=['frcnn_short', 'retina_focal', 'yolo11s_p2'], weights=[1, 1, 2], iou=0.7),
    dict(tag='detr_frcnn_p2_eq',   models=['rtdetr_l_fixwarmup', 'frcnn_short', 'yolo11s_p2'], weights=[1, 1, 1], iou=0.7),
    dict(tag='detr2_frcnn_p2',     models=['rtdetr_l_fixwarmup', 'frcnn_short', 'yolo11s_p2'], weights=[2, 1, 1], iou=0.7),
    dict(tag='detr_frcnn_p2_p2m',  models=['rtdetr_l_fixwarmup', 'frcnn_short', 'yolo11s_p2', 'yolo11m_p2'], weights=[1, 1, 1, 1], iou=0.7),
]
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
        if not os.path.exists(tp):
            print('no matching test_preds, skipped:', vp); continue
        found[name] = (vp, tp)
    return found

def load_preds(path):
    return {key(p): d for p, d in pickle.load(open(path, 'rb')).items()}

found = find_models()
for n, (vp, tp) in found.items():
    print(f'{n:28s} val: {vp}\n{"":28s} test: {tp}')
names = list(dict.fromkeys(USE or found))
missing = [n for n in names if n not in found] + [n for m in MANUAL for n in m['models'] if n not in found]
assert not missing, f'not found (val+test pair) for {sorted(set(missing))}: check Input versions and file names'
assert names, 'no val_preds*.pkl + test_preds*.pkl pairs found in /kaggle/input'
names = list(dict.fromkeys(names + [n for m in MANUAL for n in m['models']]))
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
SETTINGS = list(MANUAL)
if SEARCH:
    search_names = [n for n in (USE or names)]
    t0 = time.time()
    best = (-1, None)
    for r in range(2, len(search_names) + 1):
        for sub in itertools.combinations(search_names, r):
            score = evaluate('subset', sub, None, 0.6, fuse({n: VAL[n] for n in sub}, quick, None, 0.6))
            best = max(best, (score, sub))
    best_sub = best[1]
    print(f'best subset: {best_sub} ({(time.time() - t0) / 60:.1f} min)')

# %% Step 2: WBF IoU threshold, then weights (2x for each member in turn) for the best subset
if SEARCH:
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
    best_single = pd.DataFrame(rows).query("setting == 'single'").sort_values('mAP').iloc[-1]
    print('chosen:', best_sub, 'weights', best_w, 'iou', best_iou, f'| quick val mAP {best_score:.4f}',
          f'vs best single {best_single.models} {best_single.mAP:.4f}')
    if best_score <= best_single.mAP:
        print('WARNING: the ensemble does not beat the best single model on quick val')
    SETTINGS.insert(0, dict(tag='search_best', models=list(best_sub), weights=best_w, iou=best_iou))
for m in MANUAL:
    evaluate('manual', m['models'], m['weights'], m['iou'], fuse({n: VAL[n] for n in m['models']}, quick, m['weights'], m['iou']))
pd.DataFrame(rows).to_csv(f'{OUT}/search_quick_val.csv', index=False)

# %% Full val + test for every setting -> wbf/<tag>/{results.json, val_preds.pkl, test_preds.pkl, submission.csv}
# Quick val was used for the search, so full-val numbers of 'search_best' are slightly optimistic.
full_frames = [k for k in sorted(SIZE) if k in VAL[names[0]]]
test_frames = [key(p) for p in test_paths()]
TEST = {}
summary = []
for st in SETTINGS:
    d = f"{OUT}/{st['tag']}"
    os.makedirs(d, exist_ok=True)
    val_wbf = fuse({n: VAL[n] for n in st['models']}, full_frames, st['weights'], st['iou'])
    pickle.dump(val_wbf, open(f'{d}/val_preds.pkl', 'wb'))
    for n in st['models']:
        TEST.setdefault(n, load_preds(found[n][1]))
    test_wbf = fuse({n: TEST[n] for n in st['models']}, test_frames[:16] if DEBUG else test_frames, st['weights'], st['iou'])
    pickle.dump(test_wbf, open(f'{d}/test_preds.pkl', 'wb'))
    write_submission(test_wbf, out=f'{d}/submission.csv')
    with contextlib.redirect_stdout(io.StringIO()):
        metrics = coco_eval(val_wbf, verbose=False)
    cfg = dict(models=list(st['models']), weights=st['weights'], iou_thr=st['iou'], skip_box_thr=SKIP_THR,
               top_k=TOP_K, conf_type=CONF_TYPE)
    json.dump({'config': cfg, 'metrics': metrics}, open(f'{d}/results.json', 'w'), indent=1)
    summary.append({'tag': st['tag'], 'models': '+'.join(st['models']), 'weights': st['weights'], 'iou': st['iou'], **metrics})
    print(st['tag'], metrics)
summary = pd.DataFrame(summary)
summary.to_csv(f'{OUT}/full_val_summary.csv', index=False)
summary
