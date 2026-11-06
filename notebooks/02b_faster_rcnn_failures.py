# %% [markdown]
# 02b — Failure-case analysis of the tuned two-stage detector (Faster R-CNN, frcnn_short).
# CPU only: Accelerator = None. Inputs: competition data + output of the Faster R-CNN notebook version
# that produced frcnn_short/val_preds.pkl. Run common.py as the first cell.
# Error types follow TIDE (Bolya et al., ECCV 2020, github.com/dbolya/tide) in a simplified form:
# TP, duplicate, wrong class, localization, background; plus missed ground truth (FN).
# Figures are saved to /kaggle/working/figs (download them into images/ of the repo).

# %%
import pickle
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from pycocotools.cocoeval import COCOeval

RUN = 'frcnn_short'
IOU_TP = 0.5            # IoU needed for a true positive (AP50 definition)
# val_preds.pkl either as notebook output (<RUN>/val_preds.pkl) or uploaded as a dataset named like the run
# (Kaggle turns '_' into '-' in dataset names). Notebook inputs default to the LATEST version of a notebook.
hits = (glob.glob(f'/kaggle/input/**/{RUN}/val_preds.pkl', recursive=True) +
        glob.glob(f"/kaggle/input/**/*{RUN.replace('_', '-')}*/**/val_preds.pkl", recursive=True))
assert hits, f'{RUN} val_preds.pkl not found: upload it as a dataset named {RUN.replace("_", "-")}'
preds = {key(p): d for p, d in pickle.load(open(hits[0], 'rb')).items()}
FIG = f'{WORK}/figs'
os.makedirs(FIG, exist_ok=True)
print(len(preds), 'val frames with predictions')

# %% Ground truth per frame: objects (cls0, x, y, w, h, occlusion) and ignored regions
ds = json.load(open(f'{ROOT}/val/annotations_coco.json'))
cat2cls = {c['id']: CLASSES.index(c['name']) for c in ds['categories']}
images = {im['id']: im for im in ds['images']}
gt, ignore = defaultdict(list), defaultdict(set)
for a in ds['annotations']:
    k = key(images[a['image_id']]['file_name'])
    if a.get('iscrowd', 0):
        ignore[k].add(tuple(a['bbox']))
    else:
        gt[k].append((cat2cls[a['category_id']], *a['bbox'], a.get('occlusion', 0)))
clip_of = {key(im['file_name']): key(im['file_name']).split('/')[0] for im in ds['images']}

# %% Per-class AP from COCOeval (same evaluator as coco_eval in common.py)
G = load_gt(f'{ROOT}/val/annotations_coco.json')
key2id = {key(im['file_name']): im['id'] for im in G.dataset['images']}
res = [{'image_id': key2id[k], 'category_id': G.cls2cat[c], 'score': s, 'bbox': [x, y, w, h]}
       for k, dets in preds.items() for c, s, x, y, w, h in dets]
E = COCOeval(G, G.loadRes(res), 'bbox')
E.params.imgIds = sorted(key2id[k] for k in preds)
E.params.maxDets = [1, 100, 500]
E.evaluate(); E.accumulate()
prec = E.eval['precision']                       # [IoU thr, recall, class, area, maxDets]

def ap(k, area=0, iou=None):
    p = prec[:, :, k, area, -1] if iou is None else prec[iou, :, k, area, -1]
    p = p[p > -1]
    return round(float(p.mean()), 3) if p.size else float('nan')

cls_ids = sorted(G.cls2cat)                       # COCOeval orders classes by category id
per_class = pd.DataFrame({'class': [CLASSES[c] for c in cls_ids],
                          'n_gt': [sum(o[0] == c for g in gt.values() for o in g) for c in cls_ids],
                          'AP': [ap(i) for i in range(len(cls_ids))],
                          'AP50': [ap(i, iou=0) for i in range(len(cls_ids))],
                          'AP_small': [ap(i, area=1) for i in range(len(cls_ids))]})
per_class

# %% Greedy matching in score order (as COCOeval does), once for all thresholds:
# a prediction's outcome depends only on higher-scored predictions, so any score threshold is a prefix.
def to_xyxy(b):
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    return np.c_[b[:, :2], b[:, :2] + b[:, 2:]]

def iou_matrix(a, b, ioa=False):
    """IoU (or intersection over area of a) between xywh boxes a (N) and b (M)."""
    a, b = to_xyxy(a), to_xyxy(b)
    lt = np.maximum(a[:, None, :2], b[None, :, :2]); rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    area_a = (a[:, 2:] - a[:, :2]).prod(-1)[:, None]
    if ioa:
        return inter / np.maximum(area_a, 1e-9)
    area_b = (b[:, 2:] - b[:, :2]).prod(-1)[None]
    return inter / np.maximum(area_a + area_b - inter, 1e-9)

P_rows, G_rows = [], []
for k, dets in preds.items():
    d = np.array(sorted(dets, key=lambda t: -t[1]), dtype=float).reshape(-1, 6)
    g = np.array(gt.get(k, []), dtype=float).reshape(-1, 6)
    I = iou_matrix(d[:, 2:], g[:, 1:5]) if len(g) else np.zeros((len(d), 0))
    same = d[:, :1] == g[None, :, 0] if len(g) else np.zeros((len(d), 0), bool)
    ign = (iou_matrix(d[:, 2:], list(ignore[k]), ioa=True).max(1) >= 0.5) if ignore[k] else np.zeros(len(d), bool)
    owner = np.full(len(g), -1)
    for i in range(len(d)):
        free = same[i] & (I[i] >= IOU_TP) & (owner < 0)
        if free.any():
            j = np.flatnonzero(free)[np.argmax(I[i, free])]; owner[j] = i; kind = 'TP'
        elif ign[i]:
            kind = 'ignored'                       # inside an ignored region: not scored by the evaluator
        elif (same[i] & (I[i] >= IOU_TP)).any():
            kind = 'duplicate'
        elif (I[i] >= IOU_TP).any():
            kind = 'wrong class'
        elif len(g) and I[i].max() >= 0.1:
            kind = 'localization'
        else:
            kind = 'background'
        gt_cls = int(g[np.argmax(I[i]), 0]) if kind == 'wrong class' else -1
        P_rows.append((k, int(d[i, 0]), d[i, 1], kind, gt_cls))
    for j, (c, x, y, w, h, occ) in enumerate(g):
        G_rows.append((k, int(c), x, y, w, h, int(occ), d[owner[j], 1] if owner[j] >= 0 else np.nan, len(g)))

P = pd.DataFrame(P_rows, columns=['frame', 'cls', 'score', 'kind', 'gt_cls'])
GT = pd.DataFrame(G_rows, columns=['frame', 'cls', 'x', 'y', 'w', 'h', 'occlusion', 'match_score', 'n_in_frame'])
GT['size'] = np.sqrt(GT.w * GT.h)
print(len(P), 'predictions |', len(GT), 'GT objects')

# %% Operating threshold = best F1 at IoU 0.5 (for counting errors; mAP itself uses all scores)
thrs = np.round(np.arange(0.05, 0.96, 0.05), 2)
scored = P[P.kind != 'ignored']
f1 = []
for t in thrs:
    tp = ((scored.score >= t) & (scored.kind == 'TP')).sum()
    f1.append(2 * tp / ((scored.score >= t).sum() + len(GT)))
THR = float(thrs[int(np.argmax(f1))])
GT['found'] = GT.match_score >= THR
Pt = P[(P.score >= THR) & (P.kind != 'ignored')]
print(f'threshold {THR}: precision {(Pt.kind == "TP").mean():.3f}, recall {GT.found.mean():.3f}, F1 {max(f1):.3f}')

# %% Error breakdown at the operating threshold
errors = pd.concat([Pt.kind.value_counts(), pd.Series({'missed (FN)': (~GT.found).sum()})])
fig, ax = plt.subplots(figsize=(7, 3.5))
errors.drop('TP').sort_values().plot.barh(ax=ax, color='#c0504d')
ax.set_title(f'{RUN}: errors on val at score >= {THR} (TP = {errors["TP"]})'); ax.set_xlabel('count')
plt.tight_layout(); plt.savefig(f'{FIG}/frcnn_error_types.png', dpi=150); plt.show()

# %% Recall by object size, occlusion, scene density and class
size_bins = pd.cut(GT['size'], [0, 8, 16, 32, 64, 96, 1e4], labels=['<8', '8-16', '16-32', '32-64', '64-96', '>96'])
dens_bins = pd.cut(GT.n_in_frame, [0, 25, 50, 100, 200, 1e4], labels=['<=25', '26-50', '51-100', '101-200', '>200'])
occ_names = {0: 'none', 1: 'partial', 2: 'heavy'}             # VisDrone occlusion levels
panels = [('object size, sqrt(area) px', GT.groupby(size_bins, observed=True)),
          ('occlusion', GT.groupby(GT.occlusion.map(occ_names))),
          ('objects in frame', GT.groupby(dens_bins, observed=True)),
          ('class', GT.groupby(GT.cls.map(dict(enumerate(CLASSES)))))]
fig, axes = plt.subplots(1, 4, figsize=(18, 3.8))
for ax, (title, grp) in zip(axes, panels):
    r = grp.found.agg(['mean', 'size'])
    ax.bar(r.index.astype(str), r['mean'], color='#4f81bd')
    for i, (m, n) in enumerate(zip(r['mean'], r['size'])):
        ax.text(i, m + 0.01, f'n={n}', ha='center', fontsize=7)
    ax.set_ylim(0, 1); ax.set_title(f'recall vs {title}'); ax.tick_params(axis='x', rotation=45)
plt.tight_layout(); plt.savefig(f'{FIG}/frcnn_recall_breakdown.png', dpi=150); plt.show()

# %% Which classes get confused (wrong-class errors: predicted vs ground-truth class)
wc = Pt[Pt.kind == 'wrong class']
conf = pd.crosstab(wc.gt_cls.map(dict(enumerate(CLASSES))), wc.cls.map(dict(enumerate(CLASSES))))
conf.index.name, conf.columns.name = 'ground truth', 'predicted'
conf

# %% Per-clip recall and precision (how results vary across sequences)
Pt_clip = Pt.assign(clip=Pt.frame.map(clip_of))
per_clip = pd.DataFrame({'recall': GT.groupby(GT.frame.map(clip_of)).found.mean(),
                         'precision': Pt_clip.groupby('clip').kind.apply(lambda s: (s == 'TP').mean()),
                         'objects/frame': GT.groupby(GT.frame.map(clip_of)).n_in_frame.mean(),
                         'median size px': GT.groupby(GT.frame.map(clip_of))['size'].median()}).round(3)
per_clip.sort_values('recall')

# %% Example frames: the frame where each failure type is most frequent
def frame_path(k):
    return f'{ROOT}/val/images/{k}'

def show_case(k, title, fname, focus):
    """Full frame + zoom centred on the boxes of the `focus` failure type."""
    img = Image.open(frame_path(k))
    g = GT[GT.frame == k]
    det = np.array(sorted(preds[k], key=lambda t: -t[1]), dtype=float).reshape(-1, 6)
    det = det[det[:, 1] >= THR]
    kinds = P[(P.frame == k) & (P.score >= THR)].kind.values      # same order as det: sorted by score
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), gridspec_kw={'width_ratios': [1.6, 1]})
    if focus == 'missed':
        err = [tuple(r) for r in g.loc[~g.found, ['x', 'y', 'w', 'h']].values]
    else:
        err = [(x, y, w, h) for (_, _, x, y, w, h), kd in zip(det, kinds) if kd == focus]
    cx, cy = (np.median([e[0] + e[2] / 2 for e in err]), np.median([e[1] + e[3] / 2 for e in err])) if err else (img.width / 2, img.height / 2)
    zw, zh = img.width / 4, img.height / 4
    zoom = (max(0, cx - zw / 2), max(0, cy - zh / 2))
    for ax in axes:
        ax.imshow(img); ax.axis('off')
        for _, r in g.iterrows():
            if not r.found:
                ax.add_patch(mpatches.Rectangle((r.x, r.y), r.w, r.h, fill=False, ec='red', lw=1.2))
        for (c, s, x, y, w, h), kd in zip(det, kinds):
            col = {'TP': 'lime', 'wrong class': 'magenta', 'localization': 'orange',
                   'duplicate': 'cyan', 'background': 'yellow'}.get(kd)
            if col:
                ax.add_patch(mpatches.Rectangle((x, y), w, h, fill=False, ec=col, lw=1))
    axes[0].add_patch(mpatches.Rectangle(zoom, zw, zh, fill=False, ec='white', lw=2, ls='--'))
    axes[1].set_xlim(zoom[0], zoom[0] + zw); axes[1].set_ylim(zoom[1] + zh, zoom[1])
    axes[0].set_title(title, fontsize=10)
    axes[1].legend(handles=[mpatches.Patch(color=c, label=l) for c, l in
                            [('lime', 'TP'), ('red', 'missed GT'), ('magenta', 'wrong class'),
                             ('orange', 'localization'), ('cyan', 'duplicate'), ('yellow', 'background')]],
                   loc='lower right', fontsize=8)
    plt.tight_layout(); plt.savefig(f'{FIG}/{fname}', dpi=150); plt.show()

per_frame = pd.DataFrame({'missed': (~GT.found).groupby(GT.frame).sum(),
                          'n_gt': GT.groupby('frame').size()}).join(
    Pt.groupby(['frame', 'kind']).size().unstack(fill_value=0), how='left').fillna(0)
cases = [('missed', 'most missed objects'), ('wrong class', 'most class confusions'),
         ('background', 'most background false positives'), ('localization', 'most poorly localized boxes')]
for col, desc in cases:
    if col in per_frame and per_frame[col].max() > 0:
        k = per_frame[col].idxmax()
        show_case(k, f'{k}: {desc} ({int(per_frame.loc[k, col])} of {int(per_frame.loc[k, "n_gt"])} GT objects)',
                  f'frcnn_case_{col.replace(" ", "_")}.png', col)

# %% Numbers for the report (copy into the experiment log)
print('per-class AP:\n', per_class.to_string(index=False))
print('\nerrors at threshold', THR, ':\n', errors.to_string())
print('\nrecall by size:\n', GT.groupby(size_bins, observed=True).found.mean().round(3).to_string())
