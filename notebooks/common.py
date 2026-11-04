# %% [markdown]
# Shared utilities (paste as the FIRST cell of every Kaggle notebook, or add as a Kaggle "utility script").
# One protocol for all models: same val set (given val/ = 7 clips, sequence-level split),
# same COCO evaluator (ignored regions = iscrowd=1, maxDets=500 like the leaderboard),
# same submission writer.

# %%
import os, glob, json, random
import numpy as np
import pandas as pd
import torch

SEED = 42
def seed_everything(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
seed_everything()

# Competition data root = folder with train/ and val/ (works for any /kaggle/input layout)
_hits = [os.path.dirname(os.path.dirname(p)) for p in glob.glob('/kaggle/input/**/train/images', recursive=True)
         if os.path.isdir(p.replace('/train/images', '/val/images'))]
assert _hits, 'Competition data not found: add it via "Add Input"; see `!find /kaggle/input -maxdepth 3`'
ROOT = _hits[0]
SAMPLE_SUB = (glob.glob(f'{ROOT}/sample_submission.csv') + glob.glob(f'{ROOT}/*ubmission*.csv') + [None])[0]
WORK = '/kaggle/working'
CLASSES = ['person', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'bus', 'motor']
print('ROOT =', ROOT, '| files:', sorted(os.listdir(ROOT)))


def key(path):
    """'.../val/images/<clip>/<frame>.jpg' -> '<clip>/<frame>.jpg' (matches COCO file_name tails)."""
    return '/'.join(path.replace('\\', '/').split('/')[-2:])


def frame_paths(split, stride=1):
    """All frames of a split, keeping every `stride`-th frame (adjacent frames are near-duplicates)."""
    paths = sorted(glob.glob(f'{ROOT}/{split}/images/*/*.jpg'))
    return [p for p in paths if int(os.path.splitext(os.path.basename(p))[0]) % stride == 0]


# %% Prediction format used everywhere: {path: [(cls0, score, x, y, w, h), ...]}, pixels, top-left xy
def predict_ultralytics(model, paths, imgsz=1280, conf=0.001, iou=0.6, max_det=500, bs=32):
    out = {}
    for i in range(0, len(paths), bs):
        chunk = paths[i:i + bs]
        res = model.predict(chunk, imgsz=imgsz, conf=conf, iou=iou, max_det=max_det, verbose=False,
                            half=torch.cuda.is_available())
        for p, r in zip(chunk, res):
            b = r.boxes
            xyxy = b.xyxy.cpu().numpy(); sc = b.conf.cpu().numpy(); cl = b.cls.cpu().numpy().astype(int)
            out[p] = [(int(c), float(s), float(x1), float(y1), float(x2 - x1), float(y2 - y1))
                      for (x1, y1, x2, y2), s, c in zip(xyxy, sc, cl)]
    return out


# %% COCO ground truth with class-agnostic ignored regions
def load_gt(ann_path):
    """COCO object where every iscrowd=1 region is copied to all 8 categories.
    COCOeval matches crowd regions only within the same category, so without this a detection of
    another class inside an ignored region would be counted as a false positive."""
    from pycocotools.coco import COCO
    ds = json.load(open(ann_path))
    names = {c['id']: c['name'] for c in ds['categories']}
    if sorted(names.values()) == sorted(CLASSES):
        cls2cat = {i: next(cid for cid, n in names.items() if n == c) for i, c in enumerate(CLASSES)}
    else:                                       # names differ -> fall back to documented 1-based ids
        cls2cat = {i: i + 1 for i in range(len(CLASSES))}
        assert set(cls2cat.values()) <= set(names), f'unexpected categories: {names}'
    ds['categories'] = [{'id': cls2cat[i], 'name': c} for i, c in enumerate(CLASSES)]
    anns, nid = [], max(a['id'] for a in ds['annotations']) + 1
    for a in ds['annotations']:
        a.setdefault('area', a['bbox'][2] * a['bbox'][3])
        if a.get('iscrowd', 0):
            for cid in cls2cat.values():
                anns.append({**a, 'id': nid, 'category_id': cid}); nid += 1
        elif a['category_id'] in cls2cat.values():
            anns.append(a)
    ds['annotations'] = anns
    gt = COCO(); gt.dataset = ds; gt.createIndex()
    gt.cls2cat = cls2cat
    return gt


# %% COCO evaluation on val (ignored regions are iscrowd=1 in annotations_coco.json)
def coco_eval(preds, ann_path=None, verbose=True):
    from pycocotools.cocoeval import COCOeval
    gt = load_gt(ann_path or f'{ROOT}/val/annotations_coco.json')
    key2id = {key(im['file_name']): im['id'] for im in gt.dataset['images']}
    assert len(key2id) == len(gt.dataset['images']), 'COCO file_name tails are not unique'
    missing = [p for p in preds if key(p) not in key2id]
    assert not missing, f'{len(missing)} predicted images not in GT, e.g. {missing[0]}'
    res = [{'image_id': key2id[key(p)], 'category_id': gt.cls2cat[c], 'score': s, 'bbox': [x, y, w, h]}
           for p, dets in preds.items() for c, s, x, y, w, h in dets]
    assert res, 'no detections'
    E = COCOeval(gt, gt.loadRes(res), 'bbox')
    E.params.imgIds = sorted({key2id[key(p)] for p in preds})   # frames with 0 dets still count
    E.params.maxDets = [1, 100, 500]          # leaderboard scores up to 500 dets / image
    E.evaluate(); E.accumulate()
    if verbose: E.summarize()
    else:
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()): E.summarize()
    names = ['mAP', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large']
    return dict(zip(names, [round(float(v), 4) for v in E.stats[:6]]))


# %% Test set: map test frames to image_id by '<clip>/<frame>.jpg' (robust to how file_name is prefixed)
def _test_table():
    test = pd.read_csv(f'{ROOT}/test_images.csv')
    on_disk = {key(p): p for p in glob.glob(f'{ROOT}/test/images/*/*.jpg')}
    test['path'] = test.file_name.map(lambda fn: on_disk.get(key(fn)))
    assert test.path.notna().all(), f'{test.path.isna().sum()} test frames not found on disk'
    return test


def test_paths():
    return _test_table().path.tolist()


# %% Submission writer (row order = sample_submission.csv if provided, else test_images.csv)
def write_submission(preds, out=f'{WORK}/submission.csv'):
    test = _test_table()
    key2id = dict(zip(test.path.map(key), test.image_id))
    strings = {}
    for p, dets in preds.items():
        dets = sorted(dets, key=lambda d: -d[1])[:500]
        strings[key2id[key(p)]] = ' '.join(f'{c} {s:.4f} {x:.1f} {y:.1f} {w:.1f} {h:.1f}' for c, s, x, y, w, h in dets)
    sub = pd.read_csv(SAMPLE_SUB)[['image_id']] if SAMPLE_SUB else test[['image_id']].copy()
    assert set(sub.image_id) == set(test.image_id), 'sample_submission and test_images ids differ'
    sub['PredictionString'] = sub.image_id.map(strings).fillna('')
    sub.to_csv(out, index=False)
    print('saved', out, sub.shape, 'frames without preds:', (sub.PredictionString == '').sum())
    return sub
