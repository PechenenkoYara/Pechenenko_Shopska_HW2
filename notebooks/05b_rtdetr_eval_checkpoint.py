# %% [markdown]
# 05b — Full-val evaluation of an RT-DETR checkpoint trained elsewhere (giou4: 8 of 20 epochs on Colab T4,
# stopped by the Colab GPU limit). Same predictions, FPS and COCO eval as the last cells of 05_rtdetr.py.
# Kaggle: Accelerator = GPU T4, Internet = ON. Input = competition data + dataset with best.pt + results.csv of the run
# (+ notebook 05 fixwarmup output for the curve comparison). Run common.py as the first cell.

# %%
!pip -q install ultralytics pycocotools
import pickle, time
import cv2
import matplotlib.pyplot as plt
from ultralytics import RTDETR

RUN = 'rtdetr_l_giou4'
REF = 'rtdetr_l_fixwarmup'   # run to compare the training curve with (same budget, GIoU weight 2)
IMGSZ = 1280
DEBUG = True                 # smoke test: every 50th val frame, 16 test frames
VAL_STRIDE = 50 if DEBUG else 1


def find_run_file(fname, run):
    """First /kaggle/input/**/fname whose path mentions the run name (dataset slugs use '-' instead of '_')."""
    hits = [p for p in sorted(glob.glob(f'/kaggle/input/**/{fname}', recursive=True)) if run in p.replace('-', '_')]
    assert hits, f'{fname} of {run} not found in /kaggle/input: add it via Add Input'
    return hits[0]

best = find_run_file('best.pt', RUN)
print('checkpoint:', best)

# %% Predictions first: they survive if the session dies during the slow COCO eval
model = RTDETR(best)
val_preds = predict_ultralytics(model, frame_paths('val', VAL_STRIDE), imgsz=IMGSZ, bs=8)
pickle.dump(val_preds, open(f'{WORK}/val_preds_{RUN}.pkl', 'wb'))
tp = test_paths()
test_preds = predict_ultralytics(model, tp[:16] if DEBUG else tp, imgsz=IMGSZ, bs=8)
pickle.dump(test_preds, open(f'{WORK}/test_preds_{RUN}.pkl', 'wb'))
write_submission(test_preds)

# %% FPS: batch 1, fp32, frames already in memory (same protocol as 02_faster_rcnn.py and 05_rtdetr.py)
def measure_fps(model, paths, warmup=10):
    ims = [cv2.imread(p) for p in paths]
    kw = dict(imgsz=IMGSZ, conf=0.001, max_det=500, half=False, verbose=False)
    for im in ims[:warmup]:
        model.predict(im, **kw)
    t = time.time()
    for im in ims:
        model.predict(im, **kw)
    return round(len(ims) / (time.time() - t), 1)

fps = measure_fps(model, frame_paths('val', 20)[:200])

# %% COCO eval on val
metrics = coco_eval(val_preds, verbose=False)
json.dump({'run': RUN, 'best': best, 'fps': fps, **metrics}, open(f'{WORK}/results_{RUN}.json', 'w'), indent=1)
print(RUN, metrics, '| fps', fps)

# %% Training curves: giou4 vs fixwarmup (Ultralytics mAP on every 10th val frame; for the curve only)
cur = pd.read_csv(find_run_file('results.csv', RUN))
ref = pd.read_csv(find_run_file('results.csv', REF))
plt.figure(figsize=(7, 4))
plt.plot(ref.epoch, ref['metrics/mAP50-95(B)'], marker='o', label=f'{REF} (GIoU weight 2)')
plt.plot(cur.epoch, cur['metrics/mAP50-95(B)'], marker='o', label=f'{RUN} (GIoU weight 4)')
plt.xlabel('epoch'); plt.ylabel('val mAP50-95 (Ultralytics)'); plt.legend(); plt.tight_layout()
plt.savefig(f'{WORK}/curve_{RUN}_vs_{REF}.png', dpi=130)
both = cur[['epoch', 'metrics/mAP50-95(B)']].merge(ref[['epoch', 'metrics/mAP50-95(B)']], on='epoch', suffixes=('_giou4', '_fixwarmup'))
both
