# %% [markdown]
# 03b — YOLO11 with an extra P2/4 detection head. SCALE='s': one change against 03_yolo_baseline.py; 'm': bigger model.
# Why: objects of 13-20 px cover only 2-3 cells on P3/8, a stride-4 map gives them twice the resolution.
# Kaggle: Accelerator = GPU T4, Internet = ON, Input = competition data. Run common.py as the first cell.
# Head layout: Ultralytics yolov8-p2.yaml; blocks: Ultralytics yolo11.yaml. Ultralytics ships no YOLO11 P2 config.

# %%
!pip -q install ultralytics pycocotools
import pickle, time
import cv2, yaml
from ultralytics import YOLO

# Same protocol as the baseline. patience only stops a run that stopped improving (best epoch was 8 of 20).
TRAIN_STRIDE = 5
IMGSZ = 1280
EPOCHS = 20
PATIENCE = 6
SCALE = 'm'           # 's' (done): val 0.2045, public 0.21219
BATCH = 6             # 's' used 8. If the GPU runs out of memory: 4, and write it in the journal
RUN = f'yolo11{SCALE}_p2'
DEBUG = True          # smoke test: few frames, 1 epoch. Set False for the real run.
if DEBUG:
    TRAIN_STRIDE, EPOCHS, RUN = 100, 1, RUN + '_debug'
VAL_STRIDE = 50 if DEBUG else 1

# %% Model config: yolo11.yaml + P2 branch. The file name must keep `11s`: Ultralytics reads the scale from it.
P2_YAML = """
nc: 80
scales:
  s: [0.50, 0.50, 1024]
  m: [0.50, 1.00, 512]
backbone:
  - [-1, 1, Conv, [64, 3, 2]]            # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]]           # 1-P2/4
  - [-1, 2, C3k2, [256, False, 0.25]]    # 2
  - [-1, 1, Conv, [256, 3, 2]]           # 3-P3/8
  - [-1, 2, C3k2, [512, False, 0.25]]    # 4
  - [-1, 1, Conv, [512, 3, 2]]           # 5-P4/16
  - [-1, 2, C3k2, [512, True]]           # 6
  - [-1, 1, Conv, [1024, 3, 2]]          # 7-P5/32
  - [-1, 2, C3k2, [1024, True]]          # 8
  - [-1, 1, SPPF, [1024, 5]]             # 9
  - [-1, 2, C2PSA, [1024]]               # 10
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, False]]          # 13
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, C3k2, [256, False]]          # 16
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 2], 1, Concat, [1]]
  - [-1, 2, C3k2, [128, False]]          # 19 (P2/4)
  - [-1, 1, Conv, [128, 3, 2]]
  - [[-1, 16], 1, Concat, [1]]
  - [-1, 2, C3k2, [256, False]]          # 22 (P3/8)
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 13], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, False]]          # 25 (P4/16)
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 10], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]          # 28 (P5/32)
  - [[19, 22, 25, 28], 1, Detect, [nc]]
"""
model_yaml = f'{WORK}/yolo11{SCALE}-p2.yaml'
open(model_yaml, 'w').write(P2_YAML)

# %% Dataset config (same frames as the baseline)
train_list, val_list = f'{WORK}/train_s{TRAIN_STRIDE}.txt', f'{WORK}/val_s{VAL_STRIDE}.txt'
open(train_list, 'w').write('\n'.join(frame_paths('train', TRAIN_STRIDE)))
open(val_list, 'w').write('\n'.join(frame_paths('val', VAL_STRIDE)))
yaml.safe_dump({'path': WORK, 'train': train_list, 'val': val_list, 'names': dict(enumerate(CLASSES))},
               open(f'{WORK}/data.yaml', 'w'))

# %% Train: COCO weights go to the backbone and to every head layer with a matching shape
model = YOLO(model_yaml).load(f'yolo11{SCALE}.pt')
t0 = time.time()
model.train(data=f'{WORK}/data.yaml', imgsz=IMGSZ, epochs=EPOCHS, patience=PATIENCE, batch=BATCH, seed=SEED,
            deterministic=True, cos_lr=True, close_mosaic=5, workers=4, project=f'{WORK}/runs', name=RUN, exist_ok=True)
train_min = (time.time() - t0) / 60
best = str(model.trainer.best)

# %% Predictions first: they survive if the session dies during the slow COCO eval
model = YOLO(best)
val_preds = predict_ultralytics(model, frame_paths('val', VAL_STRIDE), imgsz=IMGSZ, bs=8)
pickle.dump(val_preds, open(f'{WORK}/val_preds_{RUN}.pkl', 'wb'))
tp = test_paths()
test_preds = predict_ultralytics(model, tp[:64] if DEBUG else tp, imgsz=IMGSZ, bs=8)
pickle.dump(test_preds, open(f'{WORK}/test_preds_{RUN}.pkl', 'wb'))
write_submission(test_preds)

# %% FPS: batch 1, fp32, frames already in memory (same protocol as 02_faster_rcnn.py)
def measure_fps(model, paths, warmup=10):
    ims = [cv2.imread(p) for p in paths]
    kw = dict(imgsz=IMGSZ, conf=0.001, iou=0.6, max_det=500, half=False, verbose=False)
    for im in ims[:warmup]:
        model.predict(im, **kw)
    t = time.time()
    for im in ims:
        model.predict(im, **kw)
    return round(len(ims) / (time.time() - t), 1)

fps = measure_fps(model, frame_paths('val', 20)[:200])

# %% Proper COCO eval on val
metrics = coco_eval(val_preds, verbose=False)
json.dump({'run': RUN, 'best': best, 'train_min': round(train_min, 1), 'fps': fps, **metrics},
          open(f'{WORK}/results_{RUN}.json', 'w'), indent=1)
print(RUN, metrics, '| fps', fps, '| train min', round(train_min, 1))
