# %% [markdown]
# 03 — Anchor-free baseline: YOLO11s (Ultralytics), first valid submission.
# Kaggle: Accelerator = GPU T4, Internet = ON, add the competition data as Input.
# Run common.py as the first cell.

# %%
!pip -q install ultralytics pycocotools
import yaml
from ultralytics import YOLO

# Fixed protocol (keep identical for every detector so results are comparable)
TRAIN_STRIDE = 5      # every 5th frame of train/ (~4.8k imgs): neighbours are near-duplicates
IMGSZ = 1280          # objects are 10-30 px, 640 would shrink most of them to a few pixels
EPOCHS = 20
RUN = 'yolo11s_baseline'
DEBUG = True          # smoke test (~2 min): few frames, 1 epoch. Set False for the real run.
if DEBUG:
    TRAIN_STRIDE, EPOCHS, RUN = 100, 1, RUN + '_debug'
VAL_STRIDE = 50 if DEBUG else 1

# %% Dataset config. Ultralytics finds labels by replacing /images/ with /labels/ in the path.
train_list = f'{WORK}/train_s{TRAIN_STRIDE}.txt'
val_list = f'{WORK}/val_s{VAL_STRIDE}.txt'
open(train_list, 'w').write('\n'.join(frame_paths('train', TRAIN_STRIDE)))
open(val_list, 'w').write('\n'.join(frame_paths('val', VAL_STRIDE)))
yaml.safe_dump({'path': WORK, 'train': train_list, 'val': val_list, 'names': dict(enumerate(CLASSES))},
               open(f'{WORK}/data.yaml', 'w'))

# %% Train (baseline: ignored regions NOT handled yet -> masking them is a later experiment)
model = YOLO('yolo11s.pt')
model.train(data=f'{WORK}/data.yaml', imgsz=IMGSZ, epochs=EPOCHS, batch=8, seed=SEED, deterministic=True,
            cos_lr=True, close_mosaic=5, workers=4, project=f'{WORK}/runs', name=RUN, exist_ok=True)

# %% Proper COCO eval on val (respects ignored regions, maxDets=500)
model = YOLO(f'{WORK}/runs/{RUN}/weights/best.pt')
val_preds = predict_ultralytics(model, frame_paths('val', VAL_STRIDE), imgsz=IMGSZ)
print(RUN, coco_eval(val_preds))

# %% Predict test and write submission.csv (download it from Output, submit on Kaggle)
tp = test_paths()
write_submission(predict_ultralytics(model, tp[:64] if DEBUG else tp, imgsz=IMGSZ))
