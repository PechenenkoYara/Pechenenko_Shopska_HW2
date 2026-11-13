# %% [markdown]
# 05 — Query-based detector: RT-DETR-l (Ultralytics), same data and budget as the YOLO baseline.
# Kaggle: Accelerator = GPU T4, Internet = ON, Input = competition data. Run common.py as the first cell.
# Sources: Zhao et al., "DETRs Beat YOLOs on Real-time Object Detection", CVPR 2024; Ultralytics RT-DETR docs.

# %%
!pip -q install ultralytics pycocotools
import pickle, time, warnings
from pathlib import Path
import cv2, yaml
import matplotlib.pyplot as plt
from ultralytics import RTDETR
from ultralytics.utils import ROOT as UL_ROOT
warnings.filterwarnings('ignore', message='.*get_trial_id.*')   # Ultralytics Ray Tune callback: Kaggle has ray installed, we do not tune

# One change per experiment. baseline = Ultralytics defaults; fixwarmup_* and giou4 = fixwarmup + one change.
EXPERIMENTS = {'baseline': dict(nq=300, lr0=1e-4),
               'fixwarmup': dict(nq=300, lr0=1e-4, warmup_bias_lr=0.0),   # baseline + bias warmup fix (see train cell): full val 0.2934 (final)
               'q500': dict(nq=500, lr0=1e-4),                                   # not run (broken warmup, see fixwarmup)
               'fixwarmup_q500': dict(nq=500, lr0=1e-4, warmup_bias_lr=0.0),   # fixwarmup + 500 queries (up to 180 GT/frame): not run, no GPU left
               'lr5e-5': dict(nq=300, lr0=5e-5),                                 # not run
               # Loss experiment: no denoising queries -> no auxiliary DN loss (RT-DETR default nd=100; DN-DETR, Li et al. 2022).
               # Kaggle T4, best epoch 10, stopped at 16 (patience 6): mAP 0.2969 (fixwarmup 0.2934), AP_small 0.1107 (0.1385),
               # AP_large 0.4975 (0.4904), public 0.30325 (fixwarmup 0.29676), 246 min -> DN helps small objects and the first epochs.
               'fixwarmup_nodn': dict(nq=300, lr0=1e-4, warmup_bias_lr=0.0, nd=0),
               # Loss experiment: GIoU weight 2 -> 4 in the loss and the matching cost (loss cell). Trained on Colab T4, stopped
               # after 8/20 epochs by the Colab GPU limit; evaluated in 05b_rtdetr_eval_checkpoint.py: mAP 0.2928 (fixwarmup
               # 0.2934), AP_small 0.1245 (0.1385), AP_large 0.5030 (0.4904) -> not better overall: helps large boxes, hurts small.
               'giou4': dict(nq=300, lr0=1e-4, warmup_bias_lr=0.0, giou_gain=4)}
EXP = 'fixwarmup'
TRAIN_STRIDE = 5
IMGSZ = 1280
EPOCHS = 20
PATIENCE = 6
BATCH = 4             # RT-DETR at 1280 needs more memory than YOLO; if it runs out: 2, and write it in the journal
TRAIN_VAL_STRIDE = 10 # val during training on every 10th frame, final metrics on the full val/
RUN = f'rtdetr_l_{EXP}'
DEBUG = True          # smoke test: few frames, 1 epoch. Set False for the real run.
if DEBUG:
    TRAIN_STRIDE, TRAIN_VAL_STRIDE, EPOCHS, RUN = 100, 50, 1, RUN + '_debug'
VAL_STRIDE = 50 if DEBUG else 1
cfg = EXPERIMENTS[EXP]

# %% Dataset config (same train frames as the baseline)
train_list, val_list = f'{WORK}/train_s{TRAIN_STRIDE}.txt', f'{WORK}/val_s{TRAIN_VAL_STRIDE}.txt'
open(train_list, 'w').write('\n'.join(frame_paths('train', TRAIN_STRIDE)))
open(val_list, 'w').write('\n'.join(frame_paths('val', TRAIN_VAL_STRIDE)))
yaml.safe_dump({'path': WORK, 'train': train_list, 'val': val_list, 'names': dict(enumerate(CLASSES))},
               open(f'{WORK}/data.yaml', 'w'))

# %% Model: the number of queries is an argument of RTDETRDecoder -> edit it in rtdetr-l.yaml
def build_model(nq):
    if nq == 300:
        return RTDETR('rtdetr-l.pt')
    m = yaml.safe_load(open(UL_ROOT / 'cfg/models/rt-detr/rtdetr-l.yaml'))
    m['head'][-1][3] = ['nc', 256, nq]         # RTDETRDecoder(nc, ch, hd=256, nq=300, ...), ch is inserted by Ultralytics
    path = f'{WORK}/rtdetr-l-q{nq}.yaml'
    yaml.safe_dump(m, open(path, 'w'))
    return RTDETR(path).load('rtdetr-l.pt')


# %% Loss weights. Ultralytics builds the loss in RTDETRDetectionModel.init_criterion with fixed gains
# (loss: class 1, L1 bbox 5, GIoU 2; Hungarian matching cost: class 2, L1 bbox 5, GIoU 2), so we override that method.
# Why GIoU: L1 is computed on normalized coordinates, so a 12 px box at 1280 px gives tiny L1 errors and little gradient,
# while GIoU is scale-invariant. A higher GIoU weight in both the loss and the matching cost gives small objects more say.
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.models.utils.loss import RTDETRDetectionLoss
GIOU_GAIN = cfg.get('giou_gain', 2)   # 2 = Ultralytics default

def init_criterion(self):
    crit = RTDETRDetectionLoss(nc=getattr(self, 'nc', len(CLASSES)), use_vfl=True, loss_gain={'class': 1, 'bbox': 5, 'giou': GIOU_GAIN})
    crit.matcher.cost_gain['giou'] = GIOU_GAIN
    return crit

RTDETRDetectionModel.init_criterion = init_criterion

# %% Train. optimizer=auto would pick lr 0.00083, too high for a DETR. deterministic=True does not support F.grid_sample.
model = build_model(cfg['nq'])
crit = RTDETRDetectionModel.init_criterion(model.model)   # check the GIoU override before spending GPU time
assert crit.loss_gain['giou'] == crit.matcher.cost_gain['giou'] == GIOU_GAIN

# The trainer rebuilds the model from yaml and loads our weights, so the head is changed in a callback on trainer.model.
# RTDETRDecoder.num_denoising -> ops.get_cdn_group returns no DN queries -> RTDETRDetectionLoss sets *_dn losses to 0.
def set_denoising(trainer):
    head = getattr(trainer.model, 'module', trainer.model).model[-1]
    head.num_denoising = cfg['nd']
    print('num_denoising =', head.num_denoising)

if 'nd' in cfg:
    model.add_callback('on_pretrain_routine_end', set_denoising)
t0 = time.time()
model.train(data=f'{WORK}/data.yaml', imgsz=IMGSZ, epochs=EPOCHS, patience=PATIENCE, batch=BATCH, seed=SEED,
            optimizer='AdamW', lr0=cfg['lr0'], weight_decay=1e-4, deterministic=False, cos_lr=True, close_mosaic=5,
            workers=4, project=f'{WORK}/runs', name=RUN, exist_ok=True,
            # Ultralytics default warmup_bias_lr=0.1 is meant for SGD on YOLO. With AdamW (lr 1e-4) it gave all biases
            # ~700x the weight LR for 2 epochs (results.csv lr/pg2 0.067 vs lr/pg0 3.3e-5): cls_loss ~0 and mAP 0 for 12 epochs.
            warmup_bias_lr=cfg.get('warmup_bias_lr', 0.1))
train_min = (time.time() - t0) / 60
print('loss gains used in training:', model.trainer.model.criterion.loss_gain)   # should show giou = GIOU_GAIN
best = str(model.trainer.best)
assert 'nd' not in cfg or getattr(model.trainer.model, 'module', model.trainer.model).model[-1].num_denoising == cfg['nd']

# %% Predictions first: they survive if the session dies during the slow COCO eval
model = RTDETR(best)
val_preds = predict_ultralytics(model, frame_paths('val', VAL_STRIDE), imgsz=IMGSZ, bs=8)
pickle.dump(val_preds, open(f'{WORK}/val_preds_{RUN}.pkl', 'wb'))
tp = test_paths()
test_preds = predict_ultralytics(model, tp[:16] if DEBUG else tp, imgsz=IMGSZ, bs=8)
pickle.dump(test_preds, open(f'{WORK}/test_preds_{RUN}.pkl', 'wb'))
write_submission(test_preds)

# %% FPS: batch 1, fp32, frames already in memory (same protocol as 02_faster_rcnn.py)
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

# %% Proper COCO eval on val
metrics = coco_eval(val_preds, verbose=False)
json.dump({'run': RUN, 'config': cfg, 'best': best, 'train_min': round(train_min, 1), 'fps': fps, **metrics},
          open(f'{WORK}/results_{RUN}.json', 'w'), indent=1)
print(RUN, metrics, '| fps', fps, '| train min', round(train_min, 1))

# %% Val mAP per epoch vs the CNN baseline. Built-in Ultralytics metric: for the curve only, report numbers come from coco_eval.
# YOLO11s baseline curve: results.csv of 03_yolo_baseline.py (full val/). RT-DETR curve: every 10th val frame.
YOLO_BASELINE_MAP = [0.19156, 0.19051, 0.19172, 0.19309, 0.1863, 0.18018, 0.19372, 0.20222, 0.19738, 0.19009,
                     0.1865, 0.1835, 0.18567, 0.19447, 0.18527, 0.18458, 0.18411, 0.17988, 0.1808, 0.17928]
hist = pd.read_csv(Path(best).parents[1] / 'results.csv')
plt.figure(figsize=(7, 4))
plt.plot(range(1, len(YOLO_BASELINE_MAP) + 1), YOLO_BASELINE_MAP, marker='o', label='YOLO11s baseline')
plt.plot(hist.epoch, hist['metrics/mAP50-95(B)'], marker='o', label=RUN)
plt.xlabel('epoch'); plt.ylabel('val mAP50-95 (Ultralytics)'); plt.legend(); plt.tight_layout()
plt.savefig(f'{WORK}/curve_{RUN}.png', dpi=130)
