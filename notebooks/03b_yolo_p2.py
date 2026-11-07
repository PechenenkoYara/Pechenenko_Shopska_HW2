# %% [markdown]
# 03b — YOLO11 with an extra P2/4 detection head. SCALE='s': one change against 03_yolo_baseline.py; 'm': bigger model.
# LOSS='nwd': box loss = 0.5 * CIoU loss + 0.5 * NWD loss, one change against the 's' + 'ciou' run.
# NWD: Wang et al., "A Normalized Gaussian Wasserstein Distance for Tiny Object Detection", arXiv 2110.13389.
# Why: objects of 13-20 px cover only 2-3 cells on P3/8, a stride-4 map gives them twice the resolution.
# Kaggle: Accelerator = GPU T4, Internet = ON, Input = competition data. Run common.py as the first cell.
# Head layout: Ultralytics yolov8-p2.yaml; blocks: Ultralytics yolo11.yaml. Ultralytics ships no YOLO11 P2 config.

# %%
!pip -q install ultralytics pycocotools
import pickle, time, warnings
import cv2, yaml
from ultralytics import YOLO
warnings.filterwarnings('ignore', message='.*get_trial_id.*')   # Ultralytics Ray Tune callback: Kaggle has ray installed, we do not tune

# Same protocol as the baseline. patience only stops a run that stopped improving (best epoch was 8 of 20).
TRAIN_STRIDE = 5
IMGSZ = 1280
EPOCHS = 20
PATIENCE = 6
SCALE = 's'           # 's' + 'ciou' (done): val 0.2045, public 0.21219; 'm' + 'ciou' (done): val 0.2079
LOSS = 'nwd'          # 'ciou' = Ultralytics default
BATCH = 8             # 'm' used 6. If the GPU runs out of memory: lower it and write it in the journal
RUN = f'yolo11{SCALE}_p2' + ('' if LOSS == 'ciou' else f'_{LOSS}')
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

# %% NWD box loss. CIoU of a 10 px box drops to ~0 after a 5 px shift, NWD decays smoothly with the distance.
# Boxes are Gaussians N((cx, cy), diag(w/2, h/2)^2); W2 = ||(cx1, cy1, w1/2, h1/2) - (cx2, cy2, w2/2, h2/2)||; NWD = exp(-W2 / C).
# C = mean box size, as in the paper; computed in pixels of the 1280 training scale.
import inspect
from ultralytics.utils import loss as ul_loss

NWD_RATIO = 0.5
ds = json.load(open(f'{ROOT}/train/annotations_coco.json'))
long_side = {im['id']: max(im['width'], im['height']) for im in ds['images']}
NWD_C = float(np.mean([np.sqrt(a['bbox'][2] * a['bbox'][3]) * IMGSZ / long_side[a['image_id']]
                       for a in ds['annotations'] if not a.get('iscrowd', 0)]))
del ds
print('NWD constant C =', round(NWD_C, 1), 'px')


def nwd(b1, b2, c):
    """NWD between xyxy boxes, shape (N, 4) -> (N,)."""
    g1 = torch.cat([(b1[:, :2] + b1[:, 2:]) / 2, (b1[:, 2:] - b1[:, :2]) / 2], 1)
    g2 = torch.cat([(b2[:, :2] + b2[:, 2:]) / 2, (b2[:, 2:] - b2[:, :2]) / 2], 1)
    return torch.exp(-torch.linalg.norm(g1 - g2, dim=1) / c)


_bbox_forward = ul_loss.BboxLoss.forward
assert 'stride' in inspect.signature(_bbox_forward).parameters, 'BboxLoss.forward has no stride: this Ultralytics version is not supported'


def bbox_forward_nwd(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum,
                     fg_mask, imgsz, stride):
    """Ultralytics BboxLoss.forward (CIoU + DFL) with the IoU term mixed with NWD. Boxes come in grid units -> x stride."""
    loss_iou, loss_dfl = _bbox_forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                       target_scores_sum, fg_mask, imgsz, stride)
    b, a = fg_mask.nonzero(as_tuple=True)
    s = stride[a].float()
    weight = target_scores[b, a].sum(-1)
    d = nwd(pred_bboxes[b, a].float() * s, target_bboxes[b, a].float() * s, NWD_C)
    loss_nwd = ((1.0 - d) * weight).sum() / target_scores_sum
    bbox_forward_nwd.calls += 1
    return (1 - NWD_RATIO) * loss_iou + NWD_RATIO * loss_nwd, loss_dfl


bbox_forward_nwd.calls = 0
if LOSS == 'nwd':
    ul_loss.BboxLoss.forward = bbox_forward_nwd
    box = torch.tensor([[0., 0., 10., 10.]])
    print('NWD check: same box', nwd(box, box, NWD_C).item(), '| 5 px shift', round(nwd(box, box + 5, NWD_C).item(), 3))


# %% Train: COCO weights go to the backbone and to every head layer with a matching shape
model = YOLO(model_yaml).load(f'yolo11{SCALE}.pt')
t0 = time.time()
model.train(data=f'{WORK}/data.yaml', imgsz=IMGSZ, epochs=EPOCHS, patience=PATIENCE, batch=BATCH, seed=SEED,
            deterministic=True, cos_lr=True, close_mosaic=5, workers=4, project=f'{WORK}/runs', name=RUN, exist_ok=True)
train_min = (time.time() - t0) / 60
assert LOSS == 'ciou' or bbox_forward_nwd.calls > 0, 'NWD loss was never called'
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
json.dump({'run': RUN, 'loss': LOSS, 'best': best, 'train_min': round(train_min, 1), 'fps': fps, **metrics},
          open(f'{WORK}/results_{RUN}.json', 'w'), indent=1)
print(RUN, metrics, '| fps', fps, '| train min', round(train_min, 1))
