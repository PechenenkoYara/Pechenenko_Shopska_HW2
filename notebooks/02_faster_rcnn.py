# %% [markdown]
# 02 — Two-stage anchor-based: Faster R-CNN (torchvision, ResNet50-FPN v2, COCO weights).
# Kaggle: Accelerator = GPU T4 (or P100), Internet = ON, add the competition data as Input.
# Run common.py as the first cell. Set DEBUG = False for the real run.
# Sources: TorchVision object detection finetuning tutorial and references/detection
# (github.com/pytorch/vision/tree/main/references/detection): model builder, collate, SGD + warmup, AMP.
# Model: Ren et al., "Faster R-CNN", NeurIPS 2015; Lin et al., "Feature Pyramid Networks", CVPR 2017.

# %%
!pip -q install pycocotools
import time, math, pickle
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image, ImageReadMode
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.anchor_utils import AnchorGenerator

# %% Config. Fixed protocol: every 5th train frame, full val/, long side 1280, seed 42.
# Defaults below = tuned model (frcnn_short). Each experiment changed ONE knob vs frcnn_short (val mAP, full val/):
#   frcnn_baseline: EPOCHS=20, LR_STEPS=[16, 19], EVAL_EVERY=5, last epoch  -> 0.1924, val peaked at epoch 5 then overfit
#   frcnn_short:    EPOCHS=8,  LR_STEPS=[5, 7], best epoch by quick val     -> 0.2273 (tuned)
#   frcnn_anchors:  + ANCHOR_SIZES=(16, 32, 64, 128, 256)                   -> 0.2177, AP_small unchanged
#   frcnn_giou:     + BOX_LOSS='giou'                                       -> 0.2234, within noise
#   frcnn_mask:     + MASK_IGNORE=True                                      -> 0.2158
RUN = 'frcnn_short'
TRAIN_STRIDE, VAL_STRIDE = 5, 1
EPOCHS = 8
BATCH = 4                 # T4 16 GB with AMP; use 2 on CUDA OOM
LR = 0.005                # reference recipe: 0.02 for 16 images -> linear scaling to batch 4
LR_STEPS = [5, 7]         # step decay x0.1 at these epochs
WARMUP_ITERS = 500
ANCHOR_SIZES = None       # None = torchvision default (32, 64, 128, 256, 512), one size per FPN level
BOX_LOSS = 'smooth_l1'    # RoI head box loss: 'smooth_l1' (torchvision default) or 'giou'
GIOU_WEIGHT = 10.0        # as in MMDetection faster_rcnn_r50_fpn_giou_1x (loss_weight=10)
MASK_IGNORE = False       # data experiment: fill ignored regions with gray 114 on train images
EVAL_EVERY = 1            # quick val mAP (every 10th val frame) each epoch -> curve + best checkpoint
RESUME = None             # path to a last.pt from a previous run (Kaggle sessions stop after 12 h)
USE_WANDB = False
DEBUG = True              # smoke test: ~250 train frames, 1 epoch, 57 val frames, 16 test frames
if DEBUG:
    TRAIN_STRIDE, VAL_STRIDE, EPOCHS, EVAL_EVERY, RUN = 100, 50, 1, 1, RUN + '_debug'

OUT = f'{WORK}/{RUN}'
os.makedirs(OUT, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
AMP = DEVICE.type == 'cuda'
CFG = dict(run=RUN, train_stride=TRAIN_STRIDE, epochs=EPOCHS, batch=BATCH, lr=LR, lr_steps=LR_STEPS,
           warmup=WARMUP_ITERS, anchors=ANCHOR_SIZES, box_loss=BOX_LOSS, mask_ignore=MASK_IGNORE, seed=SEED)

# %% Annotations from annotations_coco.json.
# category_id is 1-based, which is exactly torchvision's convention (label 0 = background).
# Ignored regions (iscrowd=1) are stored once per class in the json, so they are deduplicated here.
def load_annotations(split):
    ds = json.load(open(f'{ROOT}/{split}/annotations_coco.json'))
    cat2label = {c['id']: CLASSES.index(c['name']) + 1 for c in ds['categories']}
    id2key = {im['id']: key(im['file_name']) for im in ds['images']}
    objs, ignore = defaultdict(list), defaultdict(set)
    for a in ds['annotations']:
        x, y, w, h = a['bbox']
        k = id2key[a['image_id']]
        if a.get('iscrowd', 0):
            ignore[k].add((x, y, w, h))
        elif w >= 1 and h >= 1:                         # drop degenerate boxes
            objs[k].append((x, y, x + w, y + h, cat2label[a['category_id']]))
    return objs, ignore


class VisDroneDet(Dataset):
    """Returns (image tensor 0..1, target dict in torchvision format, path)."""
    def __init__(self, paths, objs=None, ignore=None, train=False):
        self.paths, self.objs, self.ignore, self.train = paths, objs or {}, ignore or {}, train

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        p = self.paths[i]
        k = key(p)
        img = read_image(p, ImageReadMode.RGB).float() / 255
        _, H, W = img.shape
        o = self.objs.get(k, [])
        boxes = torch.tensor([b[:4] for b in o], dtype=torch.float32).reshape(-1, 4)
        labels = torch.tensor([b[4] for b in o], dtype=torch.int64)
        if self.train:
            if MASK_IGNORE:                              # model never sees unlabeled objects as background
                for x, y, w, h in self.ignore.get(k, ()):
                    img[:, int(y):math.ceil(y + h), int(x):math.ceil(x + w)] = 114 / 255
            if random.random() < 0.5:                    # horizontal flip, the only augmentation
                img = img.flip(-1)
                boxes[:, [0, 2]] = W - boxes[:, [2, 0]]
        boxes[:, 0::2] = boxes[:, 0::2].clamp(0, W)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0, H)
        return img, {'boxes': boxes, 'labels': labels}, p


def collate(batch):
    return tuple(zip(*batch))

# %% GIoU box loss for the RoI head (Rezatofighi et al., CVPR 2019).
# torchvision's fastrcnn_loss gets only encoded deltas, not the proposals. Decoding a delta against a unit
# reference box differs from decoding against the real proposal by an affine map (shift + per-axis scale),
# which leaves IoU and GIoU unchanged, so GIoU can be computed from deltas alone.
import torch.nn.functional as F
import torchvision.models.detection.roi_heads as tv_roi_heads
from torchvision.ops import generalized_box_iou_loss

def _decode_unit(d, wts=(10., 10., 5., 5.), clip=math.log(1000. / 16)):
    dx, dy = d[:, 0] / wts[0], d[:, 1] / wts[1]
    w, h = (d[:, 2] / wts[2]).clamp(max=clip).exp(), (d[:, 3] / wts[3]).clamp(max=clip).exp()
    return torch.stack([dx - w / 2, dy - h / 2, dx + w / 2, dy + h / 2], 1)

def fastrcnn_loss_giou(class_logits, box_regression, labels, regression_targets):
    labels, regression_targets = torch.cat(labels, 0), torch.cat(regression_targets, 0)
    loss_cls = F.cross_entropy(class_logits, labels)
    pos = torch.where(labels > 0)[0]
    box_regression = box_regression.reshape(class_logits.shape[0], -1, 4)[pos, labels[pos]]
    loss_box = generalized_box_iou_loss(_decode_unit(box_regression.float()),
                                        _decode_unit(regression_targets[pos].float()), reduction='sum')
    return loss_cls, GIOU_WEIGHT * loss_box / labels.numel()   # same normalisation as torchvision

_fastrcnn_loss_default = tv_roi_heads.fastrcnn_loss

# %% Model. Resize: frames are 16:9, so min_size=720 / max_size=1280 gives long side 1280 (same scale as YOLO).
def build_model():
    m = fasterrcnn_resnet50_fpn_v2(
        weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1,
        min_size=720, max_size=1280,
        box_detections_per_img=500,    # default 100 cuts recall: ~19% of frames have >100 objects
        box_score_thresh=0.001,        # keep low-score boxes for mAP, like YOLO conf=0.001
        box_nms_thresh=0.6,            # same NMS IoU as predict_ultralytics
    )
    in_f = m.roi_heads.box_predictor.cls_score.in_features
    m.roi_heads.box_predictor = FastRCNNPredictor(in_f, len(CLASSES) + 1)   # new head: 8 classes + bg
    tv_roi_heads.fastrcnn_loss = fastrcnn_loss_giou if BOX_LOSS == 'giou' else _fastrcnn_loss_default
    if ANCHOR_SIZES:   # same 3 aspect ratios per level -> RPN head shapes unchanged, pretrained weights still load
        m.rpn.anchor_generator = AnchorGenerator(tuple((s,) for s in ANCHOR_SIZES), ((0.5, 1.0, 2.0),) * 5)
    return m.to(DEVICE)

# %% Prediction in the shared format {path: [(cls0, score, x, y, w, h), ...]}.
# Inference runs in fp32: fp16 has ~1 px resolution at x~1500, which matters for 10-30 px objects.
@torch.no_grad()
def predict_frcnn(model, paths, bs=8):
    model.eval()
    dl = DataLoader(VisDroneDet(paths), batch_size=bs, num_workers=4, collate_fn=collate)
    out = {}
    for imgs, _, ps in dl:
        res = model([im.to(DEVICE) for im in imgs])     # boxes are already in original-image pixels
        for p, r in zip(ps, res):
            b = r['boxes'].cpu().numpy(); s = r['scores'].cpu().numpy(); c = r['labels'].cpu().numpy() - 1
            out[p] = [(int(ci), float(si), float(x1), float(y1), float(x2 - x1), float(y2 - y1))
                      for (x1, y1, x2, y2), si, ci in zip(b, s, c)]
    return out


@torch.no_grad()
def measure_fps(model, paths, n=100):
    """Batch 1, fp32, model resize/normalize included, disk read excluded."""
    model.eval()
    imgs = [read_image(p, ImageReadMode.RGB).float().div(255).to(DEVICE) for p in paths[:n]]
    for im in imgs[:10]:
        model([im])                                      # warm-up
    if DEVICE.type == 'cuda': torch.cuda.synchronize()
    t0 = time.time()
    for im in imgs:
        model([im])
    if DEVICE.type == 'cuda': torch.cuda.synchronize()
    return round(len(imgs) / (time.time() - t0), 2)

# %% Data
train_objs, train_ignore = load_annotations('train')
train_paths = [p for p in frame_paths('train', TRAIN_STRIDE) if key(p) in train_objs or key(p) in train_ignore]
val_paths = frame_paths('val', VAL_STRIDE)
val_quick = frame_paths('val', max(VAL_STRIDE, 10))
g = torch.Generator().manual_seed(SEED)
train_dl = DataLoader(VisDroneDet(train_paths, train_objs, train_ignore, train=True), batch_size=BATCH,
                      shuffle=True, num_workers=4, collate_fn=collate, generator=g, drop_last=True)
print(f'train frames: {len(train_paths)} | iters/epoch: {len(train_dl)} | val frames: {len(val_paths)}')

# %% Optimizer and LR schedule (per-iteration: linear warmup, then step decay)
model = build_model()
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.SGD(params, lr=LR, momentum=0.9, weight_decay=1e-4)
iters_per_epoch = len(train_dl)

def lr_factor(it):
    warm = min(1.0, 0.001 + 0.999 * it / WARMUP_ITERS)
    epoch = it // iters_per_epoch
    return warm * 0.1 ** sum(epoch >= s for s in LR_STEPS)

sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
scaler = torch.amp.GradScaler('cuda', enabled=AMP)
start_epoch, history, best_map = 0, [], -1.0
if RESUME:
    ck = torch.load(RESUME, map_location=DEVICE)
    model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt'])
    sched.load_state_dict(ck['sched']); scaler.load_state_dict(ck['scaler'])
    start_epoch, history = ck['epoch'] + 1, ck['history']
    best_map = max([h.get('val_quick_mAP', -1) for h in history] + [-1])

if USE_WANDB:
    import wandb
    from kaggle_secrets import UserSecretsClient
    os.environ['WANDB_API_KEY'] = UserSecretsClient().get_secret('WANDB_API_KEY')
    wandb.init(project='visdrone-hw2', name=RUN, config=CFG)

# %% Train
t_train = time.time()
for epoch in range(start_epoch, EPOCHS):
    model.train()
    t0, sums = time.time(), defaultdict(float)
    for imgs, tgts, _ in train_dl:
        imgs = [im.to(DEVICE, non_blocking=True) for im in imgs]
        tgts = [{k: v.to(DEVICE) for k, v in t.items()} for t in tgts]
        with torch.autocast(DEVICE.type, enabled=AMP):
            losses = model(imgs, tgts)          # RPN: objectness + rpn_box_reg; RoI head: classifier + box_reg
            loss = sum(losses.values())
        assert math.isfinite(loss.item()), f'loss is {loss.item()}'
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update(); sched.step()
        for k, v in losses.items():
            sums[k] += v.item() / iters_per_epoch
    rec = {'epoch': epoch + 1, 'min': round((time.time() - t0) / 60, 1), 'lr': opt.param_groups[0]['lr'],
           **{k: round(v, 4) for k, v in sums.items()}}
    if (epoch + 1) % EVAL_EVERY == 0 or epoch + 1 == EPOCHS:
        rec['val_quick_mAP'] = coco_eval(predict_frcnn(model, val_quick), verbose=False)['mAP']
        if rec['val_quick_mAP'] > best_map:     # keep the best epoch (same idea as Ultralytics best.pt)
            best_map = rec['val_quick_mAP']
            torch.save(model.state_dict(), f'{OUT}/best.pt')
    history.append(rec)
    print(rec)
    if USE_WANDB: wandb.log(rec)
    torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(),
                'scaler': scaler.state_dict(), 'epoch': epoch, 'history': history}, f'{OUT}/last.pt')
train_min = round((time.time() - t_train) / 60, 1)

# %% Full validation of the best epoch: mAP / AP50 / AP_small + FPS (numbers for the report table)
model.load_state_dict(torch.load(f'{OUT}/best.pt', map_location=DEVICE))
best_epoch = max(history, key=lambda h: h.get('val_quick_mAP', -1))['epoch']
print('best epoch:', best_epoch)
val_preds = predict_frcnn(model, val_paths)
metrics = coco_eval(val_preds)
metrics.update(fps=measure_fps(model, val_paths), train_min=train_min, best_epoch=best_epoch)
print(RUN, metrics)
json.dump({'config': CFG, 'metrics': metrics, 'history': history}, open(f'{OUT}/results.json', 'w'), indent=1)
pickle.dump(val_preds, open(f'{OUT}/val_preds.pkl', 'wb'))       # reused for failure cases and WBF
if USE_WANDB: wandb.summary.update(metrics); wandb.finish()

# %% Test predictions -> submission.csv (download from Output and submit on Kaggle)
tp = test_paths()
test_preds = predict_frcnn(model, tp[:16] if DEBUG else tp)
pickle.dump(test_preds, open(f'{OUT}/test_preds.pkl', 'wb'))
write_submission(test_preds, out=f'{OUT}/submission.csv')
