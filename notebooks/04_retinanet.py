# %% [markdown]
# 04 — One-stage anchor-based: RetinaNet (torchvision, ResNet50-FPN v2, COCO weights).
# Kaggle: Accelerator = GPU T4, Internet = ON, competition data as Input. Run common.py as the first cell.
# Same protocol as 02_faster_rcnn (every 5th train frame, full val/, long side 1280, 8 epochs, best epoch by quick val).
# Sources: TorchVision detection reference scripts (github.com/pytorch/vision/tree/main/references/detection);
# Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017; anchors from k-means as in YOLOv2 (Redmon & Farhadi, 2017).

# %%
!pip -q install pycocotools
import time, math, pickle
from functools import partial
from collections import defaultdict
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image, ImageReadMode
from torchvision.ops import sigmoid_focal_loss
from torchvision.models.detection import retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights
from torchvision.models.detection.anchor_utils import AnchorGenerator
import torchvision.models.detection.retinanet as tv_retina

# %% Config. Same protocol as 02_faster_rcnn. Defaults below = tuned model (retina_focal).
# Each experiment changed ONE knob vs retina_baseline (val mAP, full val/):
#   retina_baseline: ANCHOR_BASE=None (32..512), FOCAL_ALPHA=0.25, FOCAL_GAMMA=2.0  -> 0.1723
#   retina_anchors:  ANCHOR_BASE=(16, 32, 64, 128, 256), sizes from k-means (anchors.json) -> 0.1598 (AP75 dropped)
#   retina_focal:    FOCAL_GAMMA=1.5                                                -> 0.1751 (tuned, within noise)
RUN = 'retina_focal'
TRAIN_STRIDE, VAL_STRIDE = 5, 1
EPOCHS = 8
BATCH = 4                 # T4 16 GB with AMP; use 2 on CUDA OOM
LR = 0.0025               # reference recipe: 0.01 for 16 images -> linear scaling to batch 4
LR_STEPS = [5, 7]         # same schedule as tuned Faster R-CNN (frcnn_short)
WARMUP_ITERS = 500
ANCHOR_BASE = None        # None = torchvision default: base sizes (32, 64, 128, 256, 512) on P3..P7
FOCAL_ALPHA = 0.25        # focal loss defaults from the RetinaNet paper
FOCAL_GAMMA = 1.5
FG_IOU, BG_IOU = 0.5, 0.4 # anchor is positive if IoU >= FG_IOU, negative if < BG_IOU, ignored in between
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
           warmup=WARMUP_ITERS, anchor_base=ANCHOR_BASE, focal_alpha=FOCAL_ALPHA, focal_gamma=FOCAL_GAMMA,
           fg_iou=FG_IOU, bg_iou=BG_IOU, mask_ignore=MASK_IGNORE, seed=SEED)

# %% Annotations from annotations_coco.json.
# RetinaNet classifies with per-class sigmoids (no background class), so labels are 0-based.
# Ignored regions (iscrowd=1) are stored once per class in the json, so they are deduplicated here.
def load_annotations(split):
    ds = json.load(open(f'{ROOT}/{split}/annotations_coco.json'))
    cat2label = {c['id']: CLASSES.index(c['name']) for c in ds['categories']}
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

# %% Anchors. torchvision default: 3 octave scales (x, x*2^(1/3), x*2^(2/3)) x 3 ratios per level P3..P7.
def anchor_sizes(base):
    return tuple((b, int(b * 2 ** (1 / 3)), int(b * 2 ** (2 / 3))) for b in base)

ANCHOR_RATIOS = (0.5, 1.0, 2.0)      # h/w; k-means ratios on train lie in 0.54..1.97, so defaults already cover them


def anchor_coverage(base, n_frames=500):
    """Share of train boxes (at the 1280 input scale) whose best shape-only IoU with some anchor reaches FG_IOU / BG_IOU.
    Shape-only (centres aligned) = upper bound: real anchors also sit on a grid with stride 8..128."""
    ds = json.load(open(f'{ROOT}/train/annotations_coco.json'))
    ims = {im['id']: im for im in ds['images']}
    keep = set(sorted(ims)[::max(1, len(ims) // n_frames)])
    wh = np.array([[a['bbox'][2], a['bbox'][3], 1280 / max(ims[a['image_id']]['width'], ims[a['image_id']]['height'])]
                   for a in ds['annotations'] if not a.get('iscrowd', 0) and a['image_id'] in keep])
    wh = wh[:, :2] * wh[:, 2:]
    anc = np.array([(s / math.sqrt(r), s * math.sqrt(r)) for lvl in anchor_sizes(base) for s in lvl for r in ANCHOR_RATIOS])
    inter = np.minimum(wh[:, None, 0], anc[None, :, 0]) * np.minimum(wh[:, None, 1], anc[None, :, 1])
    best = (inter / (wh.prod(1)[:, None] + anc.prod(1)[None] - inter)).max(1)
    return {'positive (IoU>=fg)': round(float((best >= FG_IOU).mean()), 3),
            'negative only (IoU<bg)': round(float((best < BG_IOU).mean()), 3), 'median best IoU': round(float(np.median(best)), 3)}

for base in [(32, 64, 128, 256, 512), (16, 32, 64, 128, 256)]:
    print('anchor base', base, anchor_coverage(base))

# %% Model. Resize: frames are 16:9, so min_size=720 / max_size=1280 gives long side 1280 (same scale as YOLO / Faster R-CNN).
NUM_ANCHORS = 3 * len(ANCHOR_RATIOS)

def build_model():
    # focal loss parameters: torchvision calls ops.sigmoid_focal_loss with its defaults -> patch the module reference
    tv_retina.sigmoid_focal_loss = partial(sigmoid_focal_loss, alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
    m = retinanet_resnet50_fpn_v2(
        weights=RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1,
        min_size=720, max_size=1280,
        detections_per_img=500,        # same cap as the leaderboard and the other detectors
        score_thresh=0.001,            # keep low-score boxes for mAP
        nms_thresh=0.6,                # same NMS IoU as YOLO / Faster R-CNN
        fg_iou_thresh=FG_IOU, bg_iou_thresh=BG_IOU,
    )
    if ANCHOR_BASE:                    # same 9 anchors per location -> pretrained head shapes still fit
        m.anchor_generator = AnchorGenerator(anchor_sizes(ANCHOR_BASE), (ANCHOR_RATIOS,) * len(ANCHOR_BASE))
    # new class logits only (8 classes); keep the pretrained conv tower. Prior-probability init from the focal loss paper.
    cls = m.head.classification_head
    cls.cls_logits = nn.Conv2d(cls.cls_logits.in_channels, NUM_ANCHORS * len(CLASSES), 3, padding=1)
    nn.init.normal_(cls.cls_logits.weight, std=0.01)
    nn.init.constant_(cls.cls_logits.bias, -math.log((1 - 0.01) / 0.01))
    cls.num_classes = len(CLASSES)
    return m.to(DEVICE)

# %% Prediction in the shared format {path: [(cls0, score, x, y, w, h), ...]}.
# Inference runs in fp32: fp16 has ~1 px resolution at x~1500, which matters for 10-30 px objects.
@torch.no_grad()
def predict_retina(model, paths, bs=8):
    model.eval()
    dl = DataLoader(VisDroneDet(paths), batch_size=bs, num_workers=4, collate_fn=collate)
    out = {}
    for imgs, _, ps in dl:
        res = model([im.to(DEVICE) for im in imgs])     # boxes are already in original-image pixels
        for p, r in zip(ps, res):
            b = r['boxes'].cpu().numpy(); s = r['scores'].cpu().numpy(); c = r['labels'].cpu().numpy()          # already 0-based
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
            losses = model(imgs, tgts)          # classification (focal) + bbox_regression (GIoU in v2)
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
        rec['val_quick_mAP'] = coco_eval(predict_retina(model, val_quick), verbose=False)['mAP']
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
val_preds = predict_retina(model, val_paths)
metrics = coco_eval(val_preds)
metrics.update(fps=measure_fps(model, val_paths), train_min=train_min, best_epoch=best_epoch)
print(RUN, metrics)
json.dump({'config': CFG, 'metrics': metrics, 'history': history}, open(f'{OUT}/results.json', 'w'), indent=1)
pickle.dump(val_preds, open(f'{OUT}/val_preds.pkl', 'wb'))       # reused for failure cases and WBF
if USE_WANDB: wandb.summary.update(metrics); wandb.finish()

# %% Test predictions -> submission.csv (download from Output and submit on Kaggle)
tp = test_paths()
test_preds = predict_retina(model, tp[:16] if DEBUG else tp)
pickle.dump(test_preds, open(f'{OUT}/test_preds.pkl', 'wb'))
write_submission(test_preds, out=f'{OUT}/submission.csv')
