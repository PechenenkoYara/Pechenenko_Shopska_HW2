# %% [markdown]
# 01 — EDA: classes, box sizes, density, per-clip variation, ignored regions, temporal redundancy.
# Kaggle: Accelerator = None (CPU is enough). Run common.py as the first cell.
# All figures are saved to /kaggle/working/figs -> download and put the needed ones into images/.

# %%
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

FIG = f'{WORK}/figs'; os.makedirs(FIG, exist_ok=True)
COLORS = plt.cm.tab10(np.arange(len(CLASSES)))
def save(name): plt.tight_layout(); plt.savefig(f'{FIG}/{name}.png', dpi=130, bbox_inches='tight'); plt.show()

# %% Sanity check of the raw files (format of file_name / ids before trusting common.py mappings)
coco = {s: json.load(open(f'{ROOT}/{s}/annotations_coco.json')) for s in ['train', 'val']}
print('categories:', coco['train']['categories'])
print('image example:', coco['train']['images'][0])
print('ann example:', coco['train']['annotations'][0])
print('crowd category ids:', {a['category_id'] for a in coco['train']['annotations'] if a.get('iscrowd', 0)})
display(pd.read_csv(f'{ROOT}/test_images.csv').head(3))
print('sample submission:', SAMPLE_SUB)
print('test frames matched on disk:', len(test_paths()))

# %% Tables: one row per image, one row per annotation
cat2cls = {c['id']: CLASSES.index(c['name']) if c['name'] in CLASSES else c['id'] - 1
           for c in coco['train']['categories']}
def tables(split):
    ims = pd.DataFrame(coco[split]['images'])
    ims['clip'] = ims.file_name.map(lambda f: key(f).split('/')[0])
    ims['split'] = split
    an = pd.DataFrame(coco[split]['annotations'])
    an['iscrowd'] = an.get('iscrowd', 0)
    an[['x', 'y', 'w', 'h']] = pd.DataFrame(an.bbox.tolist(), index=an.index)
    an = an.merge(ims[['id', 'clip', 'width', 'height']].rename(columns={'id': 'image_id'}), on='image_id')
    an['split'] = split
    return ims, an
ims, anns = map(pd.concat, zip(*[tables(s) for s in ['train', 'val']]))
obj = anns[anns.iscrowd == 0].copy()
obj['cls'] = obj.category_id.map(cat2cls)
obj['size'] = np.sqrt(obj.w * obj.h)                         # px in the original frame
obj['size1280'] = obj['size'] * 1280 / obj[['width', 'height']].max(axis=1)   # after letterbox to 1280
ign = anns[anns.iscrowd == 1]

test_sizes = pd.Series([Image.open(p).size for p in test_paths()]).value_counts()   # header read only
summary = ims.groupby('split').agg(clips=('clip', 'nunique'), frames=('id', 'size'))
summary['objects'] = obj.groupby('split').size()
summary['ignored_regions'] = ign.groupby('split').size()
display(summary)

# %% Resolutions per split (train / val / test)
res = ims.assign(res=ims.width.astype(str) + 'x' + ims.height.astype(str)).groupby(['split', 'res'])['clip'].nunique()
display(pd.concat([res.unstack(0), test_sizes.rename(lambda s: f'{s[0]}x{s[1]}').rename('test_frames')], axis=1)
        .fillna(0).astype(int).sort_index())

# %% Class distribution (instances and unique tracks) train vs val
cnt = obj.groupby(['cls', 'split']).size().unstack()
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
(cnt / cnt.sum()).rename(index=dict(enumerate(CLASSES))).plot.bar(ax=ax[0], rot=0)
ax[0].set_title('Class share of instances'); ax[0].set_ylabel('fraction')
if 'track_id' in obj:
    tracks = obj.groupby(['split', 'clip', 'track_id']).cls.first().reset_index().groupby(['cls', 'split']).size().unstack()
    tracks.rename(index=dict(enumerate(CLASSES))).plot.bar(ax=ax[1], rot=0, logy=True)
    ax[1].set_title('Unique objects (tracks), log scale')
save('class_distribution')
display(cnt.rename(index=dict(enumerate(CLASSES))).assign(share_train=lambda d: (d.train / d.train.sum()).round(3)))

# %% Box sizes: sqrt(area) per class + COCO size buckets (small < 32^2, medium < 96^2) in original pixels
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
bins = np.logspace(0, np.log10(obj['size'].max()), 60)
for c in range(len(CLASSES)):
    ax[0].hist(obj.loc[obj.cls == c, 'size'], bins=bins, histtype='step', color=COLORS[c], label=CLASSES[c], density=True)
ax[0].set_xscale('log'); ax[0].axvline(32, ls='--', c='k'); ax[0].axvline(96, ls='--', c='k')
ax[0].set_xlabel('sqrt(w*h), px'); ax[0].legend(fontsize=8); ax[0].set_title('Box size per class')
ax[1].hexbin(obj.w, obj.h, gridsize=80, bins='log', xscale='log', yscale='log', cmap='viridis')
ax[1].set_xlabel('w, px'); ax[1].set_ylabel('h, px'); ax[1].set_title('Box w vs h (log counts)')
save('box_sizes')
bucket = pd.cut(obj['size'], [0, 32, 96, np.inf], labels=['small', 'medium', 'large'])
display(pd.crosstab(obj.cls.map(dict(enumerate(CLASSES))), bucket, normalize='index').round(3))

# %% Why imgsz=1280: how many objects fall below 8 / 16 px after letterboxing to 640 vs 1280
long_side = obj[['width', 'height']].max(axis=1)
display(pd.DataFrame({s: {f'<{t}px': round(float((obj['size'] * s / long_side < t).mean()), 3) for t in (8, 16, 32)}
                      for s in (640, 960, 1280, 1920)}).rename_axis('share of boxes'))

# %% Objects per frame + variation across clips
per_frame = obj.groupby(['split', 'image_id']).size().reindex(
    pd.MultiIndex.from_frame(ims[['split', 'id']]), fill_value=0)
clip_stats = obj.groupby(['split', 'clip']).agg(objs=('id', 'size'), median_size=('size', 'median'),
                                                width=('width', 'first'), height=('height', 'first'))
clip_stats['frames'] = ims.groupby(['split', 'clip']).size()
clip_stats['objs_per_frame'] = (clip_stats.objs / clip_stats.frames).round(1)
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
ax[0].hist(per_frame.loc['train'], bins=60, alpha=.6, density=True, label='train')
ax[0].hist(per_frame.loc['val'], bins=60, alpha=.6, density=True, label='val')
ax[0].set_xlabel('objects per frame'); ax[0].legend(); ax[0].set_title('Density')
for s, m in [('train', 'o'), ('val', '^')]:
    d = clip_stats.loc[s]
    ax[1].scatter(d.median_size, d.objs_per_frame, marker=m, label=s, alpha=.7)
ax[1].set_xscale('log'); ax[1].set_xlabel('median box size per clip, px'); ax[1].set_ylabel('objects / frame')
ax[1].set_title('Per-clip variation'); ax[1].legend()
save('objects_per_frame')
print('max objects in a frame:', per_frame.max(), '| frames > 500 objects:', int((per_frame > 500).sum()))

# %% Class composition per clip (heatmap): clips differ a lot -> clip-level split matters
comp = obj.groupby(['split', 'clip', 'cls']).size().unstack(fill_value=0)
comp = comp.div(comp.sum(1), axis=0).sort_values(2)
plt.figure(figsize=(8, 12))
plt.imshow(comp.values, aspect='auto', cmap='magma')
plt.xticks(range(len(CLASSES)), CLASSES, rotation=45)
plt.yticks(range(len(comp)), [f'{s}:{c}' for s, c in comp.index], fontsize=6)
plt.colorbar(label='class share in clip'); plt.title('Class composition per clip')
save('clip_composition')

# %% Ignored regions: how much of the frame they cover per clip (pixels, overlaps counted once)
def ignored_fraction(g):
    W, H = int(g.width.iloc[0]), int(g.height.iloc[0])
    m = np.zeros((H // 4 + 1, W // 4 + 1), bool)               # 4x downsampled mask is enough
    for x, y, w, h in g[['x', 'y', 'w', 'h']].values // 4:
        m[int(y):int(y + h) + 1, int(x):int(x + w) + 1] = True
    return m.mean()
ign_frac = ign.groupby(['split', 'clip', 'image_id'])[['x', 'y', 'w', 'h', 'width', 'height']].apply(ignored_fraction).groupby(['split', 'clip']).mean()
clip_stats['ignored_area'] = ign_frac.reindex(clip_stats.index).fillna(0).round(3)
clip_stats['frames_with_ignore'] = (ign.groupby(['split', 'clip']).image_id.nunique() / clip_stats.frames).fillna(0).round(2)
display(clip_stats.sort_values('objs_per_frame', ascending=False))

# %% Temporal redundancy -> justification of TRAIN_STRIDE=5: share of objects (track_id) still present after k frames
if 'track_id' in obj:
    tr = obj[obj.split == 'train'].merge(ims[['id', 'frame_id']].rename(columns={'id': 'image_id'}), on='image_id') \
        if 'frame_id' in ims else None
    if tr is not None:
        sets = tr.groupby(['clip', 'frame_id']).track_id.apply(set)
        rows = []
        for k in [1, 2, 5, 10, 20, 50]:
            shared = [len(a & sets.get((c, f + k), set())) / max(len(a), 1) for (c, f), a in sets.items()
                      if (c, f + k) in sets]
            rows.append({'k': k, 'shared_tracks': np.mean(shared)})
        display(pd.DataFrame(rows).round(3))

# %% GT visualisation: boxes per class + ignored regions (hatched grey). Densest / sparsest / smallest clips.
def show_gt(ax, img_row, crop=None):
    p = f'{ROOT}/{img_row.split}/images/{key(img_row.file_name)}'
    ax.imshow(Image.open(p)); ax.set_axis_off()
    a = anns[(anns.image_id == img_row.id) & (anns.split == img_row.split)]
    for _, r in a.iterrows():
        if r.iscrowd:
            ax.add_patch(patches.Rectangle((r.x, r.y), r.w, r.h, fc=(.5, .5, .5, .45), ec='w', hatch='//', lw=.5))
        else:
            ax.add_patch(patches.Rectangle((r.x, r.y), r.w, r.h, fill=False, lw=.8, ec=COLORS[cat2cls[r.category_id]]))
    if crop: ax.set_xlim(crop[0], crop[2]); ax.set_ylim(crop[3], crop[1])
    ax.set_title(f"{img_row['clip']} ({len(a)} boxes)", fontsize=8)

dense = clip_stats.objs_per_frame.idxmax(); sparse = clip_stats.objs_per_frame.idxmin()
small = clip_stats.median_size.idxmin(); with_ign = clip_stats.ignored_area.idxmax()
def mid_frame(s, c):                                         # ims['clip'], not ims.clip (= DataFrame.clip method)
    f = ims[(ims.split == s) & (ims['clip'] == c)]
    return f.iloc[len(f) // 2]
picks = [mid_frame(*sc) for sc in [dense, sparse, small, with_ign]]
fig, axs = plt.subplots(2, 2, figsize=(16, 10))
for ax, r in zip(axs.flat, picks): show_gt(ax, r)
fig.legend(handles=[patches.Patch(color=COLORS[i], label=c) for i, c in enumerate(CLASSES)] +
           [patches.Patch(fc='grey', hatch='//', label='ignored')], loc='lower center', ncol=9)
save('gt_examples')

# %% Zoom into the densest frame: how small and overlapping the objects really are
r = picks[0]
fig, ax = plt.subplots(figsize=(10, 6))
show_gt(ax, r, crop=(r.width * .35, r.height * .35, r.width * .65, r.height * .65))
save('gt_zoom')

# %% Preview of ignore masking (grey 114 like Ultralytics letterbox) — used in the masking experiment
r = picks[3]
img = np.array(Image.open(f'{ROOT}/{r.split}/images/{key(r.file_name)}'))
for x, y, w, h in ign.loc[(ign.image_id == r.id) & (ign.split == r.split), ['x', 'y', 'w', 'h']].values.astype(int):
    img[y:y + h, x:x + w] = 114
plt.figure(figsize=(10, 6)); plt.imshow(img); plt.axis('off'); plt.title(f"{r['clip']}: ignored regions masked")
save('ignore_masking')

# %% Anchor candidates for RetinaNet: k-means (IoU distance, YOLOv2-style) on train boxes at the 1280 scale
def kmeans_iou(wh, k=9, iters=100, seed=SEED):
    rng = np.random.default_rng(seed); c = wh[rng.choice(len(wh), k, replace=False)]
    for _ in range(iters):
        inter = np.minimum(wh[:, None, 0], c[None, :, 0]) * np.minimum(wh[:, None, 1], c[None, :, 1])
        iou = inter / (wh[:, None].prod(2) + c[None].prod(2) - inter)
        a = iou.argmax(1)
        new = np.array([np.median(wh[a == j], 0) if (a == j).any() else c[j] for j in range(k)])
        if np.allclose(new, c): break
        c = new
    return c[np.argsort(c.prod(1))], iou.max(1).mean()
tr = obj[obj.split == 'train']
s = 1280 / tr[['width', 'height']].max(axis=1).values
wh = np.stack([tr.w.values * s, tr.h.values * s], 1)
wh = wh[np.random.default_rng(SEED).choice(len(wh), min(len(wh), 200_000), replace=False)]
anchors, mean_iou = kmeans_iou(wh)
print('anchors (w,h @1280):', anchors.round(1).tolist(), '| mean best IoU:', round(float(mean_iou), 3))
json.dump({'anchors_wh_1280': anchors.round(2).tolist(), 'mean_iou': float(mean_iou)}, open(f'{WORK}/anchors.json', 'w'))
