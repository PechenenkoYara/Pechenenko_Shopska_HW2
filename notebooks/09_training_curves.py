# %% [markdown]
# 09 — Training curves for the report: val mAP per epoch, RT-DETR variants vs the CNN baseline (YOLO11s).
# CPU, no Kaggle data: the numbers are copied from the results.csv files of the runs (sources next to each list).
# Metric: Ultralytics built-in val mAP50-95, for curves only; report numbers come from common.coco_eval.
# YOLO11s: full val/; RT-DETR: every 10th val frame (TRAIN_VAL_STRIDE in 05_rtdetr.py).

# %%
import os
import matplotlib.pyplot as plt

OUT = '../images' if os.path.isdir('../images') else '/kaggle/working'

# 03_yolo_baseline.py, runs/yolo11s_baseline/results.csv (same list as YOLO_BASELINE_MAP in 05_rtdetr.py)
YOLO11S = [0.19156, 0.19051, 0.19172, 0.19309, 0.1863, 0.18018, 0.19372, 0.20222, 0.19738, 0.19009,
           0.1865, 0.1835, 0.18567, 0.19447, 0.18527, 0.18458, 0.18411, 0.17988, 0.1808, 0.17928]
# 05_rtdetr.py EXP='baseline' (Ultralytics warmup_bias_lr=0.1), runs/rtdetr_l_baseline/results.csv, all 20 epochs
BASELINE = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.06406, 0.08647, 0.08956, 0.1083, 0.10665, 0.10952, 0.1103, 0.11383]
# 05_rtdetr.py EXP='fixwarmup', runs/rtdetr_l_fixwarmup/results.csv (stopped at 16 by patience 6, best epoch 10)
FIXWARMUP = [0.16688, 0.25196, 0.2751, 0.2902, 0.28803, 0.29136, 0.28604, 0.28928, 0.28881, 0.29347,
             0.27592, 0.28063, 0.26915, 0.27914, 0.28332, 0.28817]
# 05_rtdetr.py EXP='fixwarmup_nodn', runs/rtdetr_l_fixwarmup_nodn/results.csv (stopped at 16, best epoch 10)
NODN = [0.14444, 0.21041, 0.25451, 0.28189, 0.28173, 0.28953, 0.29177, 0.28859, 0.29256, 0.29513,
        0.29293, 0.28777, 0.28694, 0.28149, 0.28724, 0.27798]
# 05_rtdetr.py EXP='giou4', runs/rtdetr_l_giou4/results.csv (Colab, stopped after 8/20 epochs by the Colab GPU limit)
GIOU4 = [0.15667, 0.22709, 0.24583, 0.27984, 0.28144, 0.28319, 0.28711, 0.28716]

# Fixed categorical order (blue, orange, aqua, yellow); marker shape + direct label as secondary encoding
SERIES = [('YOLO11s baseline (CNN)', YOLO11S, '#2a78d6', 'o'),
          ('RT-DETR-l fixwarmup', FIXWARMUP, '#eb6834', 's'),
          ('RT-DETR-l no denoising', NODN, '#1baf7a', '^'),
          ('RT-DETR-l GIoU x2', GIOU4, '#eda100', 'D'),
          ('RT-DETR-l baseline (warmup_bias_lr=0.1)', BASELINE, '#e87ba4', 'v')]

# %% Plot
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': '#52514e', 'axes.labelcolor': '#0b0b0b',
                     'xtick.color': '#52514e', 'ytick.color': '#52514e'})
fig, ax = plt.subplots(figsize=(8, 4.6), facecolor='#fcfcfb')
ax.set_facecolor('#fcfcfb')
for name, ys, color, marker in SERIES:
    ep = range(1, len(ys) + 1)
    ax.plot(ep, ys, color=color, lw=2, marker=marker, ms=5, label=name)
ax.axvline(10, color='#a3a29d', lw=0.8, ls=':')     # best.pt of fixwarmup and no denoising; both stopped at 16 (patience 6)
ax.text(10.15, 0.313, 'best.pt (both full RT-DETR runs)', fontsize=8.5, color='#52514e')
ax.set_xlabel('epoch')
ax.set_ylabel('val mAP50-95 (Ultralytics)')
ax.set_xticks(range(1, 21))
ax.set_xlim(0.5, 20.5); ax.set_ylim(-0.005, 0.32)
ax.grid(axis='y', color='#e4e3df', lw=0.8); ax.set_axisbelow(True)
for side in ['top', 'right']:
    ax.spines[side].set_visible(False)
ax.legend(frameon=False, loc='lower left', bbox_to_anchor=(0.12, 0.05))   # empty area: epochs 2-11, mAP 0.02-0.13
ax.set_title('Val mAP per epoch: query-based RT-DETR vs CNN YOLO11s, same budget (20 epochs, imgsz 1280)',
             fontsize=10, loc='left', color='#0b0b0b')
fig.tight_layout()
fig.savefig(f'{OUT}/curve_rtdetr_vs_yolo.png', dpi=150, facecolor=fig.get_facecolor())
plt.show()


# %% Warmup diagnosis (two measures, two panels, no shared y-axis): bias LR vs weight LR, and the classification loss.
# results.csv columns: lr/pg0 = weights with decay, lr/pg2 = biases; train/cls_loss.
LR_W_BASE = [3.33057e-05, 6.62329e-05, 9.75503e-05, 9.46048e-05, 9.05463e-05, 8.55018e-05]   # baseline lr/pg0, epochs 1-6
LR_B_BASE = [0.0667276, 0.0334272, 0.00012519, 9.46048e-05, 9.05463e-05, 8.55018e-05]       # baseline lr/pg2
CLS_BASE = [0.00199, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 0, 0, 0, 0, 0, 0.00012, 0.71775, 0.69238, 0.63249,
            0.67556, 0.66343, 0.65203, 0.64126, 0.63607]
CLS_FIX = [0.40534, 0.42362, 0.4528, 0.45238, 0.44928, 0.4477, 0.44487, 0.44295, 0.44136, 0.43573,
           0.43416, 0.43211, 0.43172, 0.42931, 0.4266, 0.43176]   # fixwarmup: lr/pg2 == lr/pg0 every epoch

fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.6), facecolor='#fcfcfb')
ep6 = range(1, 7)
a1.plot(ep6, LR_B_BASE, color='#e87ba4', lw=2, marker='v', ms=5, label='baseline: biases (lr/pg2)')
a1.plot(ep6, LR_W_BASE, color='#2a78d6', lw=2, marker='o', ms=5, label='baseline: weights (lr/pg0) = fixwarmup: all groups')
a1.set_yscale('log'); a1.set_xlabel('epoch'); a1.set_ylabel('learning rate (log)')
a1.set_title('Learning rate per parameter group', fontsize=10, loc='left')
a2.plot(range(1, 21), CLS_BASE, color='#e87ba4', lw=2, marker='v', ms=5, label='baseline (warmup_bias_lr=0.1)')
a2.plot(range(1, 17), CLS_FIX, color='#eb6834', lw=2, marker='s', ms=5, label='fixwarmup (warmup_bias_lr=0)')
a2.set_xlabel('epoch'); a2.set_ylabel('train cls_loss'); a2.set_xticks(range(1, 21, 2))
a2.set_title('Classification loss', fontsize=10, loc='left')
for a in (a1, a2):
    a.set_facecolor('#fcfcfb'); a.grid(axis='y', color='#e4e3df', lw=0.8); a.set_axisbelow(True)
    a.spines[['top', 'right']].set_visible(False); a.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(f'{OUT}/rtdetr_warmup_diagnosis.png', dpi=150, facecolor=fig.get_facecolor())
plt.show()
