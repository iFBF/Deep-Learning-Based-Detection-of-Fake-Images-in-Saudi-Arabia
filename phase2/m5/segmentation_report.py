"""Report figures with split, input resolution and threshold context on every page."""
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from cohort_data import safe_path
from segmentation_metrics import from_counts


def make_figures(work, rows, comparisons, grid, selection, per_image):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    work = Path(work)
    out = work / "figures"
    out.mkdir(parents=True, exist_ok=True)
    context = "Validation • letterbox 640×640 • native-pixel scoring • no postprocessing"
    selected_threshold = selection["probability_threshold"]
    with (work / "metrics" / "val_before_finetuning_comparison.csv").open() as stream:
        before = {r["architecture"]: float(r["foreground_iou"]) for r in csv.DictReader(stream)}
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(comparisons))
    ax.bar(x - 0.27, [before[r["architecture"]] for r in comparisons], 0.18, label="Before fine-tuning: IoU at 0.50", color="#a8a8a8")
    ax.bar(x - 0.09, [r["fixed_0.5_foreground_iou"] for r in comparisons], 0.18, label="Fine-tuned: IoU at 0.50")
    ax.bar(x + 0.09, [r["foreground_iou"] for r in comparisons], 0.18, label="Fine-tuned: IoU at tuned threshold")
    ax.bar(x + 0.27, [r["foreground_dice"] for r in comparisons], 0.18, label="Fine-tuned: Dice at tuned threshold")
    ax.set_xticks(x, [r["architecture"] + f'\nt ≥ {r["threshold"]:.2f}' for r in comparisons], fontsize=9)
    ax.set(ylim=(0, 1), ylabel="Global foreground score")
    ax.set_title(context + "\nBefore/fixed: t = 0.50; tuned thresholds shown under model names", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "val_segmenter_comparison.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for name in [r["architecture"] for r in comparisons]:
        values = [r for r in grid if r["architecture"] == name]
        ax.plot([r["threshold"] for r in values], [r["foreground_iou"] for r in values], marker=".", label=name)
    metric = selection["validation_metrics"]
    ax.scatter([selected_threshold], [metric["foreground_iou"]], marker="*", s=140, color="black", zorder=5, label="Selected")
    ax.set(xlabel="Probability threshold (0.05…0.95, step 0.05)", ylabel="Global foreground IoU", ylim=(0, 1))
    ax.set_title(context + "\nValidation threshold sweep; foreground if probability ≥ threshold", fontsize=10)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "val_segmentation_threshold_sweep.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for row in comparisons:
        with (work / "runs" / row["architecture"] / "results.csv").open() as stream:
            history = list(csv.DictReader(stream))
        epochs = [int(r["epoch"]) for r in history]
        short = row["architecture"].split("_")[1]
        axes[0].plot(epochs, [float(r["train_loss"]) for r in history], label=short + " train")
        axes[0].plot(epochs, [float(r["val_loss"]) for r in history], linestyle="--", label=short + " val")
        axes[1].plot(epochs, [float(r["val_foreground_iou_at_0.5"]) for r in history], label=short)
    axes[0].set(xlabel="Epoch", ylabel="Masked BCE + positive-mask soft Dice loss")
    axes[1].set(xlabel="Epoch", ylabel="Validation foreground IoU at t = 0.50", ylim=(0, 1))
    for ax in axes:
        ax.legend(fontsize=8)
    fig.suptitle("Train/validation • letterbox 640×640\nLoss excludes padding; IoU uses native masks at threshold 0.50; no postprocessing", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "train_val_segmentation_learning_curves.png", dpi=160)
    plt.close(fig)

    matrix = np.array([[metric["tn"], metric["fp"]], [metric["fn"], metric["tp"]]], dtype=np.int64)
    normalized = matrix / np.maximum(matrix.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{matrix[i,j]:,}\n{normalized[i,j]:.1%}", ha="center", va="center",
                    color="white" if normalized[i,j] > 0.5 else "black", fontsize=10)
    ax.set_xticks([0, 1], ["Background", "Edited region"])
    ax.set_yticks([0, 1], ["Background", "Edited region"])
    ax.set(xlabel="Predicted pixel class", ylabel="True pixel class")
    ax.set_title(context + f"\nThreshold ≥ {selected_threshold:.2f}; counts and percentages within true-class rows", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "val_pixel_confusion_matrix.png", dpi=160)
    plt.close(fig)

    # Six deterministic development examples; the final test failure gallery is still reserved.
    positives = [r for r in per_image if r["kind"] == "edited"]
    negatives = [r for r in per_image if r["kind"] == "pristine"]
    dice = lambda r: from_counts(**{k: int(r[k]) for k in ["tp", "fp", "fn", "tn"]})["foreground_dice"]
    candidates = [r["uid"] for r in sorted(positives, key=lambda r: (dice(r), r["uid"]))[:2]]
    candidates += [r["uid"] for r in sorted(positives, key=lambda r: (-dice(r), r["uid"]))[:2]]
    candidates += [r["uid"] for r in sorted(negatives, key=lambda r: (-r["fp"], r["uid"]))[:1]]
    candidates += [r["uid"] for r in sorted(negatives, key=lambda r: (r["fp"], r["uid"]))[:1]]
    uids = list(dict.fromkeys(candidates))
    uids += [r["uid"] for r in per_image if r["uid"] not in uids]
    uids = uids[:6]
    by_uid = {r["uid"]: r for r in rows}
    fig, axes = plt.subplots(len(uids), 4, figsize=(14, 3 * len(uids)), squeeze=False)
    for line, uid in enumerate(uids):
        row = by_uid[uid]
        with Image.open(safe_path(work / "data_view", row["export_image"])) as opened:
            rgb = np.asarray(opened.convert("RGB"))
        with Image.open(safe_path(work / "data_view", row["export_mask"])) as opened:
            truth = np.asarray(opened, dtype=np.uint8)
        with Image.open(work / "val_prediction_masks" / (uid + ".png")) as opened:
            predicted = np.asarray(opened, dtype=np.uint8)
        overlay = rgb.astype(np.float32) / 255
        overlay[predicted > 0] = 0.55 * overlay[predicted > 0] + 0.45 * np.array([1.0, 0.15, 0.1])
        for ax, value, title in zip(axes[line], [rgb, truth, predicted, overlay],
                                    [uid, "Native ground truth", "Predicted 0/1 mask", "Prediction overlay"], strict=True):
            ax.imshow(value, cmap="gray" if value.ndim == 2 else None, vmin=0 if value.ndim == 2 else None,
                      vmax=1 if value.ndim == 2 else None)
            ax.set_title(title, fontsize=8)
            ax.axis("off")
    fig.suptitle(context + f"\n{selection['selected_architecture']} • probability ≥ {selected_threshold:.2f}\n"
                 "Validation development examples; white/red = edited region", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "val_segmentation_development_gallery.png", dpi=120)
    plt.close(fig)
