"""Reports and deterministic failure figures from cached predictions; never calls a model."""
from pathlib import Path
import json

import numpy as np
from PIL import Image

from final_assets import dump


def pct(value):
    return "N/A" if value is None else f"{100 * value:.2f}%"


def build_reports(root, rows, entries, results, work, release):
    from cohort_data import safe_path, write_csv
    from segmentation_metrics import from_counts
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    work, release = Path(work), Path(release)
    reports, figures = work / "reports", work / "figures"
    reports.mkdir(parents=True, exist_ok=True);figures.mkdir(parents=True, exist_ok=True)
    indexed = {e["metadata"]["uid"]: e for e in entries}
    error_index = {r["uid"]: r for r in results["detector_per_image_errors"]}
    per_image = []
    for row in rows:
        m = indexed[row["uid"]]["metadata"];r = m["result"]
        segment = from_counts(**m["segmentation_counts"])
        per_image.append({"uid": row["uid"], "source_id": row["source_id"], "kind": row["kind"],
            "manipulation": row.get("manipulation", ""), "split": "test", "width": m["width"], "height": m["height"],
            "global_fake_score": r["classifier"]["fake_score"], "global_label": r["classifier"]["label"],
            "review_recommended": r["decision"]["review_recommended"], "edited_area_fraction": r["mask"]["edited_area_fraction"],
            "foreground_iou": segment["foreground_iou"], "foreground_dice": segment["foreground_dice"],
            "mask_area_at_input640": m["mask_area_at_input"],
            **{"pixel_" + k: v for k, v in m["segmentation_counts"].items()},
            **{"detection_" + k: error_index[row["uid"]][k] for k in ["tp", "fp", "fn", "duplicate", "localisation", "background"]}})
    write_csv(reports / "test_per_image_metrics.csv", per_image)
    with (reports / "test_prediction_records.jsonl").open("w") as stream:
        for entry in entries:
            record = {"metadata": entry["metadata"], **{key: entry[key].tolist() for key in
                ["ap_input_predictions", "ap_native_predictions", "ap_input_correct", "ap_native_correct"]}}
            stream.write(json.dumps(record, allow_nan=False) + "\n")
    detector = results["detector_operating_metrics"]
    errors = []
    for name, key in [("duplicate", "duplicate_fp"), ("localisation", "localisation_fp"), ("background", "background_fp")]:
        count = detector[key]
        errors.append({"type": name, "count": count, "share_of_FP": count / detector["fp"] if detector["fp"] else 0.0})
    errors.append({"type": "classification_not_applicable_single_class", "count": 0, "share_of_FP": None})
    errors.append({"type": "missed_FN", "count": detector["fn"], "share_of_FP": None})
    write_csv(reports / "test_detector_error_taxonomy.csv", errors)
    write_csv(reports / "test_detector_size_recall.csv", [{"size": name, **value} for name, value in detector["recall_by_size"].items()])
    ap = results["detector_AP_primary_input_coordinates"]
    write_csv(reports / "test_detector_AP_by_iou.csv", [{"matching_iou": t, "AP": v} for t, v in zip(ap["matching_ious"], ap["AP_by_matching_iou"], strict=True)])
    segment = results["segmentation"]
    write_csv(reports / "test_segmentation_per_class.csv", [
        {"class": "background", "IoU": segment["background_iou"], "Dice": segment["background_dice"]},
        {"class": "edited_region", "IoU": segment["foreground_iou"], "Dice": segment["foreground_dice"]}])
    write_csv(reports / "test_segmentation_size_slices.csv", [{"size": name, **value} for name, value in results["segmentation_by_mask_size_at640"].items()],
              fields=["size"] + sorted(set().union(*(v.keys() for v in results["segmentation_by_mask_size_at640"].values()))))

    title_scope = "SYNTHETIC LAYOUT CHECK (not project test results)" if results.get("synthetic_layout_check") else "Test114"
    condition = title_scope + " / C224; D,S640 / FP32 batch1\nC0.5; D conf>0.35 NMS0.45 matchIoU0.50; S pixel>=0.5; review area>=0.005"
    captions = {}
    referral = results["image_review_referral"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    for ax, matrix, labels, title in [
        (axes[0], [[referral["tn"], referral["fp"]], [referral["fn"], referral["tp"]]], ["pristine", "edited"], "Image review: no flag / review"),
        (axes[1], [[segment["tn"], segment["fp"]], [segment["fn"], segment["tp"]]], ["background", "edited"], "Native pixel classification")]:
        ax.imshow(matrix, cmap="Blues")
        for y in range(2):
            for x in range(2):ax.text(x, y, f"{matrix[y][x]:,}", ha="center", va="center",
                                      color="white" if matrix[y][x] > np.max(matrix) / 2 else "black", fontsize=11)
        ax.set_xticks([0, 1], labels);ax.set_yticks([0, 1], labels)
        ax.set_xlabel("Predicted");ax.set_ylabel("Ground truth");ax.set_title(title, fontsize=10)
    fig.suptitle(condition, fontsize=10);fig.tight_layout(rect=(0, 0, 1, 0.86))
    name = "test_image_and_pixel_confusion.png";fig.savefig(figures / name, dpi=160);plt.close(fig)
    captions[name] = condition + ". Image labels use the fixed referral rule; pixel labels use the untouched native mask target."

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    names = ["small", "medium", "large"]
    dr = detector["recall_by_size"]
    sr = results["segmentation_by_mask_size_at640"]
    axes[0].bar(names, [dr[n]["recall"] or 0 for n in names], color="#287a61")
    axes[1].bar(names, [sr[n].get("foreground_dice") or 0 for n in names], color="#ba933a")
    for axis, source, quantity, nkey in [(axes[0], dr, "recall", "instances"), (axes[1], sr, "foreground_dice", "images")]:
        for i, n in enumerate(names):
            value = source[n].get(quantity)
            axis.text(i, (value or 0) + 0.02, f"{pct(value)}\nn={source[n].get(nkey, 0)}", ha="center", fontsize=9)
        axis.set_ylim(0, 1.2)
    axes[0].set_title("Detection recall / union-box area bins", fontsize=10)
    axes[1].set_title("Pooled foreground Dice / mask-area bins", fontsize=10)
    fig.suptitle(condition, fontsize=10);fig.tight_layout(rect=(0, 0, 1, 0.86))
    name = "test_size_recall_and_mask_dice.png";fig.savefig(figures / name, dpi=160);plt.close(fig)
    captions[name] = condition + ". Bins at640: small<32², medium32²–96², large>=96². Detector uses union-box area; segmenter uses GT foreground-mask area. Empty bins are N/A."

    row_index = {r["uid"]: r for r in rows}
    def panel(ax, uid, kind):
        row, entry = row_index[uid], indexed[uid]
        with Image.open(safe_path(root, row["export_image"])) as opened:image = np.asarray(opened.convert("RGB")).copy()
        with Image.open(safe_path(root, row["export_mask"])) as opened:truth = np.asarray(opened, np.uint8)
        pred = entry["mask"].astype(bool)
        image[pred] = (image[pred] * 0.6 + np.array([240, 55, 55]) * 0.4).astype(np.uint8)
        ax.imshow(image)
        if truth.any() and not truth.all():ax.contour(truth, levels=[0.5], colors=["cyan"], linewidths=0.8)
        elif truth.all():ax.add_patch(Rectangle((0, 0), image.shape[1]-1, image.shape[0]-1, fill=False, edgecolor="cyan", linewidth=1))
        box = entry["metadata"]["ground_truth_box_xyxy"]
        if box is not None:
            x0, y0, x1, y1 = box;ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False, edgecolor="cyan", linewidth=1))
        for det in entry["metadata"]["result"]["detections"]:
            x0, y0, x1, y1 = det["box"];ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False, edgecolor="yellow", linewidth=1))
        item = next(r for r in per_image if r["uid"] == uid)
        ax.set_title(uid + "\n" + kind + f" / Dice {pct(item['foreground_dice'])}", fontsize=8)
        ax.axis("off")

    positives = sorted((r for r in per_image if r["kind"] == "edited"), key=lambda r: (r["foreground_dice"], r["uid"]))
    negative_false = sorted((r for r in per_image if r["kind"] == "pristine" and r["pixel_fp"] > 0),
                            key=lambda r: (-r["edited_area_fraction"], r["uid"]))
    gallery = [(r["uid"], "lowest edited-image Dice") for r in positives[:4]] + [(r["uid"], "pristine false mask") for r in negative_false[:2]]
    if gallery:
        fig, axes = plt.subplots(2, 3, figsize=(12, 7))
        for ax, item in zip(axes.flat, gallery):panel(ax, *item)
        for ax in list(axes.flat)[len(gallery):]:ax.axis("off")
        fig.suptitle(condition + "\nCyan: GT mask/union box; red: predicted mask; yellow: selected detector boxes", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.87))
        name = "test_segmentation_failure_gallery.png";fig.savefig(figures / name, dpi=160);plt.close(fig)
        captions[name] = condition + ". Deterministic selection: four lowest edited-image Dice, then up to two pristine images with largest false-mask area. No post-test tuning."

    det_fail = sorted(per_image, key=lambda r: (-r["detection_fn"], -r["detection_fp"], r["mask_area_at_input640"], r["uid"]))
    chosen = [r for r in det_fail if r["detection_fn"] or r["detection_fp"]][:6]
    if chosen:
        fig, axes = plt.subplots(2, 3, figsize=(12, 7))
        for ax, r in zip(axes.flat, chosen):panel(ax, r["uid"], f"detector FP{r['detection_fp']} FN{r['detection_fn']}")
        for ax in list(axes.flat)[len(chosen):]:ax.axis("off")
        fig.suptitle(condition + "\nCyan: GT mask/union box; red: predicted mask; yellow: selected detector boxes", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.87))
        name = "test_detector_failure_gallery.png";fig.savefig(figures / name, dpi=160);plt.close(fig)
        captions[name] = condition + ". Deterministic FN-first, then FP-first ordering with smaller GT mask-area ties."
    dump(reports / "figure_captions.json", captions)

    verdict = {"preliminary_decision": "NO-GO for automatic authenticity decisions; human-review research demonstration only",
               "original_Phase1_criteria_unchanged": True,
               "Phase1_accuracy": {"threshold": ">=90%", "reported_selected_primary_test": "68.60%", "passed": False},
               "Phase1_macro_F1": {"threshold": ">=90%", "reported_selected_primary_test": "65.72%", "passed": False},
               "Phase1_condition": "Corrected selected full Swin; closed source-disjoint Phase1 test328; not re-evaluated by this notebook",
               "Phase1_publication_commit": "26a12b0a69641990be2ef4b2794a19e420f87e20",
               "new_Phase2_test_referral_metrics_are_a_different_target": results["image_review_referral"],
               "final_presentation_verdict_and_scope": "Team must justify the final use decision with these results, model card and failure gallery"}
    dump(reports / "acceptance_verdict.json", verdict)
    summary = (f"# Phase2 fixed test results\n\nRelease `{results['release_id']}`; protocol `{results['protocol_sha256']}`.\n\n"
        f"Test114 images/19 source families; classifier224×224, detector/segmenter640×640; FP32,batch1. "
        "Classifier0.5; detector confidence>0.35,NMS0.45,max50,operating matchIoU0.50; segmenter pixel>=0.5,no morphology; review area>=0.005.\n\n"
        "| Measure | Result |\n|---|---:|\n"
        f"|Detector AP50 / AP50–95, input coordinates, confidence floor0.001|{pct(ap['AP50'])} / {pct(ap['AP50_95'])}|\n"
        f"|Detector P / R / F1 at frozen operating point|{pct(detector['precision'])} / {pct(detector['recall'])} / {pct(detector['f1'])}|\n"
        f"|Foreground IoU / Dice|{pct(segment['foreground_iou'])} / {pct(segment['foreground_dice'])}|\n"
        f"|Two-class mIoU / mean Dice|{pct(segment['mean_iou_2_classes'])} / {pct(segment['mean_dice_2_classes'])}|\n"
        f"|Mean Dice over edited images|{pct(segment['positive_image_mean_dice'])}|\n"
        f"|Mean absolute native area-fraction error|{segment['mean_absolute_area_fraction_error']:.6f}|\n"
        f"|Image referral accuracy / macro-F1|{pct(referral['accuracy'])} / {pct(referral['macro_f1'])}|\n"
        f"|Image referral sensitivity / specificity|{pct(referral['sensitivity'])} / {pct(referral['specificity'])}|\n\n"
        "Primary AP is scored before native clipping. Secondary served-native AP is retained in JSON for transparency; "
        "the earlier M4 validator/training comparison remains separately labelled. No test confidence threshold was chosen. "
        "All figures and confidence intervals derive from cached predictions; plotting does not repeat model inference.\n\n"
        "Original Phase1 accuracy/macro-F1 criteria were not met on its closed test. These Phase2 targets do not replace them. "
        "Preliminary decision: NO-GO for automatic authenticity decisions; human-review prototype only. "
        "No Saudi operational accuracy or natural-video accuracy claim is supported. The teammate independent Phase1 rerun remains pending.\n")
    (reports / "TEST_RESULTS.md").write_text(summary)
    original_card = (release / "MODEL_CARD.md").read_text()
    body = original_card[original_card.index("## Intended use"):]
    tail = body.find("Final test metrics, test failure gallery")
    if tail >= 0:body = body[:tail]
    card = "# Real-Abshar — final fixed-evaluation model card\n\nThe reserved Phase2 test evaluation is complete under the frozen service/protocol.\n\n" + body + "\n## Final Phase2 test addendum\n\n" + summary.replace("# Phase2 fixed test results", "### Cached test measurements", 1)
    (reports / "FINAL_MODEL_CARD.md").write_text(card)
