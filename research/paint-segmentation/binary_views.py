"""Binary research outputs: native probability maps, ROI-only views and errors."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from overfit import Tile, tensors
from prepare import write_image, write_json
from train import checked_sample


def mark_ignore(image, target):
    """Explicit grey diagonal hatching; no quality claim on unreviewed pixels."""
    result = image.copy()
    ignored = target == 255
    result[ignored] = np.rint(result[ignored] * 0.25 + 60).astype(np.uint8)
    yy, xx = np.indices(target.shape)
    result[ignored & ((xx + yy) % 12 < 2)] = (135, 135, 135)
    return result


def label_view(image, prediction, target):
    result = image.copy()
    valid = target != 255
    colors = np.array([[255, 40, 20], [255, 0, 255]], np.uint8)
    result[valid] = np.rint(
        0.45 * image[valid] + 0.55 * colors[prediction[valid]]
    ).astype(np.uint8)
    return mark_ignore(result, target)


def error_view(image, prediction, target):
    result = image.copy()
    result[(target == 0) & (prediction == 1)] = (0, 0, 255)  # FP red
    result[(target == 1) & (prediction == 0)] = (0, 255, 255)  # FN yellow
    return mark_ignore(result, target)


def probability_view(probability, target):
    values = np.rint(probability * 255).astype(np.uint8)
    return mark_ignore(cv2.applyColorMap(values, cv2.COLORMAP_VIRIDIS), target)


def card(views, labels, title):
    strip = np.hstack(views)
    header = np.full((56, strip.shape[1], 3), 24, np.uint8)
    cv2.putText(
        header, title, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1
    )
    for index, label in enumerate(labels):
        cv2.putText(
            header,
            label,
            (index * views[0].shape[1] + 8, 43),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
        )
    return np.vstack([header, strip])


def save_predictions(
    output, root, tiles, samples, prediction, probability, model, device, batch_size
):
    from binary_paint import binary_mask

    output.mkdir()
    (output / "tiles").mkdir()
    model.eval()
    for tile, pred_tensor, prob_tensor in zip(tiles, prediction, probability):
        pred, prob = pred_tensor.numpy().astype(np.uint8), prob_tensor.numpy()
        prefix = output / "tiles" / tile.name
        write_image(Path(f"{prefix}.prediction.png"), pred)
        write_image(Path(f"{prefix}.target.png"), tile.mask)
        np.save(Path(f"{prefix}.paint-probability.npy"), prob)
        write_image(
            Path(f"{prefix}.errors.png"), error_view(tile.image, pred, tile.mask)
        )
        write_image(
            Path(f"{prefix}.probability.png"), probability_view(prob, tile.mask)
        )
        write_image(
            Path(f"{prefix}.comparison.jpg"),
            card(
                [
                    tile.image,
                    label_view(tile.image, tile.mask, tile.mask),
                    label_view(tile.image, pred, tile.mask),
                ],
                ["Original", "Binary GT", "Binary prediction"],
                f"{tile.name}: hatched=ignore, blue=non-paint, pink=paint",
            ),
        )
    for sample in samples:
        image, original = checked_sample(sample, root)
        target = binary_mask(original)
        h, w = target.shape
        raw = np.full((h, w), 255, np.uint8)
        full_probability = np.full((h, w), np.nan, np.float32)
        selected = [(i, t) for i, t in enumerate(tiles) if t.sample_id == sample["id"]]
        size = tiles[0].mask.shape[0]

        def paste(tile, pred, prob):
            ih, iw = target[tile.y : tile.y + size, tile.x : tile.x + size].shape
            raw[tile.y : tile.y + ih, tile.x : tile.x + iw] = pred[:ih, :iw]
            full_probability[tile.y : tile.y + ih, tile.x : tile.x + iw] = prob[
                :ih, :iw
            ]

        occupied = set()
        for i, tile in selected:
            paste(tile, prediction[i].numpy(), probability[i].numpy())
            occupied.add((tile.x, tile.y))
        extra = []
        for y in range(0, h, size):
            for x in range(0, w, size):
                if (x, y) in occupied:
                    continue
                if np.any(target[y : y + size, x : x + size] != 255):
                    raise ValueError("A scored ROI is missing from validation tiles")
                patch = np.zeros((size, size, 3), np.uint8)
                crop = image[y : y + size, x : x + size]
                patch[: crop.shape[0], : crop.shape[1]] = crop
                extra.append(
                    Tile(
                        sample["id"], x, y, patch, np.full((size, size), 255, np.uint8)
                    )
                )
        # Extra tiles have no labels. They complete the native P(paint) map only,
        # never changing metrics, selection or the already-scored predictions.
        for start in range(0, len(extra), batch_size):
            batch = extra[start : start + batch_size]
            images, _ = tensors(batch)
            with torch.inference_mode():
                logits = model(images.to(device)).cpu()
                preds, probs = logits.argmax(1), logits.softmax(1)[:, 1]
            for tile, pred, prob in zip(batch, preds, probs):
                paste(tile, pred.numpy(), prob.numpy())
        if not np.isfinite(full_probability).all() or not set(np.unique(raw)) <= {0, 1}:
            raise ValueError("Native prediction map is incomplete")
        prefix = output / sample["id"]
        write_image(Path(f"{prefix}.prediction.png"), raw)
        write_image(
            Path(f"{prefix}.roi-prediction.png"),
            np.where(target != 255, raw, 255).astype(np.uint8),
        )
        write_image(Path(f"{prefix}.binary-target.png"), target)
        np.save(Path(f"{prefix}.paint-probability.npy"), full_probability)
        np.save(
            Path(f"{prefix}.roi-paint-probability.npy"),
            np.where(target != 255, full_probability, np.nan),
        )
        write_image(
            Path(f"{prefix}.probability.png"),
            probability_view(full_probability, target),
        )
        write_image(Path(f"{prefix}.errors.png"), error_view(image, raw, target))
    write_json(
        output / "README.json",
        {
            "raw_probability": (
                "native float32 HxW P(paint)=softmax(logits)[1], no resizing; "
                "raw unreviewed pixels are experimental outputs, "
                "not validated predictions"
            ),
            "roi_probability": "float32 HxW; NaN outside reviewed ROI",
            "prediction": "argmax of two logits; no threshold search",
            "views": "all label/probability/error views explicitly hatch ignore grey",
            "colors": {
                "non-paint": "blue",
                "paint": "pink",
                "FP": "red",
                "FN": "yellow",
            },
            "probability_colormap": (
                "viridis fixed 0..1, purple low / yellow high; ignore hatched"
            ),
            "full_image_completion": (
                "additional unreviewed grid tiles inferred after checkpoint "
                "selection only; evaluated ROI predictions remain exactly "
                "those used for metrics"
            ),
        },
    )


def plot_history(output, history):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(
        [h["epoch"] for h in history[1:]],
        [h["train_batch_loss"] for h in history[1:]],
        label="UFPR augmented",
    )
    axes[0].plot(
        [h["epoch"] for h in history],
        [h["validation"]["loss"] for h in history],
        label="CNR ROI",
    )
    axes[0].set(xlabel="Epoch", ylabel="Binary CE + paint Dice")
    axes[1].plot(
        [h["epoch"] for h in history],
        [h["validation"]["paint"]["iou"] for h in history],
        label="CNR paint IoU",
    )
    axes[1].set(xlabel="Epoch", ylabel="Paint IoU", ylim=(0, 1))
    for axis in axes:
        axis.legend()
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output / "learning-curves.png", dpi=150)
    plt.close(figure)
