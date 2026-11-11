"""Opt-in checkpoint readback on one manifested ROI; no baseline integration."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch

from .data import RECIPE, file_hash, local_path, verify
from .model import ARCHITECTURE, build_model, decide, tensor_input


def predict(
    checkpoint_path: Path,
    expected_sha256: str,
    data_root: Path,
    sample_id: str,
    *,
    legal_slot=False,
    view_confirmed=False,
    visibility="unassessed",
    device="cpu"
) -> dict:
    if file_hash(checkpoint_path) != expected_sha256:
        raise ValueError("Checkpoint hash mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    manifest_path = data_root / "manifest.json"
    if (
        checkpoint["architecture"] != ARCHITECTURE
        or checkpoint["recipe"] != RECIPE
        or checkpoint["classes"] != ["free", "occupied"]
        or checkpoint["manifest_sha256"] != file_hash(manifest_path)
    ):
        raise ValueError("Checkpoint/data contract mismatch")
    manifest = json.loads(manifest_path.read_text())
    verify(data_root, manifest)
    sample = next(s for s in manifest["samples"] if s["id"] == sample_id)
    model = build_model()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval().to(device)
    array = np.load(local_path(data_root, sample["array"]), allow_pickle=False)
    with torch.inference_mode():
        score = model(tensor_input(array)[None].to(device)).softmax(1)[0, 1].item()
    decision = decide(
        sample["slot_projection_id"],
        score,
        legal_slot=legal_slot,
        view_confirmed=view_confirmed,
        visibility=visibility,
    )
    return dict(
        asdict(decision),
        checkpoint_sha256=expected_sha256,
        sample_id=sample_id,
        split=sample["split"],
        calibration="none",
        architecture=ARCHITECTURE,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--confirm-legal-slot", action="store_true")
    parser.add_argument("--confirm-view", action="store_true")
    parser.add_argument(
        "--visibility",
        choices=("visible", "occluded", "unassessed"),
        default="unassessed",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(6)
    result = predict(
        args.checkpoint,
        args.sha256,
        args.data,
        args.sample,
        legal_slot=args.confirm_legal_slot,
        view_confirmed=args.confirm_view,
        visibility=args.visibility,
        device=args.device,
    )
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
