"""Verify adapters against frozen code on two frames/eight source ROIs."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from . import models, observe
from .rules import classify
from .training_v1.final_compare import acpds_frames, module_from_file


def run(protocol_path, rcnn_assets, convnext_assets, output, device="cpu"):
    output.mkdir(exist_ok=False, parents=True)
    protocol = json.loads(protocol_path.read_text())
    files = protocol["files"]
    # Verify every frozen file actually consumed by this smoke check.
    keys = [
        "candidate",
        "baseline_code",
        "candidate_code___init__.py",
        "candidate_code_model.py",
        "candidate_code_data.py",
        "acpds_manifest",
        "acpds_annotations",
        "cnr_manifest",
    ]
    for key in keys:
        if models.file_hash(Path(files[key]["path"])) != files[key]["sha256"]:
            raise ValueError(f"Changed frozen input: {key}")
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    baseline = models.BaselineModels(rcnn_assets, device)
    candidate = models.ConvNeXtModels(convnext_assets, device)
    frozen_baseline = module_from_file(
        "_pv_v2_smoke_baseline", Path(files["baseline_code"]["path"])
    ).BaselineModels(rcnn_assets, device)
    code = Path(files["candidate_code_model.py"]["path"]).parent
    module_from_file("_pv_v2_smoke_candidate", code / "__init__.py", package=True)
    frozen_model = module_from_file("_pv_v2_smoke_candidate.model", code / "model.py")
    from importlib import import_module

    frozen_data = import_module("_pv_v2_smoke_candidate.data")
    reference = frozen_model.build_model()
    reference.load_state_dict(
        torch.load(files["candidate"]["path"], map_location="cpu", weights_only=True)[
            "state_dict"
        ]
    )
    reference.eval().to(device)
    first = acpds_frames(protocol)[0]
    cnr_manifest = json.loads(Path(files["cnr_manifest"]["path"]).read_text())
    second = next(f for f in cnr_manifest["frames"] if f["camera_id"] == "camera1")
    second = dict(second, root=protocol["cohorts"]["CNR-Pisa"]["root"])
    second["rois"] = [
        (
            np.float32(s["roi"])
            / np.float32((second["width"] - 1, second["height"] - 1))
        ).tolist()
        for s in second["slots"]
    ]
    checks = []
    for index, source in enumerate((first, second)):
        image = Path(source["root"]) / source["image"]
        if models.file_hash(image) != source["sha256"]:
            raise ValueError("Changed source frame")
        bgr = cv2.imread(str(image), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        height, width = bgr.shape[:2]
        # Source order only, never select by score or correctness.
        rois = source["rois"][:4]
        camera = source.get("camera_id", "ACPDS-test-unmapped-view")
        config = {
            "schema_version": 1,
            "revision": 1,
            "approved_by": "source-geometry-integration-fixture-only",
            "approved_at": "2026-11-09T00:00:00+00:00",
            "parking": {"id": f"smoke-only-{index}", "name": "Source ROI fixture"},
            "cameras": [
                {
                    "id": camera,
                    "parking_id": f"smoke-only-{index}",
                    "name": camera,
                    "image_width": width,
                    "image_height": height,
                }
            ],
            "slots": [
                {
                    "id": f"source-roi-{i}",
                    "parking_id": f"smoke-only-{index}",
                    "label": None,
                }
                for i in range(len(rois))
            ],
            "projections": [
                {
                    "camera_id": camera,
                    "parking_slot_id": f"source-roi-{i}",
                    "polygon": [{"x": x, "y": y} for x, y in roi],
                }
                for i, roi in enumerate(rois)
            ],
        }
        config_path = output / f"source-{index}-configuration.json"
        observe.save_result(config_path, config)
        # ACPDS has no reliable capture timestamp. Deliberately synthetic replay
        # time exercises freshness; it is never represented as source EXIF time.
        replay_time = datetime(2026, 11, 9, tzinfo=timezone.utc)
        args = argparse.Namespace(
            configuration=config_path,
            camera_id=camera,
            image=image,
            captured_at=replay_time,
            evaluated_at=replay_time,
            max_age_seconds=60,
            view_confirmed=True,
            session_id="v2-integration-fixture-synthetic-clock",
            sequence=index,
            unreadable_slot_id=[],
            device=device,
        )
        configuration = observe.load_configuration(config_path)
        projections = configuration.projections_for_camera(
            camera, image_width=width, image_height=height
        )
        rgb = bgr[:, :, ::-1].copy()
        polygons = np.float32(rois) * np.float32((width - 1, height - 1))
        batch = torch.stack(
            [
                frozen_model.tensor_input(frozen_data.encode_slot(rgb, p.tolist())[0])
                for p in polygons
            ]
        ).to(device)
        with torch.inference_mode():
            expected_candidate = reference(batch).softmax(1)[:, 1].tolist()
        frame = observe.Frame(camera, index, replay_time, bgr)
        expected_baseline, expected_detections = frozen_baseline.predict(
            frame, projections
        )
        for backend, adapter, assets, expected in (
            ("acpds-rcnn", baseline, rcnn_assets, expected_baseline),
            ("convnext-roi4", candidate, convnext_assets, expected_candidate),
        ):
            args.model, args.assets = backend, assets
            result = observe.observe(args, lambda *a, **k: adapter)
            if result["status"] != "observed":
                raise AssertionError(result["error_detail"])
            actual = [r["p_occupied"] for r in result["observations"]]
            delta = max(abs(a - b) for a, b in zip(actual, expected, strict=True))
            if delta > 1e-5 or any(
                classify(a, []) != classify(b, [])
                for a, b in zip(actual, expected, strict=True)
            ):
                raise AssertionError(
                    f"Frozen inferencer divergence: {backend}: {delta}"
                )
            if result["detections"] != expected_detections:
                raise AssertionError("Detector changed across backends/frozen baseline")
            observe.save_result(output / f"source-{index}-{backend}.json", result)
            checks.append(
                {
                    "model": backend,
                    "image_sha256": source["sha256"],
                    "roi_count": len(rois),
                    "max_abs_score_difference": delta,
                    "score_band_predictions_equal": True,
                    "detector_equal": True,
                }
            )
    summary = {
        "kind": "integration_smoke_only",
        "frames": 2,
        "physical_source_rois": 8,
        "device": device,
        "torch": str(torch.__version__),
        "tolerance": 1e-5,
        "checks": checks,
        "capture_clock": "synthetic replay fixture, not acquisition time",
        "production_validation": False,
        "training_or_tuning": False,
    }
    observe.save_result(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--rcnn-assets", type=Path, required=True)
    parser.add_argument("--convnext-assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.protocol,
                args.rcnn_assets,
                args.convnext_assets,
                args.output,
                args.device,
            )
        )
    )


if __name__ == "__main__":
    main()
