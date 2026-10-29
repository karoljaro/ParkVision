import json
import runpy
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from parkvision.detection.line_segment import LineSegment
from parkvision.detection.opencv_slot_detector import OpenCVSlotDetector


def test_batch_adds_fitted_preview_and_preserves_raw_gaps_and_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = np.zeros((120, 180, 3), dtype=np.uint8)
    sample_path = tmp_path / "sample.png"
    assert cv2.imwrite(str(sample_path), image)
    sources = (
        LineSegment(4, (10.0, 60.0), (60.0, 60.0)),
        LineSegment(17, (68.0, 60.0), (120.0, 60.0)),
    )
    # Stable extraction data lets this test inspect an exact gap in real previews.
    monkeypatch.setattr(OpenCVSlotDetector, "extract_segments", lambda *_: sources)
    output_path = tmp_path / "results"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_parking_batch.py",
            "--samples",
            str(sample_path),
            "--output",
            str(output_path),
            "--min-length",
            "1000",
        ],
    )

    script = Path(__file__).resolve().parents[2] / "scripts" / "run_parking_batch.py"
    runpy.run_path(str(script), run_name="__main__")

    result = output_path / "image-1"
    assert {p.name for p in result.iterdir()} == {
        "preview_lsd.png",
        "preview_groups.png",
        "preview_fitted_lines.png",
        "segments.json",
        "source.txt",
    }
    lsd = cv2.imread(str(result / "preview_lsd.png"))
    groups = cv2.imread(str(result / "preview_groups.png"))
    fitted = cv2.imread(str(result / "preview_fitted_lines.png"))
    np.testing.assert_array_equal(lsd, image)  # Preview threshold affects LSD only.
    assert groups is not None and fitted is not None
    assert groups[60, 40].any()
    assert not groups[60, 64].any()  # Source fragments retain the eight-pixel gap.
    assert fitted[60, 64].any()  # Extent spans the gap only in its separate preview.
    data = json.loads((result / "segments.json").read_text(encoding="utf-8"))
    assert data["coordinate_system"] == "pixels"
    assert data["segments"] == [
        {
            "segment_id": s.segment_id,
            "start": list(s.start),
            "end": list(s.end),
            "length": s.length,
        }
        for s in sources
    ]
    assert (result / "source.txt").read_text(encoding="utf-8").strip() == str(
        sample_path
    )
