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
        "preview_marking_axes.png",
        "markings.json",
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
    markings = json.loads((result / "markings.json").read_text(encoding="utf-8"))
    assert markings["pairs"] == []
    assert len(markings["groups"]) == len(markings["unpaired_edge_ids"]) == 1
    assert markings["groups"][0]["source_segment_ids"] == [4, 17]


def test_batch_links_axes_to_groups_and_preserves_unpaired_sources_and_gaps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = np.zeros((140, 200, 3), dtype=np.uint8)
    sample_path = tmp_path / "sample.png"
    assert cv2.imwrite(str(sample_path), image)
    sources = (
        LineSegment(4, (10.0, 60.0), (60.0, 60.0)),
        LineSegment(17, (68.0, 60.0), (120.0, 60.0)),
        LineSegment(19, (20.0, 68.0), (60.0, 68.0)),
        LineSegment(20, (68.0, 68.0), (110.0, 68.0)),
        LineSegment(26, (150.0, 100.0), (175.0, 100.0)),
    )
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
            "--marking-min-width",
            "6",
            "--marking-max-width",
            "10",
            "--marking-angle-tolerance",
            "2",
            "--marking-min-overlap",
            "40",
            "--marking-min-overlap-ratio",
            "0.7",
            "--marking-max-width-ratio",
            "1.2",
        ],
    )
    script = Path(__file__).resolve().parents[2] / "scripts" / "run_parking_batch.py"

    runpy.run_path(str(script), run_name="__main__")

    result = output_path / "image-1"
    markings = json.loads((result / "markings.json").read_text(encoding="utf-8"))
    assert markings["pairing_parameters"] == {
        "min_width_px": 6,
        "max_width_px": 10,
        "angle_tolerance": 2,
        "min_overlap_px": 40,
        "min_overlap_ratio": 0.7,
        "max_width_ratio": 1.2,
    }
    (pair,) = markings["pairs"]
    groups = markings["groups"]
    assert pair["group_indices"] == [0, 1]
    assert [groups[i]["source_segment_ids"] for i in pair["group_indices"]] == [
        [4, 17],
        [19, 20],
    ]
    assert pair["edge_segment_ids"] == [
        groups[i]["fitted_line"]["segment_id"] for i in pair["group_indices"]
    ]
    assert pair["centerline"]["start"] == pytest.approx((20, 64))
    assert pair["centerline"]["end"] == pytest.approx((110, 64))
    assert pair["widths_px"] == pytest.approx((8, 8))
    assert markings["unpaired_edge_ids"] == [groups[2]["fitted_line"]["segment_id"]]
    assert groups[2]["source_segment_ids"] == [26]
    raw = json.loads((result / "segments.json").read_text(encoding="utf-8"))
    assert [s["segment_id"] for s in raw["segments"]] == [4, 17, 19, 20, 26]
    source_preview = cv2.imread(str(result / "preview_groups.png"))
    axes = cv2.imread(str(result / "preview_marking_axes.png"))
    assert source_preview is not None and axes is not None
    assert not source_preview[60, 64].any() and not source_preview[68, 64].any()
    assert axes[64, 64, 1] > 200  # Green axis can span original gaps.
    assert axes[100, 160].any()  # Unpaired edge stays visible in gray.
    assert not axes[64, 15].any()  # No axis beyond the common range.


@pytest.mark.parametrize("marked", [False, True], ids=["blank", "painted-stripe"])
def test_real_frame_lsd_group_fit_pair_batch_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    marked: bool,
) -> None:
    image = np.zeros((180, 300, 3), dtype=np.uint8)
    if marked:
        cv2.rectangle(image, (40, 70), (250, 78), (255, 255, 255), -1)
    sample_path = tmp_path / "sample.png"
    assert cv2.imwrite(str(sample_path), image)
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
        ],
    )
    script = Path(__file__).resolve().parents[2] / "scripts" / "run_parking_batch.py"

    runpy.run_path(str(script), run_name="__main__")

    result = output_path / "image-1"
    markings = json.loads((result / "markings.json").read_text(encoding="utf-8"))
    if marked:
        assert any(
            p["centerline"]["length"] > 180
            and abs(p["centerline"]["start"][1] - 74) < 1
            and abs(p["centerline"]["end"][1] - 74) < 1
            for p in markings["pairs"]
        )
    else:
        assert (
            markings["groups"]
            == markings["pairs"]
            == markings["unpaired_edge_ids"]
            == []
        )
    assert cv2.imread(str(result / "preview_marking_axes.png")) is not None


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--marking-min-width", "0"),
        ("--marking-max-width", "1"),
        ("--marking-angle-tolerance", "nan"),
        ("--marking-min-overlap", "0"),
        ("--marking-min-overlap-ratio", "1.1"),
        ("--marking-max-width-ratio", "0.5"),
    ],
)
def test_rejects_invalid_pairing_options_before_writing_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    flag: str,
    value: str,
) -> None:
    output_path = tmp_path / "results"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_parking_batch.py",
            "--output",
            str(output_path),
            flag,
            value,
        ],
    )
    script = Path(__file__).resolve().parents[2] / "scripts" / "run_parking_batch.py"
    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(script), run_name="__main__")
    assert error.value.code == 2
    assert not output_path.exists()
