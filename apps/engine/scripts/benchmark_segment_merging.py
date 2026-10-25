"""Compare grouping with a Git revision on one saved, unfiltered LSD extraction.

Run run_parking_batch.py with --samples samples/parking.jpg first, then:
    .venv/bin/python scripts/benchmark_segment_merging.py \
        --segments /path/to/image-1/segments.json --baseline-ref 8ce185c \
        --repeats 3 --output /path/outside/repository/benchmark.json

Only grouping is timed; extraction, rendering, Git and JSON I/O are excluded.
The baseline source is loaded without changing the working tree.
"""

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path
from statistics import median
from time import perf_counter
from types import ModuleType

from parkvision.detection.line_segment import LineSegment
from parkvision.detection.segment_group import SegmentGroup
from parkvision.detection.segment_merging import group_segments


def canonical_groups(groups: tuple[SegmentGroup, ...]) -> list[list[int]]:
    return sorted(sorted(segment.segment_id for segment in g.segments) for g in groups)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments", type=Path, required=True)
    parser.add_argument("--baseline-ref", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")

    source = json.loads(args.segments.read_text())
    segments = tuple(
        LineSegment(item["segment_id"], tuple(item["start"]), tuple(item["end"]))
        for item in source["segments"]
    )
    tolerances = dict(angle_tolerance=3.0, distance_tolerance=2.0, gap_tolerance=12.0)
    baseline_path = "apps/engine/src/parkvision/detection/segment_merging.py"
    baseline_source = subprocess.run(
        ["git", "show", f"{args.baseline_ref}:{baseline_path}"],
        cwd=Path(__file__).resolve().parents[3],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    baseline = ModuleType("parkvision.detection._benchmark_baseline")
    sys.modules[baseline.__name__] = baseline
    exec(
        compile(baseline_source, f"{args.baseline_ref}:{baseline_path}", "exec"),
        baseline.__dict__,
    )

    times: dict[str, list[float]] = {"before": [], "after": []}
    expected = None
    for repeat in range(args.repeats):
        # Alternate order to avoid consistently favoring one implementation.
        implementations = [
            ("before", baseline.group_segments),
            ("after", group_segments),
        ]
        for name, implementation in implementations[:: 1 if repeat % 2 == 0 else -1]:
            started = perf_counter()
            groups = implementation(segments, **tolerances)
            elapsed = perf_counter() - started
            times[name].append(elapsed)
            if expected is None:
                expected = groups
            assert canonical_groups(groups) == canonical_groups(expected)
            assert groups == expected, "Group/member order differs from the baseline"
            assert sorted(id(s) for g in groups for s in g.segments) == sorted(
                map(id, segments)
            ), "Source segments were lost, duplicated or replaced"
            print(f"{name} [{repeat + 1}/{args.repeats}]: {elapsed:.6f} s", flush=True)

    report = {
        "sample": str(args.segments.resolve()),
        "source": source.get("source"),
        "opencv_version": source.get("opencv_version"),
        "python": sys.version,
        "platform": platform.platform(),
        "baseline_ref": args.baseline_ref,
        "segment_count": len(segments),
        "group_count": len(expected),
        "tolerances": tolerances,
        "seconds": times,
        "median_seconds": {name: median(values) for name, values in times.items()},
        "speedup": median(times["before"]) / median(times["after"]),
        "canonical_segment_ids": canonical_groups(expected),
        "ordered_segment_ids": [[s.segment_id for s in g.segments] for g in expected],
        "exact_groups_and_source_objects_preserved": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{len(segments)} segments, {len(expected)} groups; {report['speedup']:.1f}x")


if __name__ == "__main__":
    main()
