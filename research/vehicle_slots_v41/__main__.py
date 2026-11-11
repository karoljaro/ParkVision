"""Prepare or annotate a read-only UFPR04 diagnostic queue."""

import argparse
import json
from pathlib import Path

from research.datasets_v4.storage import AssetVerifier, load, save

from .prepare import Selection, prepare
from .relations import Queue
from .review import apply, read_csv


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    build = actions.add_parser("prepare")
    for name in ("v4", "cars", "spots", "tree", "selection", "output"):
        build.add_argument(f"--{name}", type=Path, required=True)
    review = actions.add_parser("review")
    for name in ("queue", "csv", "output"):
        review.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.action == "prepare":
            selections = tuple(
                Selection(**row) for row in json.loads(args.selection.read_text())
            )
            queue = prepare(
                args.v4,
                args.cars,
                args.spots,
                args.tree,
                selections,
                args.output,
            )
            print(
                json.dumps(
                    {
                        "frames": len(queue.frames),
                        "held": len(selections) - len(queue.frames),
                        "pairs": sum(len(frame.pairs) for frame in queue.frames),
                    }
                )
            )
            return 0 if len(queue.frames) == len(selections) else 2
        queue = load(args.queue, Queue)
        verifier = AssetVerifier()
        assets = [bundle.archive for bundle in queue.bundles]
        for frame in queue.frames:
            assets.extend((frame.image, frame.source_subset))
        for asset in assets:
            if not verifier.verify(asset):
                raise ValueError(f"Source integrity mismatch: {asset.path}")
        save(args.output, apply(queue, read_csv(args.csv)))
        return 0
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(json.dumps({"error": str(exc)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
