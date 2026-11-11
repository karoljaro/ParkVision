"""Prepare a visual A/B batch or validate an operator's exported progress."""

import argparse
from pathlib import Path

from .batch import prepare
from .export import export


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("prepare")
    for name in ("v4", "v41", "selection", "output"):
        build.add_argument(f"--{name}", type=Path, required=True)
    write = commands.add_parser("export")
    for name in ("batch", "progress", "output"):
        write.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            batch = prepare(args.v4, args.v41, args.selection, args.output)
            print(f"Examples prepared: {len(batch.examples)}")
        else:
            count = export(args.batch, args.progress, args.output)
            print(f"Accepted diagnostic decisions exported: {count}")
    except (OSError, KeyError, ValueError, TypeError) as exc:
        parser.exit(2, f"{exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
