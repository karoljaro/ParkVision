"""Offline provenance registry CLI; invalid manifests return nonzero status."""

import argparse
import json
from pathlib import Path

from .compiler import compile_local
from .contracts import Manifest, Policy, Registry
from .gate import validate
from .storage import load, save


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser(
        "build", help="Compile reviewed local sources into a fresh artifact directory"
    )
    build.add_argument("--base", type=Path, required=True)
    build.add_argument("--audit", type=Path, required=True)
    build.add_argument("--v3", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    check = commands.add_parser(
        "validate", help="Validate an intended assignment; no automatic fallback"
    )
    check.add_argument("--registry", type=Path, required=True)
    check.add_argument("--policy", type=Path, required=True)
    check.add_argument("--manifest", type=Path, required=True)
    check.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "build":
            summary = compile_local(args.base, args.audit, args.v3, args.output)
            print(json.dumps(summary, sort_keys=True))
            return 0
        registry = load(args.registry, Registry)
        policy = load(args.policy, Policy)
        manifest = load(args.manifest, Manifest)
        result = validate(registry, policy, manifest)
        save(args.output, result)
        print(
            json.dumps(
                {
                    "valid": result.valid,
                    "issues": len(result.issues),
                    "output": str(args.output),
                }
            )
        )
        return 0 if result.valid else 2
    except (OSError, ValueError, TypeError, KeyError) as exc:
        error = {
            "valid": False,
            "fatal": True,
            "issues": [
                {
                    "rule": "INVALID_CONTRACT_OR_INPUT",
                    "records": [],
                    "message": str(exc),
                }
            ],
        }
        if args.command == "validate" and not args.output.exists():
            save(args.output, error)
        print(json.dumps(error))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
