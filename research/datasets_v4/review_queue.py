"""Validate and stream large unresolved similarity queues without joining origins."""

from collections import defaultdict
import json
from pathlib import Path

from .contracts import Candidate, Issue, Registry, ReviewState, identifier
from .graph import Graph
from .storage import AssetVerifier, decode


def review_queue(
    registry: Registry, graph: Graph, verifier: AssetVerifier
) -> tuple[dict[str, list[str]], list[Issue]]:
    pending: dict[str, list[str]] = defaultdict(list)
    issues: list[Issue] = []
    records = graph.photo.parent
    ids = {c.id for c in registry.candidates}
    for candidate in registry.candidates:
        if (
            candidate.status == ReviewState.PENDING
            and candidate.left in records
            and candidate.right in records
        ):
            pending[graph.photo.find(candidate.left)].append(candidate.id)
            pending[graph.photo.find(candidate.right)].append(candidate.id)
    for asset in registry.candidate_indices:
        try:
            if asset.member is not None or not verifier.verify(asset):
                raise ValueError("Candidate queue integrity mismatch")
            with Path(asset.path).open() as stream:
                for line in stream:
                    candidate = decode(Candidate, json.loads(line))
                    identifier(candidate.id)
                    if (
                        candidate.id in ids
                        or candidate.left == candidate.right
                        or not candidate.method
                    ):
                        raise ValueError("Duplicate candidate ID or self-link")
                    ids.add(candidate.id)
                    if candidate.left not in records or candidate.right not in records:
                        raise ValueError("Candidate endpoint missing")
                    if not candidate.evidence or any(
                        key not in graph.evidence for key in candidate.evidence
                    ):
                        raise ValueError("Candidate evidence missing")
                    if (
                        candidate.status != ReviewState.PENDING
                        or candidate.relation is not None
                    ):
                        raise ValueError(
                            "Reviewed candidates belong in the reviewed relation list"
                        )
                    # Bound report size while preserving the complete immutable
                    # queue. This records uncertainty, never a union operation.
                    for key in (candidate.left, candidate.right):
                        component = graph.photo.find(key)
                        if len(pending[component]) < 3:
                            pending[component].append(candidate.id)
        except (OSError, ValueError, TypeError) as exc:
            issues.append(Issue("CANDIDATE_INDEX_INVALID", (), str(exc)))
    return pending, issues
