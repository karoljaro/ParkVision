"""Lineage DAG and separate identity, content and locality dependency components."""

from collections import defaultdict, deque
from dataclasses import dataclass
import json

from .contracts import (
    Evidence,
    Fact,
    Issue,
    Knowledge,
    Method,
    Policy,
    Registry,
    RelationKind,
    ReviewState,
    Verification,
    identifier,
)


class Components:
    def __init__(self, ids: list[str]) -> None:
        self.parent = {key: key for key in ids}

    def find(self, key: str) -> str:
        while key != self.parent[key]:
            self.parent[key] = self.parent[self.parent[key]]
            key = self.parent[key]
        return key

    def join(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


@dataclass
class Graph:
    photo: Components
    dependency: Components
    content: Components
    evidence: dict[str, Evidence]
    ancestors: dict[str, set[str]]


def confirmed(evidence: dict[str, Evidence], refs: tuple[str, ...]) -> bool:
    return bool(refs) and all(
        ref in evidence
        and evidence[ref].verification == Verification.CONFIRMED
        and evidence[ref].method != Method.HEURISTIC
        for ref in refs
    )


def build_graph(registry: Registry, policy: Policy) -> tuple[Graph, list[Issue]]:
    issues: list[Issue] = []
    records = {r.id: r for r in registry.records}
    ids = list(records)
    graph = Graph(
        Components(ids),
        Components(ids),
        Components(ids),
        {},
        {key: set() for key in ids},
    )

    def error(rule: str, keys: tuple[str, ...], message: str) -> None:
        issues.append(Issue(rule, keys, message))

    for collection in (
        registry.datasets,
        registry.records,
        registry.evidence,
        registry.relations,
        registry.candidates,
        registry.approvals,
    ):
        seen: set[str] = set()
        for item in collection:
            try:
                identifier(item.id)
            except ValueError as exc:
                error("INVALID_IDENTIFIER", (), str(exc))
            if item.id in seen:
                error("DUPLICATE_IDENTIFIER", (), f"Repeated {item.id}")
            seen.add(item.id)
    for ev in (*registry.evidence, *policy.evidence):
        if ev.id in graph.evidence and graph.evidence[ev.id] != ev:
            error("CONTRADICTORY_EVIDENCE", (), ev.id)
        graph.evidence[ev.id] = ev

    def refs(keys: tuple[str, ...], values: tuple[str, ...]) -> None:
        for ref in values:
            if ref not in graph.evidence:
                error("MISSING_EVIDENCE", keys, ref)

    datasets = {d.id for d in registry.datasets}
    for dataset in registry.datasets:
        refs((), dataset.evidence)
        refs((), dataset.license.evidence)
    degrees = {key: 0 for key in ids}
    children: dict[str, list[str]] = defaultdict(list)
    fact_owners: dict[tuple[str, str], str] = {}
    contents: dict[str, str] = {}
    for row in registry.records:
        refs((row.id,), row.evidence)
        if row.dataset not in datasets:
            error("MISSING_DATASET", (row.id,), row.dataset)
        for name in ("site", "camera", "session", "day", "family", "location_group"):
            fact: Fact = getattr(row, name)
            refs((row.id,), fact.evidence)
            if fact.state == Knowledge.KNOWN and name in {
                "site",
                "family",
                "location_group",
                "session",
            }:
                key = (name, fact.value)
                if key in fact_owners:
                    graph.dependency.join(row.id, fact_owners[key])
                fact_owners[key] = row.id
        if row.asset is not None:
            # Equal bytes do not establish physical-source identity (e.g. black
            # tiles from two cities); keep this component distinct from lineage.
            sha = row.asset.sha256
            if sha in contents:
                graph.content.join(row.id, contents[sha])
            contents[sha] = row.id
        if row.rgb_size_sha256 is not None:
            key = "rgb-size-le-v1:" + row.rgb_size_sha256
            if key in contents:
                graph.content.join(row.id, contents[key])
            contents[key] = row.id
        parents = set()
        for edge in row.parents:
            refs((row.id,), edge.evidence)
            if edge.parent in parents:
                error("DUPLICATE_PARENT", (row.id,), edge.parent)
            parents.add(edge.parent)
            try:
                parsed = json.loads(edge.parameters_json)
                if not isinstance(parsed, dict):
                    raise ValueError("Expected parameter object")
            except (ValueError, TypeError):
                error("INVALID_TRANSFORM", (row.id,), edge.parameters_json)
            if edge.parent not in records:
                error("MISSING_PARENT", (row.id,), edge.parent)
                continue
            if not confirmed(graph.evidence, edge.evidence):
                error(
                    "UNCONFIRMED_LINEAGE",
                    (row.id, edge.parent),
                    "A heuristic is not approved lineage",
                )
                continue
            degrees[row.id] += 1
            children[edge.parent].append(row.id)
            graph.photo.join(row.id, edge.parent)
            graph.dependency.join(row.id, edge.parent)
        for annotation in row.annotations:
            refs((row.id,), annotation.evidence)
            if len(set(annotation.classes)) != len(annotation.classes):
                error("INVALID_ANNOTATION", (row.id,), "Duplicate class")
            if set(annotation.exhaustive_classes) - set(annotation.classes):
                error(
                    "INVALID_ANNOTATION",
                    (row.id,),
                    "Exhaustive classes outside source ontology",
                )
    ready = deque(key for key, count in degrees.items() if count == 0)
    while ready:
        key = ready.popleft()
        for child in children[key]:
            graph.ancestors[child].update(graph.ancestors[key])
            graph.ancestors[child].add(key)
            degrees[child] -= 1
            if degrees[child] == 0:
                ready.append(child)
    residual = tuple(sorted(key for key, count in degrees.items() if count))
    if residual:
        error("LINEAGE_CYCLE", residual, "Cycle or descendant of a cycle")

    relations = {r.id: r for r in registry.relations}
    for link in registry.relations:
        refs((link.left, link.right), link.evidence)
        if link.left not in records or link.right not in records:
            error("MISSING_RELATION_ENDPOINT", (link.left, link.right), link.id)
        elif not confirmed(graph.evidence, link.evidence):
            error("UNCONFIRMED_RELATION", (link.left, link.right), link.id)
        elif link.kind == RelationKind.SAME_PHOTOGRAPH:
            graph.photo.join(link.left, link.right)
            graph.dependency.join(link.left, link.right)
        elif link.kind == RelationKind.LOCATION_DEPENDENCY:
            graph.dependency.join(link.left, link.right)
    for link in registry.relations:
        if (
            link.kind == RelationKind.DIFFERENT_ORIGIN
            and link.left in records
            and link.right in records
        ):
            if graph.photo.find(link.left) == graph.photo.find(link.right):
                error("CONTRADICTORY_ORIGIN", (link.left, link.right), link.id)
    for candidate in registry.candidates:
        refs((candidate.left, candidate.right), candidate.evidence)
        if (
            candidate.left == candidate.right
            or not candidate.method
            or not candidate.evidence
        ):
            error("INVALID_CANDIDATE", (candidate.left, candidate.right), candidate.id)
        if candidate.left not in records or candidate.right not in records:
            error(
                "MISSING_CANDIDATE_ENDPOINT",
                (candidate.left, candidate.right),
                candidate.id,
            )
        if candidate.status == ReviewState.CONFIRMED:
            relation = relations.get(candidate.relation)
            if relation is None or {relation.left, relation.right} != {
                candidate.left,
                candidate.right,
            }:
                error(
                    "MISSING_REVIEWED_RELATION",
                    (candidate.left, candidate.right),
                    candidate.id,
                )
        elif candidate.relation is not None or (
            candidate.status == ReviewState.REJECTED
            and not confirmed(graph.evidence, candidate.evidence)
        ):
            error(
                "INVALID_CANDIDATE_REVIEW",
                (candidate.left, candidate.right),
                candidate.id,
            )
    for approval in registry.approvals:
        refs(approval.records, approval.evidence)
        if not approval.records or any(key not in records for key in approval.records):
            error("INVALID_APPROVAL_SCOPE", approval.records, approval.id)
        if (
            not approval.reviewer
            or not approval.rationale
            or not confirmed(graph.evidence, approval.evidence)
        ):
            error("UNAPPROVED_PROTOCOL", approval.records, approval.id)
        elif not any(
            graph.evidence[key].method == Method.OPERATOR for key in approval.evidence
        ):
            error(
                "UNAPPROVED_PROTOCOL",
                approval.records,
                "Explicit operator approval required",
            )
    return graph, issues
