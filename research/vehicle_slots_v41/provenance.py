"""Use v4.0 as the source authority for diagnostic queue eligibility."""

from research.datasets_v4.contracts import (
    Assignment,
    Knowledge,
    Manifest,
    Policy,
    Record,
    Registry,
    Role,
    Verification,
)
from research.datasets_v4.gate import validate
from research.datasets_v4.graph import Graph, build_graph, confirmed
from research.datasets_v4.storage import fingerprint


class Provenance:
    def __init__(self, registry: Registry, policy: Policy, curated: Manifest) -> None:
        self.registry = registry
        self.policy = policy
        self.registry_sha256 = fingerprint(registry)
        self.policy_sha256 = fingerprint(policy)
        self.curated_sha256 = fingerprint(curated)
        if (
            self.registry_sha256 != curated.registry_sha256
            or self.policy_sha256 != curated.policy_sha256
        ):
            raise ValueError("The existing v4 manifest does not bind these inputs")
        self.graph, issues = build_graph(registry, policy)
        if issues:
            raise ValueError(f"Invalid v4 lineage: {issues[0]}")
        self.records = {r.id: r for r in registry.records}
        self.roles = {a.record: a.role for a in curated.assignments}
        if len(self.roles) != len(curated.assignments):
            raise ValueError("Conflicting v4 assignments")
        protected = {
            key for key, role in self.roles.items() if role == Role.PROTECTED_TEST
        }
        protected.update(
            lock.record for lock in policy.locks if lock.role == Role.PROTECTED_TEST
        )
        protected.update(
            row.id
            for row in registry.records
            if row.historical_role == Role.PROTECTED_TEST
            or (row.asset is not None and row.asset.sha256 in policy.protected_assets)
        )
        self.protected_groups = {
            name: {component.find(key) for key in protected}
            for name, component in components(self.graph)
        }
        self.protected_groups["site"] = set(policy.protected_sites)

    def review_source(self, key: str) -> tuple[Record, Role]:
        row = self.records[key]
        if key not in self.roles:
            raise ValueError(f"{key}: source is absent from the curated v4 manifest")
        ancestry = [row, *(self.records[i] for i in self.graph.ancestors[key])]
        for ancestor in ancestry:
            if (
                any(
                    component.find(ancestor.id) in self.protected_groups[name]
                    for name, component in components(self.graph)
                )
                or ancestor.site.value in self.protected_groups["site"]
            ):
                raise ValueError(
                    f"{key}: protected v4 source, ancestry or physical site"
                )
        historical = self.roles[key]
        if Role.TEST in {historical, row.historical_role}:
            raise ValueError(f"{key}: reserved test role")
        if (
            row.asset is None
            or row.asset.member is not None
            or any(
                a.provenance != Verification.CONFIRMED
                or not confirmed(self.graph.evidence, a.evidence)
                for a in ancestry
            )
            or any(
                fact.state != Knowledge.KNOWN
                or not confirmed(self.graph.evidence, fact.evidence)
                for fact in (row.site, row.camera)
            )
        ):
            raise ValueError(f"{key}: source identity, site or camera requires review")
        return row, historical

    def check(self, records: tuple[str, ...]):
        assignments = []
        for key in records:
            self.review_source(key)
            assignments.append(Assignment(key, Role.DIAGNOSTIC))
        manifest = Manifest(
            "v4.1-ab-diagnostic",
            self.registry_sha256,
            self.policy_sha256,
            tuple(assignments),
        )
        return manifest, validate(self.registry, self.policy, manifest)


def components(graph: Graph):
    return (
        ("photo", graph.photo),
        ("content", graph.content),
        ("dependency", graph.dependency),
    )
