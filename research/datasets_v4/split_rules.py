"""Split checks do not conflate same content, common photo and physical site."""

from collections import defaultdict

from .contracts import Issue, Knowledge, Manifest, Record, Registry, Role
from .graph import Graph

SUPERVISED = {Role.TRAIN, Role.VALIDATION, Role.TEST, Role.PROTECTED_TEST}


def check_splits(
    registry: Registry,
    manifest: Manifest,
    graph: Graph,
    assigned: dict[str, Role],
    records: dict[str, Record],
) -> list[Issue]:
    issues: list[Issue] = []

    def error(rule: str, keys: tuple[str, ...], message: str) -> None:
        issues.append(Issue(rule, keys, message))

    # Check all intended supervised roles, including records rejected for other
    # reasons. Quarantine must not hide the known leakage in historical v3.
    for name, component in (
        ("LOCATION_OR_FAMILY_SPLIT", graph.dependency),
        ("PHOTOGRAPH_SPLIT", graph.photo),
        ("IDENTICAL_CONTENT_SPLIT", graph.content),
    ):
        groups: dict[str, list[str]] = defaultdict(list)
        for key, role in assigned.items():
            if key in records and role in SUPERVISED:
                groups[component.find(key)].append(key)
        for keys in groups.values():
            roles = {
                Role.TEST if assigned[key] == Role.PROTECTED_TEST else assigned[key]
                for key in keys
            }
            if len(roles) > 1:
                error(
                    name,
                    tuple(sorted(keys)),
                    "Related source/locality/content occurs in incompatible splits",
                )
    # Within one photograph, independently asserted sites cannot disagree.
    exact_targets: dict[tuple, list[str]] = defaultdict(list)
    for assignment in manifest.assignments:
        row = records.get(assignment.record)
        if (
            row is not None
            and row.asset is not None
            and assignment.task is not None
            and assignment.role in {Role.TRAIN, Role.VALIDATION}
        ):
            signature = tuple(
                sorted(
                    (a.meaning.value, a.binding.sha256 if a.binding else "", a.locator)
                    for a in row.annotations
                )
            )
            exact_targets[(row.asset.sha256, assignment.task, signature)].append(row.id)
    for keys in exact_targets.values():
        if len(keys) > 1:
            error(
                "DUPLICATE_LEARNING_TARGET",
                tuple(keys),
                "Same image/task/target view appears more than once",
            )
    sites: dict[str, set[str]] = defaultdict(set)
    members: dict[str, list[str]] = defaultdict(list)
    for row in registry.records:
        key = graph.photo.find(row.id)
        members[key].append(row.id)
        if row.site.state == Knowledge.KNOWN:
            sites[key].add(row.site.value)
    for key, values in sites.items():
        if len(values) > 1:
            error(
                "CONTRADICTORY_PROVENANCE",
                tuple(members[key]),
                f"Same photograph asserts different sites: {sorted(values)}",
            )
    return issues
