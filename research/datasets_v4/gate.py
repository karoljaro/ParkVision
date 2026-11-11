"""Validation decisions never promote uncertainty or rewrite historical roles."""

from collections import defaultdict

from .contracts import (
    Decision,
    Issue,
    Knowledge,
    Manifest,
    Policy,
    Registry,
    Role,
    ValidationResult,
    Verification,
    identifier,
)
from .graph import build_graph, confirmed
from .review_queue import review_queue
from .split_rules import SUPERVISED, check_splits
from .storage import AssetVerifier, fingerprint
from .supervision import check_supervision


def validate(
    registry: Registry, policy: Policy, manifest: Manifest
) -> ValidationResult:
    graph, structural = build_graph(registry, policy)
    issues = list(structural)
    records = {row.id: row for row in registry.records}
    datasets = {row.id: row for row in registry.datasets}
    approvals = {row.id: row for row in registry.approvals}
    tasks = {row.id: row for row in manifest.tasks}
    verifier = AssetVerifier()

    def error(
        rule: str, keys: tuple[str, ...], text: str, evidence: tuple[str, ...] = ()
    ) -> None:
        issues.append(Issue(rule, keys, text, evidence))

    for value in (registry, policy, manifest):
        if value.schema_version != 1:
            error("UNSUPPORTED_SCHEMA", (), "Only schema_version=1 is supported")
        try:
            identifier(value.id)
        except ValueError as exc:
            error("INVALID_IDENTIFIER", (), str(exc))
    if manifest.registry_sha256 != fingerprint(registry):
        error("REGISTRY_HASH_MISMATCH", (), "Manifest was created for another registry")
    if manifest.policy_sha256 != fingerprint(policy):
        error(
            "POLICY_HASH_MISMATCH", (), "Historical policy is a separate trusted input"
        )
    if len(tasks) != len(manifest.tasks):
        error("DUPLICATE_TASK", (), "Repeated task ID")
    for task in manifest.tasks:
        try:
            identifier(task.id)
        except ValueError as exc:
            error("INVALID_IDENTIFIER", (), str(exc))
        if not task.classes or len(set(task.classes)) != len(task.classes):
            error("INVALID_TASK", (), task.id)
        if task.approval is not None and task.approval not in approvals:
            error("MISSING_APPROVAL", (), task.approval)
    for ev in graph.evidence.values():
        try:
            if not verifier.verify(ev.asset):
                raise ValueError("Evidence hash mismatch")
        except (OSError, ValueError) as exc:
            error("EVIDENCE_INTEGRITY", (), f"{ev.id}: {exc}", (ev.id,))
    assigned: dict[str, Role] = {}
    for assignment in manifest.assignments:
        if assignment.record in assigned:
            error(
                "DUPLICATE_ASSIGNMENT",
                (assignment.record,),
                "Repeated/conflicting manifest assignment",
            )
        assigned[assignment.record] = assignment.role
        if assignment.record not in records:
            error(
                "UNKNOWN_RECORD",
                (assignment.record,),
                "Manifest references missing record",
            )
    locks: dict[str, list[Role]] = defaultdict(list)
    direct_locks: dict[str, list[Role]] = defaultdict(list)
    protected_photos: set[str] = set()
    for lock in policy.locks:
        if lock.record not in records:
            error("MISSING_POLICY_ANCHOR", (), lock.record)
        else:
            locks[graph.photo.find(lock.record)].append(lock.role)
            direct_locks[lock.record].append(lock.role)
            row = records[lock.record]
            if row.historical_role != lock.role or any(
                expected is not None and getattr(row, field) != expected
                for field, expected in (
                    ("training_eligible", lock.training_eligible),
                    ("geometry_excluded", lock.geometry_excluded),
                    ("is_parking_slot", lock.is_parking_slot),
                )
            ):
                error(
                    "HISTORICAL_METADATA_CHANGED",
                    (row.id,),
                    "Registry contradicts trusted historical policy",
                )
            if lock.role == Role.PROTECTED_TEST:
                protected_photos.add(graph.photo.find(lock.record))
        if not confirmed(graph.evidence, lock.evidence):
            error("UNCONFIRMED_POLICY", (), lock.record)
    for key, roles in locks.items():
        split_roles = {role for role in roles if role in SUPERVISED}
        if len(split_roles - {Role.TEST, Role.PROTECTED_TEST}) > 1 or (
            split_roles & {Role.TEST, Role.PROTECTED_TEST}
            and split_roles & {Role.TRAIN, Role.VALIDATION}
        ):
            error(
                "CONTRADICTORY_HISTORICAL_ROLES",
                tuple(r.id for r in registry.records if graph.photo.find(r.id) == key),
                "Conflicting source roles",
            )
    pending, queue_issues = review_queue(registry, graph, verifier)
    issues.extend(queue_issues)
    for row in registry.records:
        if row.historical_role == Role.PROTECTED_TEST:
            protected_photos.add(graph.photo.find(row.id))
    protected_dependencies = {
        graph.dependency.find(row.id)
        for row in registry.records
        if graph.photo.find(row.id) in protected_photos
    }
    protected_content = {
        graph.content.find(row.id)
        for row in registry.records
        if row.asset is not None
        and (
            row.historical_role == Role.PROTECTED_TEST
            or row.asset.sha256 in policy.protected_assets
        )
    }
    for assignment in manifest.assignments:
        row = records.get(assignment.record)
        if row is None:
            continue
        role = assignment.role
        if row.asset is not None:
            try:
                if not verifier.verify(row.asset):
                    raise ValueError("Image bytes changed")
            except (OSError, ValueError) as exc:
                error("IMAGE_INTEGRITY", (row.id,), str(exc), row.evidence)
        if role == Role.QUARANTINE:
            continue
        if role == Role.DIAGNOSTIC:
            if graph.photo.find(row.id) in protected_photos:
                error(
                    "HISTORICAL_ROLE",
                    (row.id,),
                    "Protected test remains protected_test",
                )
            continue
        photo = graph.photo.find(row.id)
        if (
            graph.dependency.find(row.id) in protected_dependencies
            and role != Role.PROTECTED_TEST
        ):
            error(
                "PROTECTED_SOURCE_DEPENDENCY",
                (row.id,),
                "Connected protected photo/family/site cannot enter another split",
            )
        ancestors = [row, *(records[key] for key in graph.ancestors[row.id])]
        for ancestor in ancestors:
            if ancestor.asset is None:
                error(
                    "MISSING_SOURCE_BYTES",
                    (row.id,),
                    f"Unmaterialized original/parent {ancestor.id}",
                )
            else:
                try:
                    if not verifier.verify(ancestor.asset):
                        raise ValueError(f"Changed ancestor {ancestor.id}")
                except (OSError, ValueError) as exc:
                    error("IMAGE_INTEGRITY", (row.id,), str(exc))
        if photo in protected_photos or any(
            graph.content.find(parent.id) in protected_content for parent in ancestors
        ):
            if role != Role.PROTECTED_TEST:
                error(
                    "PROTECTED_TEST_DERIVATIVE",
                    (row.id,),
                    "Photo or descendant belongs to protected test",
                )
        for historical in locks.get(photo, []):
            if historical in SUPERVISED and role != historical:
                error("HISTORICAL_ROLE", (row.id,), f"Source role is {historical}")
        if any(
            role in {Role.DIAGNOSTIC, Role.QUARANTINE}
            for r in ancestors
            for role in direct_locks.get(r.id, [])
        ):
            error(
                "DIAGNOSTIC_PROMOTION",
                (row.id,),
                "Trusted diagnostic/visibility exclusion is inherited",
            )
        # Restrictions flow to descendants; a diagnostic crop must not change
        # an independently authorised parent image's historical training role.
        inherited_sites = {
            r.site.value for r in ancestors if r.site.state == Knowledge.KNOWN
        }
        for site in inherited_sites:
            if site in policy.protected_sites and role != Role.PROTECTED_TEST:
                error(
                    "PROTECTED_PHYSICAL_SITE",
                    (row.id,),
                    "The entire parking site is protected",
                    row.site.evidence,
                )
            if site in policy.validation_only_sites and role != Role.VALIDATION:
                error(
                    "VALIDATION_ONLY_SITE",
                    (row.id,),
                    "Cameras and days share historical validation location",
                    row.site.evidence,
                )
        # Protected/diagnostic storage is not a new learning approval. Unknown
        # metadata remains explicit there; it still participates in split checks.
        if role == Role.PROTECTED_TEST:
            if (
                row.historical_role != Role.PROTECTED_TEST
                or Role.PROTECTED_TEST not in direct_locks.get(row.id, [])
            ):
                error(
                    "UNAUTHORIZED_PROTECTED_ROLE",
                    (row.id,),
                    "Only trusted historical protected records may use this role",
                )
            continue
        if any(
            r.provenance != Verification.CONFIRMED
            or not confirmed(graph.evidence, r.evidence)
            for r in ancestors
        ):
            error(
                "UNCONFIRMED_PROVENANCE",
                (row.id,),
                "Original-source identity has not been confirmed",
                row.evidence,
            )
        if row.site.state != Knowledge.KNOWN or not confirmed(
            graph.evidence, row.site.evidence
        ):
            error(
                "UNKNOWN_PHYSICAL_SITE",
                (row.id,),
                "A filename/session hint does not establish independent location",
            )
        if row.family.state != Knowledge.KNOWN or not confirmed(
            graph.evidence, row.family.evidence
        ):
            error("UNKNOWN_IMAGE_FAMILY", (row.id,), "Missing original image family")
        if any(not r.training_eligible for r in ancestors):
            error(
                "NOT_TRAINING_ELIGIBLE",
                (row.id,),
                "Labels do not override the historical eligibility exclusion",
            )
        if any(
            r.historical_role in {Role.DIAGNOSTIC, Role.QUARANTINE} for r in ancestors
        ):
            error(
                "DIAGNOSTIC_PROMOTION",
                (row.id,),
                "Historical diagnostic/quarantine record cannot be promoted",
            )
        if any(r.geometry_excluded for r in ancestors):
            error(
                "GEOMETRY_EXCLUDED",
                (row.id,),
                "Existing geometry exclusion remains binding",
            )
        sources = [datasets.get(key) for key in sorted({r.dataset for r in ancestors})]
        if any(source is None or source.version is None for source in sources):
            error(
                "UNKNOWN_DATASET_VERSION",
                (row.id,),
                "Source version has not been recorded",
            )
        if any(
            source is None
            or not source.license.local_research_authorized
            or source.license.verification != Verification.CONFIRMED
            or not confirmed(graph.evidence, source.license.evidence)
            for source in sources
        ):
            error(
                "LICENSE_SCOPE_UNVERIFIED",
                (row.id,),
                "No confirmed authorization for the requested local research use",
            )
        for candidate_id in pending.get(photo, []):
            error(
                "SIMILARITY_REVIEW_REQUIRED",
                (row.id,),
                f"Pending {candidate_id}; no duplicate/site relation inferred",
            )
        task = tasks.get(assignment.task)
        if task is None:
            error(
                "MISSING_TASK_CONTRACT",
                (row.id,),
                "Supervised data requires an explicit ML contract",
            )
            continue
        issues.extend(
            check_supervision(
                row, task, approvals.get(task.approval), graph.evidence, verifier
            )
        )

    issues.extend(check_splits(registry, manifest, graph, assigned, records))
    global_failure = any(not issue.records for issue in issues) or bool(structural)
    reasons: dict[str, set[str]] = defaultdict(set)
    for issue in issues:
        for key in issue.records:
            reasons[key].add(issue.rule)
    decisions = tuple(
        Decision(
            a.record,
            a.role,
            Role.QUARANTINE if global_failure or reasons[a.record] else a.role,
            not (global_failure or reasons[a.record]),
            tuple(
                sorted(
                    reasons[a.record]
                    | ({"REGISTRY_INVALID"} if global_failure else set())
                    | ({"RETAINED_QUARANTINE"} if a.role == Role.QUARANTINE else set())
                )
            ),
        )
        for a in manifest.assignments
    )
    return ValidationResult(
        not issues, decisions, tuple(issues), fingerprint(registry), fingerprint(policy)
    )
