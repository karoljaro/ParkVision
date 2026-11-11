"""Task semantics and reviewed class-coverage protocols."""

from .contracts import (
    Approval,
    Evidence,
    Issue,
    Meaning,
    Protocol,
    Record,
    Task,
    Verification,
)
from .graph import confirmed
from .storage import AssetVerifier


def check_supervision(
    row: Record,
    task: Task,
    approval: Approval | None,
    evidence: dict[str, Evidence],
    verifier: AssetVerifier,
) -> list[Issue]:
    issues: list[Issue] = []

    def error(
        rule: str, keys: tuple[str, ...], message: str, refs: tuple[str, ...] = ()
    ) -> None:
        issues.append(Issue(rule, keys, message, refs))

    if task.meaning == Meaning.FOOTPRINT and row.is_parking_slot is not True:
        error(
            "NOT_A_PARKING_SLOT",
            (row.id,),
            "T-01/unassessed region is not an approved physical slot",
        )
    matching = [a for a in row.annotations if a.meaning == task.meaning]
    covered: set[str] = set()
    verified = []
    for annotation in row.annotations:
        if annotation.binding is None:
            continue
        try:
            if not verifier.verify(annotation.binding):
                raise ValueError("Annotation bytes changed")
        except (OSError, ValueError) as exc:
            error("ANNOTATION_INTEGRITY", (row.id,), str(exc), annotation.evidence)
            continue
        verified.append(annotation)
        if (
            annotation.meaning == task.meaning
            and annotation.coverage == Verification.CONFIRMED
            and confirmed(evidence, annotation.evidence)
        ):
            covered.update(annotation.exhaustive_classes)
    approved = bool(
        approval is not None
        and row.id in approval.records
        and approval.target == task.meaning
        and set(task.classes) <= set(approval.classes)
        and any(a.meaning == approval.source for a in verified)
    )
    if approved:
        try:
            approved = verifier.verify(approval.artifact) and verifier.verify(
                approval.implementation
            )
        except (OSError, ValueError):
            approved = False
    if task.approval is not None and not approved:
        error(
            "INVALID_TASK_APPROVAL",
            (row.id,),
            "Approval scope/classes/meaning or artifact integrity mismatch",
        )
    if not matching and not (
        approved and approval.protocol == Protocol.SEMANTIC_TRANSFORM
    ):
        error(
            "ANNOTATION_SEMANTICS",
            (row.id,),
            f"Source does not provide {task.meaning}; no approved transform",
        )
    if set(task.classes) - covered and not approved:
        error(
            "INCOMPLETE_CLASS_COVERAGE",
            (row.id,),
            "No exhaustive/ignore/conditional coverage for "
            f"{sorted(set(task.classes) - covered)}",
        )
    return issues
