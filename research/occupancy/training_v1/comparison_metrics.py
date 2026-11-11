"""Fixed-threshold paired evidence metrics; no calibration or threshold search."""

from collections import Counter
from math import isfinite, log

STATES = ("free", "occupied", "unknown")


def summarize(rows, field, truth="source_state"):
    counts = Counter()
    for row in rows:
        target, prediction = row[truth], row[field]
        if target not in STATES or prediction not in STATES:
            raise ValueError("Invalid evaluation state")
        counts[target, prediction] += 1
    matrix = {
        target: {state: counts[target, state] for state in STATES} for target in STATES
    }
    free = sum(matrix["free"].values())
    occupied = sum(matrix["occupied"].values())
    known = free + occupied
    n = len(rows)
    decided = sum(row[field] != "unknown" for row in rows)
    correct_known = matrix["free"]["free"] + matrix["occupied"]["occupied"]
    false_free, false_occupied = matrix["occupied"]["free"], matrix["free"]["occupied"]
    return {
        "n": n,
        "source_free": free,
        "source_occupied": occupied,
        "source_unknown": n - known,
        "matrix": matrix,
        "correct_known": correct_known,
        "false_free": false_free,
        "false_occupied": false_occupied,
        "free_recall": matrix["free"]["free"] / free if free else None,
        "occupied_recall": (
            matrix["occupied"]["occupied"] / occupied if occupied else None
        ),
        "free_precision_known": (
            matrix["free"]["free"] / (matrix["free"]["free"] + false_free)
            if matrix["free"]["free"] + false_free
            else None
        ),
        "occupied_precision_known": (
            matrix["occupied"]["occupied"]
            / (matrix["occupied"]["occupied"] + false_occupied)
            if matrix["occupied"]["occupied"] + false_occupied
            else None
        ),
        "agreement_on_all_known": correct_known / known if known else None,
        "unknown": n - decided,
        "coverage": decided / n if n else None,
        "unsupported_decisions_on_unknown_truth": matrix["unknown"]["free"]
        + matrix["unknown"]["occupied"],
        "correct_unknown": matrix["unknown"]["unknown"],
    }


def probability_metrics(rows, field):
    loss = brier = correct = n = 0
    for row in rows:
        if row["source_state"] == "unknown":
            continue
        probability = row[field]
        if not isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Invalid score")
        target = int(row["source_state"] == "occupied")
        clipped = min(1 - 1e-7, max(1e-7, probability))
        loss -= target * log(clipped) + (1 - target) * log(1 - clipped)
        brier += (probability - target) ** 2
        correct += int((probability >= 0.5) == bool(target))
        n += 1
    return {
        "n": n,
        "accuracy_at_0_5": correct / n if n else None,
        "cross_entropy": loss / n if n else None,
        "brier": brier / n if n else None,
    }


def reason_summary(rows, reason_field):
    reasons = Counter(row[reason_field] or "decided" for row in rows)
    return {
        "counts": dict(reasons),
        "free_evidence_blocked_by_conflict": sum(
            row[reason_field] == "model_conflict" and row["source_state"] == "free"
            for row in rows
        ),
        "false_free_prevented_by_conflict": sum(
            row[reason_field] == "model_conflict" and row["source_state"] == "occupied"
            for row in rows
        ),
    }


def paired_changes(rows, before, after):
    transitions = Counter()
    for row in rows:
        target = row["source_state"]

        def outcome(state):
            if state == "unknown":
                return "abstain"
            return "correct" if state == target else "wrong"

        transitions[outcome(row[before]), outcome(row[after])] += 1
    return {a + "->" + b: count for (a, b), count in sorted(transitions.items())}


def visibility_gate(state, visibility):
    if state not in STATES or visibility not in {"clear", "partial", "unreadable"}:
        raise ValueError("Invalid manual state or visibility")
    return state if visibility == "clear" else "unknown"


def methods_summary(rows):
    result = {}
    for model in ("baseline", "candidate"):
        result[model] = {
            "argmax": summarize(rows, model + "_argmax"),
            "score_bands": summarize(rows, model + "_raw"),
            "D_fused": summarize(rows, model + "_D"),
            "C_fused_diagnostic": summarize(rows, model + "_C"),
            "D_reasons": reason_summary(rows, model + "_D_reason"),
            "C_reasons": reason_summary(rows, model + "_C_reason"),
            "probability_diagnostics": probability_metrics(rows, model + "_score"),
        }
    result["paired_D_transitions"] = paired_changes(rows, "baseline_D", "candidate_D")
    result["paired_score_transitions"] = paired_changes(
        rows, "baseline_raw", "candidate_raw"
    )
    result["shared_detector"] = {
        rule: summarize(rows, "detector_" + rule) for rule in ("C", "D")
    }
    return result
