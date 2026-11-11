"""Pure, camera-local aggregation of observations, not parking-event detection.

All clocks are supplied by the caller. No geometry, model scores, IO or wall clock
is consulted. A confirmed estimate only means repeated occupancy evidence.
"""

from dataclasses import dataclass, replace
from datetime import datetime
from math import isfinite
from typing import Literal

Occupancy = Literal["occupied", "free", "unknown"]
Status = Literal["confirmed", "held", "unknown", "stale"]
Motion = Literal["unassessed", "moving", "stationary"]


def _text(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be a nonempty string without edge whitespace")


def _time(value: datetime, name: str) -> None:
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise ValueError(f"{name} must include a timezone")


@dataclass(frozen=True, slots=True)
class StreamKey:
    parking_id: str
    camera_id: str
    slot_id: str
    configuration_revision: int
    configuration_sha256: str
    session_id: str

    def __post_init__(self) -> None:
        for name in ("parking_id", "camera_id", "slot_id", "session_id"):
            _text(getattr(self, name), name)
        if (
            type(self.configuration_revision) is not int
            or self.configuration_revision < 1
        ):
            raise ValueError("configuration_revision must be a positive integer")
        digest = self.configuration_sha256
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise ValueError("configuration_sha256 must be a lowercase SHA-256")


@dataclass(frozen=True, slots=True)
class TemporalPolicy:
    """Starting heuristics for roughly 1 Hz input; no empirical calibration."""

    occupied_confirmations: int = 3
    free_confirmations: int = 3
    occupied_min_seconds: float = 2.0
    free_min_seconds: float = 2.0
    max_observation_age_seconds: float = 10.0
    max_gap_seconds: float = 5.0
    unknown_hold_seconds: float = 3.0

    def __post_init__(self) -> None:
        for name in ("occupied_confirmations", "free_confirmations"):
            value = getattr(self, name)
            if type(value) is not int or value < 2:
                raise ValueError(f"{name} must be an integer >= 2")
        for name in (
            "occupied_min_seconds",
            "free_min_seconds",
            "max_observation_age_seconds",
            "max_gap_seconds",
            "unknown_hold_seconds",
        ):
            value = getattr(self, name)
            if (
                type(value) not in (int, float)
                or not isfinite(value)
                or value < 0
                or (
                    name in {"max_observation_age_seconds", "max_gap_seconds"}
                    and value == 0
                )
            ):
                raise ValueError(f"Invalid duration: {name}")


@dataclass(frozen=True, slots=True)
class SlotObservation:
    """Raw per-frame decision. source_id links back to immutable provenance.

    Motion and visibility are external declarations, never inferred here.
    'unassessed' preserves the offline baseline's visibility limitation.
    """

    key: StreamKey
    sequence: int
    captured_at: datetime | None
    evaluated_at: datetime
    state: Occupancy
    unknown_reason: str | None
    source_id: str
    image_sha256: str | None
    view_confirmed: bool
    frame_status: str = "observed"
    visibility: str = "unassessed"
    motion: Motion = "unassessed"

    def __post_init__(self) -> None:
        if type(self.sequence) is not int or self.sequence < 0:
            raise ValueError("sequence must be a nonnegative integer")
        if self.captured_at is not None:
            _time(self.captured_at, "captured_at")
        _time(self.evaluated_at, "evaluated_at")
        if self.state not in {"occupied", "free", "unknown"}:
            raise ValueError("Invalid raw occupancy state")
        if self.state == "unknown":
            _text(self.unknown_reason, "unknown_reason")
        elif self.unknown_reason is not None:
            raise ValueError("A decisive raw observation cannot have unknown_reason")
        if type(self.view_confirmed) is not bool:
            raise ValueError("view_confirmed must be a boolean")
        _text(self.source_id, "source_id")
        _text(self.frame_status, "frame_status")
        if self.visibility not in {
            "unassessed",
            "clear",
            "partial",
            "unreadable",
            "operator_unreadable",
        }:
            raise ValueError("Unsupported visibility declaration")
        if self.motion not in {"unassessed", "moving", "stationary"}:
            raise ValueError("Unsupported motion declaration")


@dataclass(frozen=True, slots=True)
class Candidate:
    state: Occupancy
    count: int
    since: datetime


@dataclass(frozen=True, slots=True)
class TemporalState:
    key: StreamKey
    updated_at: datetime
    estimated_state: Occupancy = "unknown"
    status: Status = "unknown"
    reason: str = "no_observations"
    state_changed_at: datetime | None = None
    last_confirmation_at: datetime | None = None
    last_observation: SlotObservation | None = None
    last_evidence: SlotObservation | None = None
    supporting_evidence: SlotObservation | None = None
    last_capture_at: datetime | None = None
    candidate: Candidate | None = None
    hold_since: datetime | None = None


@dataclass(frozen=True, slots=True)
class Update:
    state: TemporalState
    disposition: Literal["accepted", "duplicate", "out_of_order"]


def initial_state(key: StreamKey, at: datetime) -> TemporalState:
    _time(at, "at")
    return TemporalState(key=key, updated_at=at)


def _clock(state: TemporalState, at: datetime) -> None:
    _time(at, "at")
    if at < state.updated_at:
        raise ValueError("Processing clock must be chronological")


def _unknown(
    state: TemporalState,
    at: datetime,
    reason: str,
    *,
    stale: bool = False,
    clear_candidate: bool = True,
) -> TemporalState:
    return replace(
        state,
        updated_at=at,
        estimated_state="unknown",
        status="stale" if stale else "unknown",
        reason=reason,
        state_changed_at=(
            at if state.estimated_state != "unknown" else state.state_changed_at
        ),
        candidate=None if clear_candidate else state.candidate,
        hold_since=None,
    )


def reset_stream(state: TemporalState, key: StreamKey, at: datetime) -> TemporalState:
    """Explicit session/revision boundary; never carry evidence across it."""
    _clock(state, at)
    if (key.parking_id, key.camera_id, key.slot_id) != (
        state.key.parking_id,
        state.key.camera_id,
        state.key.slot_id,
    ):
        raise ValueError("Cannot reset into another parking/camera/slot")
    if key == state.key:
        raise ValueError("Reset requires a new session or configuration identity")
    return replace(initial_state(key, at), reason="stream_reset")


def disconnect(state: TemporalState, at: datetime) -> TemporalState:
    """A known disconnection invalidates the estimate immediately."""
    _clock(state, at)
    return _unknown(state, at, "camera_disconnected")


def _expire(
    state: TemporalState, at: datetime, policy: TemporalPolicy
) -> TemporalState:
    """Advance freshness even when the camera sends nothing.

    Deadlines are exclusive: at the deadline the evidence is expired. Historical
    evidence stays available for audit, but cannot resurrect an expired estimate.
    """
    _clock(state, at)
    state = replace(state, updated_at=at)
    if (
        state.last_capture_at is not None
        and (at - state.last_capture_at).total_seconds() >= policy.max_gap_seconds
    ):
        return _unknown(state, at, "observation_gap", stale=True)
    if state.estimated_state != "unknown":
        support = state.supporting_evidence
        if (at - support.captured_at).total_seconds() >= (
            policy.max_observation_age_seconds
        ):
            return _unknown(state, at, "evidence_expired", stale=True)
        if (
            state.hold_since is not None
            and (at - state.hold_since).total_seconds() >= policy.unknown_hold_seconds
        ):
            return _unknown(state, at, "hold_expired", clear_candidate=False)
    elif state.candidate is not None and state.last_evidence is not None:
        if (at - state.last_evidence.captured_at).total_seconds() >= (
            policy.max_observation_age_seconds
        ):
            return _unknown(state, at, "evidence_expired", stale=True)
    return state


def tick(state: TemporalState, at: datetime, policy: TemporalPolicy) -> TemporalState:
    """A heartbeat with no new observation exposes held state and expires it.

    Silence TTL starts at the last capture, not at a delayed heartbeat. Callers
    must schedule heartbeats: a pure state machine has no background timer.
    """
    state = _expire(state, at, policy)
    if (
        state.estimated_state != "unknown"
        and state.status == "confirmed"
        and at > state.last_capture_at
    ):
        state = replace(
            state,
            status="held",
            reason="no_observation",
            hold_since=state.last_capture_at,
        )
        state = _expire(state, at, policy)
    return state


def _same_frame(a: SlotObservation, b: SlotObservation) -> bool:
    # Re-evaluation / copying the same frame cannot supply extra confirmations.
    return replace(a, source_id=b.source_id, evaluated_at=b.evaluated_at) == b


def observe_slot(
    state: TemporalState,
    observation: SlotObservation,
    at: datetime,
    policy: TemporalPolicy,
) -> Update:
    """Consume one arrival; reject mixed streams, ignore late/duplicate frames.

    Sequences and capture times must both strictly increase within a session.
    Reusing a sequence with conflicting content is an input error. A new session
    or revision requires reset_stream(), including after a sequence restart.
    """
    if observation.key != state.key:
        raise ValueError("Mixed stream identity; explicitly reset session/revision")
    state = _expire(state, at, policy)
    previous = state.last_observation
    if previous is not None:
        if observation.sequence == previous.sequence:
            if not _same_frame(observation, previous):
                raise ValueError("Sequence reused with conflicting observation")
            return Update(tick(state, at, policy), "duplicate")
        if observation.sequence < previous.sequence:
            return Update(tick(state, at, policy), "out_of_order")
    captured = observation.captured_at
    if captured is not None and state.last_capture_at is not None:
        if captured <= state.last_capture_at:
            return Update(tick(state, at, policy), "out_of_order")

    state = replace(state, last_observation=observation)
    hard_reason = None
    stale = False
    if not observation.view_confirmed:
        hard_reason = "view_unverified"
    elif observation.frame_status in {
        "dimension_mismatch",
        "view_unverified",
        "missing_capture_time",
        "future_frame",
        "stale_frame",
    }:
        hard_reason = observation.frame_status
        stale = hard_reason == "stale_frame"
    elif captured is None:
        hard_reason = "missing_capture_time"
    elif (
        captured > at
        or observation.evaluated_at > at
        or captured > observation.evaluated_at
    ):
        hard_reason = "future_frame"
    elif (at - captured).total_seconds() >= policy.max_observation_age_seconds:
        hard_reason, stale = "stale_frame", True
    if hard_reason is not None:
        return Update(_unknown(state, at, hard_reason, stale=stale), "accepted")
    state = replace(state, last_capture_at=captured)

    raw_state = observation.state
    reason = observation.unknown_reason
    if observation.frame_status != "observed":
        raw_state, reason = "unknown", observation.frame_status
    elif observation.visibility in {"operator_unreadable", "unreadable", "partial"}:
        raw_state, reason = "unknown", "occluded"
    elif observation.motion == "moving":
        raw_state, reason = "unknown", "motion_detected"

    if raw_state == "unknown":
        state = replace(state, candidate=None)
    else:
        state = replace(state, last_evidence=observation)
        if state.estimated_state == raw_state:
            return Update(
                replace(
                    state,
                    status="confirmed",
                    reason="repeated_evidence",
                    supporting_evidence=observation,
                    candidate=None,
                    hold_since=None,
                ),
                "accepted",
            )
        old = state.candidate
        candidate = (
            Candidate(raw_state, old.count + 1, old.since)
            if old is not None and old.state == raw_state
            else Candidate(raw_state, 1, captured)
        )
        state = replace(state, candidate=candidate)
        count = getattr(policy, f"{raw_state}_confirmations")
        seconds = getattr(policy, f"{raw_state}_min_seconds")
        if (
            candidate.count >= count
            and (captured - candidate.since).total_seconds() >= seconds
        ):
            return Update(
                replace(
                    state,
                    estimated_state=raw_state,
                    status="confirmed",
                    reason="repeated_evidence",
                    state_changed_at=at,
                    last_confirmation_at=at,
                    supporting_evidence=observation,
                    candidate=None,
                    hold_since=None,
                ),
                "accepted",
            )
        reason = "pending_confirmation"

    if state.estimated_state != "unknown":
        state = replace(
            state,
            status="held",
            reason=reason,
            hold_since=state.hold_since or captured,
        )
        state = _expire(state, at, policy)
    else:
        state = replace(state, status="unknown", reason=reason)
    return Update(state, "accepted")
