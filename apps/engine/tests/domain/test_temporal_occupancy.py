from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone

import pytest

from parkvision.domain.temporal_occupancy import (
    SlotObservation,
    StreamKey,
    TemporalPolicy,
    disconnect,
    initial_state,
    observe_slot,
    reset_stream,
    tick,
)

BASE = datetime(2026, 11, 5, 12, tzinfo=timezone.utc)
KEY = StreamKey("parking", "camera", "slot", 1, "a" * 64, "session-1")
POLICY = TemporalPolicy()


def at(seconds):
    return BASE + timedelta(seconds=seconds)


def observation(seconds, state="occupied", **kwargs):
    return SlotObservation(
        key=kwargs.pop("key", KEY),
        sequence=kwargs.pop("sequence", int(seconds)),
        captured_at=kwargs.pop("captured_at", at(seconds)),
        evaluated_at=kwargs.pop("evaluated_at", at(seconds)),
        state=state,
        unknown_reason=kwargs.pop(
            "unknown_reason", "low_confidence" if state == "unknown" else None
        ),
        source_id=kwargs.pop("source_id", f"source-{seconds}"),
        image_sha256=kwargs.pop("image_sha256", "b" * 64),
        view_confirmed=kwargs.pop("view_confirmed", True),
        **kwargs,
    )


def consume(state, seconds, raw="occupied", *, policy=POLICY, now=None, **kwargs):
    return observe_slot(
        state,
        observation(seconds, raw, **kwargs),
        at(seconds if now is None else now),
        policy,
    ).state


def confirmed(raw="occupied", policy=POLICY):
    state = initial_state(KEY, BASE)
    for seconds in (0, 1, 2):
        state = consume(state, seconds, raw, policy=policy)
    assert state.estimated_state == raw
    return state


@pytest.mark.parametrize("raw", ["occupied", "free"])
def test_single_frame_is_only_raw_evidence_and_three_confirmations_are_required(raw):
    state = initial_state(KEY, BASE)
    first = consume(state, 0, raw)
    assert state.last_observation is None
    assert first.estimated_state == "unknown"
    assert first.last_observation.state == first.last_evidence.state == raw
    assert first.supporting_evidence is None
    assert first.candidate.count == 1
    second = consume(first, 1, raw)
    assert second.estimated_state == "unknown"
    third = consume(second, 2, raw)
    assert third.estimated_state == raw
    assert third.status == "confirmed"
    assert third.last_confirmation_at == third.state_changed_at == at(2)
    assert state == initial_state(KEY, BASE)
    with pytest.raises(FrozenInstanceError):
        third.status = "unknown"


def test_burst_of_frames_does_not_meet_minimum_time_span():
    state = initial_state(KEY, BASE)
    for seq, seconds in enumerate((0, 0.01, 0.02)):
        state = consume(state, seconds, sequence=seq)
    assert state.estimated_state == "unknown" and state.candidate.count == 3
    state = consume(state, 2, sequence=3)
    assert state.estimated_state == "occupied"


@pytest.mark.parametrize("first,opposite", [("occupied", "free"), ("free", "occupied")])
def test_isolated_fp_fn_and_transition_preserve_raw_and_temporal_states(
    first, opposite
):
    state = confirmed(first)
    original_confirmation = state.last_confirmation_at
    noisy = consume(state, 3, opposite)
    assert noisy.estimated_state == first and noisy.status == "held"
    assert noisy.last_observation.state == noisy.last_evidence.state == opposite
    assert noisy.supporting_evidence.state == first
    assert noisy.last_confirmation_at == original_confirmation
    recovered = consume(noisy, 4, first)
    assert recovered.estimated_state == first and recovered.status == "confirmed"
    assert recovered.candidate is recovered.hold_since is None
    for seconds in (5, 6):
        recovered = consume(recovered, seconds, opposite)
        assert recovered.estimated_state == first and recovered.status == "held"
    transitioned = consume(recovered, 7, opposite)
    assert transitioned.estimated_state == opposite
    assert transitioned.status == "confirmed"
    assert transitioned.state_changed_at == at(7)


@pytest.mark.parametrize(
    "reason",
    [
        "occluded",
        "model_conflict",
        "low_confidence",
        "leaf_shadow",
        "edge_of_frame",
        "distant_slot",
        "partial_visibility",
        "missing_frame",
        "inference_error",
        "missing_slot_observation",
    ],
)
@pytest.mark.parametrize("raw", ["occupied", "free"])
def test_unknown_has_a_nonrenewable_hold_deadline_then_requires_reconfirmation(
    reason, raw
):
    state = confirmed(raw)
    support = state.supporting_evidence
    for seconds in (3, 4, 5):
        state = consume(state, seconds, "unknown", unknown_reason=reason)
        assert state.status == "held" and state.estimated_state == raw
        assert state.hold_since == at(3)
        assert state.supporting_evidence == state.last_evidence == support
    state = consume(state, 6, "unknown", unknown_reason=reason)
    assert state.estimated_state == "unknown" and state.status == "unknown"
    for seconds in (7, 8):
        state = consume(state, seconds, raw)
        assert state.estimated_state == "unknown"
    state = consume(state, 9, raw)
    assert state.estimated_state == raw and state.status == "confirmed"


def test_unknown_breaks_confirmation_run_and_never_counts_as_free():
    state = initial_state(KEY, BASE)
    for seconds, raw in enumerate(("free", "free", "unknown", "free", "free")):
        state = consume(state, seconds, raw)
        assert state.estimated_state == "unknown"
    state = consume(state, 5, "free")
    assert state.estimated_state == "free"
    unknown = initial_state(KEY, BASE)
    for seconds in range(20):
        unknown = consume(unknown, seconds, "unknown")
        assert unknown.estimated_state == "unknown"
        assert unknown.last_evidence is None


def test_opposing_evidence_cannot_refresh_support_age_and_candidate_can_outlive_hold():
    policy = replace(POLICY, free_confirmations=5, unknown_hold_seconds=1)
    state = confirmed(policy=policy)
    for seconds in range(3, 8):
        state = consume(state, seconds, "free", policy=policy)
        if seconds >= 4 and seconds < 7:
            assert state.estimated_state == "unknown"
    assert state.estimated_state == "free"


def test_duplicate_late_and_equal_capture_frames_do_not_add_evidence():
    first = observation(0)
    state = observe_slot(initial_state(KEY, BASE), first, BASE, POLICY).state
    duplicate = observe_slot(state, first, at(0.5), POLICY)
    assert duplicate.disposition == "duplicate"
    assert duplicate.state.candidate.count == 1
    same_pixels = replace(first, source_id="copy", evaluated_at=at(0.5))
    assert observe_slot(state, same_pixels, at(0.5), POLICY).disposition == "duplicate"
    state = consume(state, 2)
    for raw in (observation(1), observation(3, captured_at=at(2))):
        result = observe_slot(state, raw, at(3), POLICY)
        assert result.disposition == "out_of_order"
        assert result.state.candidate.count == 2
        assert result.state.last_observation == state.last_observation
    with pytest.raises(ValueError, match="Sequence reused"):
        observe_slot(state, observation(2, "free"), at(3), POLICY)


def test_duplicate_at_a_later_clock_cannot_keep_confirmed_state_alive():
    state = confirmed()
    result = observe_slot(state, state.last_observation, at(7), POLICY)
    assert result.disposition == "duplicate"
    assert result.state.estimated_state == "unknown"
    assert result.state.status == "stale"
    assert result.state.last_evidence == state.last_evidence


def test_silence_is_held_then_unknown_and_gap_discards_pending_run():
    state = confirmed()
    held = tick(state, at(3), POLICY)
    assert held.status == "held" and held.reason == "no_observation"
    assert held.hold_since == at(2)
    expired = tick(held, at(5), POLICY)
    assert expired.estimated_state == "unknown" and expired.reason == "hold_expired"
    stale = tick(expired, at(7), POLICY)
    assert stale.status == "stale" and stale.reason == "observation_gap"
    restarted = consume(stale, 7)
    assert restarted.estimated_state == "unknown" and restarted.candidate.count == 1
    pending = consume(initial_state(KEY, BASE), 0)
    pending = consume(pending, 5)
    assert pending.candidate.count == 1


def test_evidence_age_is_not_renewed_by_unknown_or_opposing_observations():
    policy = replace(POLICY, max_observation_age_seconds=4, unknown_hold_seconds=10)
    state = confirmed(policy=policy)
    for seconds in (3, 4, 5):
        state = consume(state, seconds, "unknown", policy=policy)
        assert state.status == "held"
    expired = tick(state, at(6), policy)
    assert expired.status == "stale" and expired.reason == "evidence_expired"
    assert expired.last_evidence.captured_at == at(2)


@pytest.mark.parametrize(
    "kwargs,now,reason,status",
    [
        ({"view_confirmed": False}, 3, "view_unverified", "unknown"),
        ({"frame_status": "dimension_mismatch"}, 3, "dimension_mismatch", "unknown"),
        ({"frame_status": "stale_frame"}, 3, "stale_frame", "stale"),
        ({"captured_at": None}, 3, "missing_capture_time", "unknown"),
        ({"captured_at": at(4)}, 3, "future_frame", "unknown"),
        ({"evaluated_at": at(4)}, 3, "future_frame", "unknown"),
        ({}, 13, "stale_frame", "stale"),
    ],
)
def test_invalid_time_or_view_immediately_invalidates_without_hold(
    kwargs, now, reason, status
):
    state = consume(confirmed(), 3, now=now, **kwargs)
    assert state.estimated_state == "unknown" and state.status == status
    assert state.reason == reason and state.candidate is None
    assert state.last_evidence.captured_at == at(2)


@pytest.mark.parametrize("visibility", ["partial", "unreadable", "operator_unreadable"])
def test_visibility_declaration_overrides_a_decisive_raw_state(visibility):
    state = initial_state(KEY, BASE)
    for seconds in range(5):
        state = consume(state, seconds, "free", visibility=visibility)
        assert state.estimated_state == "unknown"
        assert state.last_observation.state == "free"
        assert state.last_evidence is None


def test_moving_is_only_an_external_unknown_gate_stationary_is_not_a_parking_fact():
    state = initial_state(KEY, BASE)
    for seconds in range(5):
        state = consume(state, seconds, motion="moving")
        assert state.estimated_state == "unknown" and state.reason == "motion_detected"
    for seconds in range(5, 8):
        state = consume(state, seconds, motion="stationary")
    assert state.estimated_state == "occupied"


def test_disconnection_is_immediate_and_reconnection_needs_a_new_confirmation_run():
    state = confirmed()
    state = disconnect(state, at(2.5))
    assert state.estimated_state == "unknown" and state.reason == "camera_disconnected"
    for seconds in (3, 4):
        state = consume(state, seconds)
        assert state.estimated_state == "unknown"
    assert consume(state, 5).estimated_state == "occupied"


@pytest.mark.parametrize(
    "change",
    [
        {"parking_id": "other"},
        {"camera_id": "other"},
        {"slot_id": "other"},
        {"configuration_revision": 2},
        {"configuration_sha256": "c" * 64},
        {"session_id": "session-2"},
    ],
)
def test_mixed_identity_never_fuses(change):
    state = confirmed()
    with pytest.raises(ValueError, match="Mixed stream"):
        consume(state, 3, key=replace(KEY, **change))


@pytest.mark.parametrize(
    "key",
    [replace(KEY, session_id="session-2"), replace(KEY, configuration_revision=2)],
)
def test_explicit_reset_drops_evidence_and_allows_sequence_restart(
    key,
):
    state = reset_stream(confirmed(), key, at(3))
    assert state.estimated_state == "unknown" and state.last_evidence is None
    assert state.last_observation is state.supporting_evidence is None
    state = consume(state, 3, key=key, sequence=0)
    assert state.candidate.count == 1 and state.estimated_state == "unknown"
    with pytest.raises(ValueError, match="Mixed stream"):
        consume(state, 4, key=KEY, sequence=100)
    with pytest.raises(ValueError, match="another parking/camera/slot"):
        reset_stream(state, replace(key, camera_id="other"), at(4))
    with pytest.raises(ValueError, match="requires a new"):
        reset_stream(state, key, at(4))


def test_processing_clock_must_be_aware_and_monotonic():
    state = confirmed()
    for operation in (
        lambda: tick(state, at(1), POLICY),
        lambda: disconnect(state, at(1)),
        lambda: consume(state, 3, now=1),
    ):
        with pytest.raises(ValueError, match="chronological"):
            operation()
    with pytest.raises(ValueError, match="timezone"):
        initial_state(KEY, BASE.replace(tzinfo=None))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"occupied_confirmations": 1},
        {"free_confirmations": True},
        {"max_gap_seconds": 0},
        {"max_observation_age_seconds": -1},
        {"unknown_hold_seconds": float("nan")},
        {"free_min_seconds": float("inf")},
    ],
)
def test_policy_rejects_invalid_or_single_frame_configuration(kwargs):
    with pytest.raises(ValueError):
        TemporalPolicy(**kwargs)


def test_configurable_asymmetry_and_zero_hold():
    policy = replace(POLICY, free_confirmations=4, unknown_hold_seconds=0)
    state = confirmed(policy=policy)
    state = consume(state, 3, "unknown", policy=policy)
    assert state.estimated_state == "unknown"
    for seconds in (4, 5, 6):
        state = consume(state, seconds, "free", policy=policy)
        assert state.estimated_state == "unknown"
    assert consume(state, 7, "free", policy=policy).estimated_state == "free"


def test_sequence_replay_is_deterministic_for_the_same_policy_and_clock():
    def run():
        state = initial_state(KEY, BASE)
        states = []
        for seconds, raw in enumerate(
            ["occupied"] * 3 + ["free", "occupied", "unknown"] + ["free"] * 3
        ):
            state = consume(state, seconds, raw)
            states.append(state)
        return states

    assert run() == run()
