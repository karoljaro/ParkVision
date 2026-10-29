from dataclasses import replace
from math import cos, sin

import pytest

from parkvision.detection.line_fitting import fit_group_line
from parkvision.detection.line_segment import LineSegment
from parkvision.detection.marking_pair import MarkingPair
from parkvision.detection.marking_pairing import pair_marking_edges
from parkvision.detection.marking_topology import find_topology_candidates
from parkvision.detection.segment_group import SegmentGroup
from parkvision.detection.topology_candidate import TopologyCandidates
from parkvision.detection.topology_support_measurement import measure_topology_support


def pipeline(groups):
    first_id = max(s.segment_id for group in groups for s in group.segments) + 1
    fitted = tuple(
        fit_group_line(g, segment_id=first_id + i) for i, g in enumerate(groups)
    )
    pairs = pair_marking_edges(fitted, min_width_px=2, max_width_px=12)
    paired = {edge.segment_id for pair in pairs for edge in pair.edges}
    context = tuple(line for line in fitted if line.segment_id not in paired)
    topology = find_topology_candidates(pairs, context, max_extension_px=6)
    return topology, pairs, dict(zip((line.segment_id for line in fitted), groups))


def cross_groups():
    return (
        SegmentGroup(
            (LineSegment(0, (-60, -3), (-6, -3)), LineSegment(1, (6, -3), (60, -3)))
        ),
        SegmentGroup(
            (LineSegment(2, (-60, 3), (-6, 3)), LineSegment(3, (6, 3), (60, 3)))
        ),
        SegmentGroup((LineSegment(4, (-3, -60), (-3, 60)),)),
        SegmentGroup((LineSegment(5, (3, -60), (3, 60)),)),
    )


def test_source_gap_is_measured_per_edge_without_filtering_the_X_or_sources():
    topology, pairs, groups = pipeline(cross_groups())
    before = repr((topology, pairs, groups))
    (evidence,) = measure_topology_support(topology, pairs, groups)
    assert topology.junctions[0].intersections[0].topology == "X"
    assert not topology.junctions[0].intersections[0].hypothetical
    assert (evidence.candidate_id, evidence.intersection_index) == (0, 0)
    horizontal, vertical = evidence.lines
    for edge in horizontal.edges:
        assert edge.nearest_fragment_gap_px == 6
        assert edge.junction.coverage_ratio == 0
        assert edge.junction.source_segment_ids == ()
        assert edge.before.covered_length_px == edge.after.covered_length_px == 9
        assert edge.before.coverage_ratio == edge.after.coverage_ratio == 0.75
    for edge in vertical.edges:
        assert edge.nearest_fragment_gap_px == 0
        assert edge.junction.coverage_ratio == 1
        assert edge.before.coverage_ratio == edge.after.coverage_ratio == 1
    assert repr((topology, pairs, groups)) == before


def test_extrapolated_curb_keeps_two_edges_separate_and_sources_beyond_axis_extent():
    groups = (
        SegmentGroup((LineSegment(0, (0, -3), (60, -3)),)),
        SegmentGroup((LineSegment(1, (0, 3), (70, 3)),)),
        SegmentGroup((LineSegment(2, (64, -60), (64, 60)),)),
        SegmentGroup((LineSegment(3, (200, 200), (201, 200)),)),
    )
    topology, pairs, sources = pipeline(groups)
    (evidence,) = measure_topology_support(topology, pairs, sources)
    hit = topology.junctions[0].intersections[0]
    assert hit.uses_unpaired_context and hit.hypothetical
    context, axis = evidence.lines  # Context IDs sort before centerlines.
    assert context.kind == "unpaired_context" and len(context.edges) == 1
    assert context.edges[0].junction.coverage_ratio == 1
    first, second = axis.edges
    assert first.nearest_fragment_gap_px == 4
    assert first.junction.coverage_ratio == 0 and first.after.coverage_ratio == 0
    assert first.before.coverage_ratio == pytest.approx(11 / 12)
    assert second.nearest_fragment_gap_px == 0
    assert second.junction.coverage_ratio == 1
    assert second.after.coverage_ratio == pytest.approx(3 / 12)
    assert len(topology.lines) == 3 and len(topology.endpoints) == 6
    assert any(not e.junction_ids for e in topology.endpoints)


def test_overlaps_count_once_and_only_positive_window_overlap_records_ids():
    groups = list(cross_groups())
    groups[0] = SegmentGroup(
        (
            LineSegment(0, (-60, -3), (60, -3)),
            LineSegment(1, (-10, -3), (10, -3)),
            LineSegment(6, (3, -3), (20, -3)),  # Touches J at +3; not J support.
        )
    )
    topology, pairs, sources = pipeline(tuple(groups))
    edge = measure_topology_support(topology, pairs, sources)[0].lines[0].edges[0]
    assert edge.junction.covered_length_px == 6
    assert edge.junction.coverage_ratio == 1
    assert edge.junction.source_segment_ids == (0, 1)
    assert edge.after.covered_length_px == 12
    assert edge.after.source_segment_ids == (0, 1, 6)


@pytest.mark.parametrize("angle", [0, 0.7, 2.1])
def test_rotation_translation_raw_endpoint_and_group_order_do_not_change_support(angle):
    original = cross_groups()
    topology, pairs, sources = pipeline(original)
    expected = measure_topology_support(topology, pairs, sources)

    def point(p):
        return (
            1000 + p[0] * cos(angle) - p[1] * sin(angle),
            -700 + p[0] * sin(angle) + p[1] * cos(angle),
        )

    changed = tuple(
        SegmentGroup(
            tuple(
                LineSegment(s.segment_id, point(s.end), point(s.start))
                for s in reversed(g.segments)
            )
        )
        for g in original
    )
    topology, pairs, sources = pipeline(changed)
    actual = measure_topology_support(
        topology, pairs, dict(reversed(list(sources.items())))
    )
    # Fit canonicalization may reverse the topology direction after rotation.
    for hit in actual:
        for line in hit.lines:
            for edge in line.edges:
                reference = next(
                    e
                    for expected_line in expected[0].lines
                    for e in expected_line.edges
                    if e.fitted_edge_id == edge.fitted_edge_id
                )
                assert edge.junction.coverage_ratio == pytest.approx(
                    reference.junction.coverage_ratio, abs=1e-10
                )
                assert edge.nearest_fragment_gap_px == pytest.approx(
                    reference.nearest_fragment_gap_px, abs=1e-10
                )
                assert edge.before.coverage_ratio == pytest.approx(
                    reference.before.coverage_ratio, abs=1e-10
                )
                assert edge.after.coverage_ratio == pytest.approx(
                    reference.after.coverage_ratio, abs=1e-10
                )


def test_reversing_topology_direction_exchanges_signed_arms_only():
    topology, pairs, sources = pipeline(
        (
            SegmentGroup((LineSegment(0, (0, -3), (60, -3)),)),
            SegmentGroup((LineSegment(1, (0, 3), (60, 3)),)),
            SegmentGroup((LineSegment(2, (64, -60), (64, 60)),)),
        )
    )
    before = measure_topology_support(topology, pairs, sources)[0].lines[1].edges
    pair = pairs[0]
    reversed_pair = replace(
        pair,
        centerline=replace(
            pair.centerline, start=pair.centerline.end, end=pair.centerline.start
        ),
    )
    context = tuple(
        line for line in topology.lines if line.segment_id in topology.unpaired_line_ids
    )
    changed = find_topology_candidates((reversed_pair,), context, max_extension_px=6)
    after = (
        measure_topology_support(changed, (reversed_pair,), sources)[0].lines[1].edges
    )
    for a, b in zip(before, after):
        assert a.junction.coverage_ratio == b.junction.coverage_ratio
        assert a.before.coverage_ratio == b.after.coverage_ratio
        assert a.after.coverage_ratio == b.before.coverage_ratio


def test_support_uses_exact_hits_in_a_cluster_and_configurable_windows():
    topology, pairs, sources = pipeline(cross_groups())
    normal = measure_topology_support(topology, pairs, sources)
    wider = measure_topology_support(
        topology, pairs, sources, junction_radius_px=8, arm_length_px=20
    )
    assert len(normal) == len(wider) == 1
    assert wider[0].lines[0].edges[0].junction.coverage_ratio == 0.25
    # A second vertical axis gives a real hit at x=2 in the same radius-3 node.
    edges = (LineSegment(102, (-1, -60), (-1, 60)), LineSegment(103, (5, -60), (5, 60)))
    second_pair = MarkingPair(edges, LineSegment(104, (2, -60), (2, 60)), (6, 6), 1)
    for edge, raw_id in zip(edges, (100, 101)):
        sources[edge.segment_id] = SegmentGroup((replace(edge, segment_id=raw_id),))
    pairs = (*pairs, second_pair)
    changed = find_topology_candidates(pairs)
    assert len(changed.junctions) == 1
    assert len(changed.junctions[0].intersections) == 2
    evidence = measure_topology_support(changed, pairs, sources)
    assert evidence[1].intersection_index == 1
    assert evidence[0].lines[0].edges[0].nearest_fragment_gap_px == 6
    assert evidence[1].lines[0].edges[0].nearest_fragment_gap_px == 4


@pytest.mark.parametrize(
    "radius,length",
    [(0, 15), (-1, 15), (3, 3), (4, 3), (float("nan"), 15), (3, float("inf"))],
)
def test_invalid_windows_are_rejected_even_for_empty_topology(radius, length):
    with pytest.raises(ValueError, match="finite"):
        measure_topology_support(
            TopologyCandidates((), (), (), ()),
            (),
            {},
            junction_radius_px=radius,
            arm_length_px=length,
        )


def test_empty_and_isolated_lines_have_no_support_records_but_require_provenance():
    assert measure_topology_support(TopologyCandidates((), (), (), ()), (), {}) == ()
    topology, pairs, sources = pipeline(cross_groups()[:2])
    assert not topology.junctions
    assert measure_topology_support(topology, pairs, sources) == ()
    with pytest.raises(ValueError, match="match all"):
        measure_topology_support(topology, pairs, {})


@pytest.mark.parametrize(
    "problem",
    ["missing", "extra", "empty", "duplicate", "invalid", "derived_id", "wrong_pair"],
)
def test_provenance_errors_do_not_silently_become_absent_support(problem):
    topology, pairs, sources = pipeline(cross_groups())
    key = next(iter(sources))
    if problem == "missing":
        sources.pop(key)
    if problem == "extra":
        sources[999] = sources[key]
    if problem == "empty":
        sources[key] = SegmentGroup(())
    if problem == "duplicate":
        sources[key] = SegmentGroup(
            (sources[key].segments[0], sources[key].segments[0])
        )
    if problem == "invalid":
        sources[key] = SegmentGroup((LineSegment(100, (float("nan"), 0), (1, 0)),))
    if problem == "derived_id":
        sources[key] = SegmentGroup(
            (replace(sources[key].segments[0], segment_id=key),)
        )
    if problem == "wrong_pair":
        pairs = ()
    with pytest.raises(ValueError):
        measure_topology_support(topology, pairs, sources)
