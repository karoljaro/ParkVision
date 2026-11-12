import pytest

from parkvision.detection.line_fitting import fit_group_line
from parkvision.detection.line_segment import LineSegment
from parkvision.detection.marking_fragments import pair_marking_fragments
from parkvision.detection.segment_group import SegmentGroup


@pytest.mark.parametrize(("second_start", "expected_pairs"), [(75, 1), (95, 2)])
def test_marks_are_joined_only_across_a_local_gap(
    second_start: int, expected_pairs: int
) -> None:
    raw = tuple(
        LineSegment(index, start, end)
        for index, (start, end) in enumerate(
            (
                ((10.0, 10.0), (60.0, 10.0)),
                ((10.0, 15.0), (60.0, 15.0)),
                ((float(second_start), 10.0), (145.0, 10.0)),
                ((float(second_start), 15.0), (145.0, 15.0)),
            )
        )
    )
    groups = {segment.segment_id + 4: SegmentGroup((segment,)) for segment in raw}
    fitted = tuple(
        fit_group_line(group, segment_id=key) for key, group in groups.items()
    )

    final_edges, final_sources, pairs = pair_marking_fragments(fitted, groups, 0.5)

    assert len(pairs) == expected_pairs
    assert {edge.segment_id for edge in final_edges} == set(final_sources)
    assert {s for group in final_sources.values() for s in group.segments} == set(raw)
    assert len({p.centerline.segment_id for p in pairs}) == len(pairs)
    assert set(final_sources).isdisjoint(p.centerline.segment_id for p in pairs)
    if expected_pairs == 1:
        assert pairs[0].centerline.length == pytest.approx(135.0)
        assert {len(group.segments) for group in final_sources.values()} == {2}
