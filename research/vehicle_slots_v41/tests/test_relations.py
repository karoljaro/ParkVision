from dataclasses import replace
from pathlib import Path

import pytest

from research.datasets_v4.contracts import Asset, Fact, Role
from research.datasets_v4.storage import load, save
from research.vehicle_slots_v41.relations import Frame, Link, Queue, Relation, join
from research.vehicle_slots_v41.review import Review, apply, read_csv
from research.vehicle_slots_v41.source import FrameSource, read_bundle


def image(**changes):
    return dict(
        id=11,
        file_name="11.jpg",
        width=100,
        height=80,
        annotationsRectangle=[0, 0, 100, 80],
        **changes
    )


def row(identity, car, **changes):
    polygon = [10, 10, 25, 10, 25, 30, 10, 30]
    data = dict(
        id=identity,
        image_id=11,
        car_id=car,
        category_id=1,
        bbox=[10, 10, 15, 20],
        segmentation=[polygon],
        iscrowd=0,
    )
    return data | changes


def build(spots, cars):
    return join(FrameSource(image(), spots), FrameSource(image(), cars), "camera:a")


def test_exact_image_car_join_keeps_all_slots_and_no_labels():
    slots, cars, pairs, issues = build([row(21, 9), row(22, 9)], [row(31, 9)])
    assert not issues
    assert len(cars) == 1
    assert all(s.link == Link.MATCHED and s.car_annotation_ids == (31,) for s in slots)
    assert {p.spot_annotation_id for p in pairs} == {21, 22}
    assert all(
        p.relation == Relation.NOT_ASSESSED and p.reviewer is None for p in pairs
    )


@pytest.mark.parametrize(
    ("car_rows", "expected"),
    [
        ([], Link.MISSING_MASK),
        ([row(31, 2)], Link.MISSING_MASK),
        ([row(31, 9, segmentation=[])], Link.INVALID_MASK),
        ([row(31, 9, segmentation=[[10, 10, 200, 10, 10, 25]])], Link.INVALID_MASK),
        ([row(31, 9), row(32, 9)], Link.AMBIGUOUS),
    ],
)
def test_missing_invalid_and_duplicate_masks(car_rows, expected):
    slots, _, pairs, _ = build([row(21, 9)], car_rows)
    assert slots[0].link == expected
    assert all(p.relation == Relation.NOT_ASSESSED for p in pairs)
    if expected == Link.AMBIGUOUS:
        assert slots[0].car_annotation_ids == (31, 32)


@pytest.mark.parametrize(
    ("car_id", "category", "expected"),
    [
        (-1, 0, Link.NO_VEHICLE),
        (-2, 1, Link.OTHER_OBJECT),
        (-3, 1, Link.INVALID_REFERENCE),
        (-1, 1, Link.INVALID_REFERENCE),
        (-2, 0, Link.INVALID_REFERENCE),
        (None, 1, Link.INVALID_REFERENCE),
        (True, 1, Link.INVALID_REFERENCE),
    ],
)
def test_source_sentinels_do_not_generate_occupancy_or_relation(
    car_id, category, expected
):
    slots, _, pairs, _ = build([row(21, car_id, category_id=category)], [])
    assert slots[0].link == expected
    assert not pairs


def test_missing_cars_frame_and_reference_to_another_frame():
    a = FrameSource(image(), [row(21, 9)])
    slots, _, pairs, issues = join(a, None, "camera:a")
    assert slots[0].link == Link.MISSING_MASK and not pairs
    assert issues == ("cars_frame_missing",)
    with pytest.raises(ValueError, match="different image"):
        build([row(21, 9)], [row(31, 9, image_id=12)])
    b = FrameSource(image() | {"file_name": "other.jpg"}, [row(31, 9)])
    slots, _, pairs, issues = join(a, b, "camera:a")
    assert slots[0].link == Link.METADATA_CONFLICT and not pairs
    assert issues == ("cars_spots_image_metadata_conflict",)


def test_annotation_rectangle_keeps_partial_coverage_visible():
    a = FrameSource(image() | {"annotationsRectangle": [12, 12, 60, 60]}, [row(21, 9)])
    b = FrameSource(a.image, [row(31, 9)])
    slots, cars, pairs, _ = join(a, b, "camera:a")
    assert slots[0].inside_annotation_rectangle is False
    assert cars[0].inside_annotation_rectangle is False
    assert pairs[0].relation == Relation.NOT_ASSESSED


def test_slot_key_stable_for_reversed_or_rotated_vertices():
    a = row(21, 9)
    b = row(22, 9, segmentation=[[25, 10, 10, 10, 10, 30, 25, 30]])
    slots, *_ = build([a, b], [])
    assert slots[0].camera_slot_key == slots[1].camera_slot_key


def test_archive_pin_rejects_wrong_bytes(tmp_path):
    path = tmp_path / "ufpr04_cars.tar.xz"
    path.write_bytes(b"not original")
    with pytest.raises(ValueError, match="pinned original"):
        read_bundle(path, "cars", {11})


def test_operator_can_mark_one_vehicle_against_two_slots(tmp_path):
    slots, cars, pairs, _ = build(
        [row(21, 9), row(22, -1, category_id=0)], [row(31, 9)]
    )
    anchor = Asset(str(Path(tmp_path) / "test.jpg"), "0" * 64)
    frame = Frame(
        11,
        anchor,
        "11.jpg",
        100,
        80,
        "source:11",
        Role.TRAIN,
        Role.TRAIN,
        Role.DIAGNOSTIC,
        Fact(),
        Fact(),
        Fact(),
        Fact(),
        "2012-12-07T12:00:00",
        anchor,
        True,
        (0, 0, 100, 80),
        (),
        slots,
        cars,
        pairs,
    )
    queue = Queue(*("0" * 64,) * 4, (), (frame,))
    csv = tmp_path / "decisions.csv"
    csv.write_text(
        "image_id,car_annotation_id,spot_annotation_id,relation,reviewer,note\n"
        "11,31,21,owning_slot,operator,contact visible\n"
        "11,31,22,physically_blocks_neighbor,operator,wheels cross line\n"
    )
    reviewed = apply(queue, read_csv(csv))
    assert [p.relation for p in reviewed.frames[0].pairs] == [
        Relation.OWNING,
        Relation.BLOCKS,
    ]
    assert all(p.relation == Relation.NOT_ASSESSED for p in frame.pairs)
    assert reviewed.frames[0].historical_role == Role.TRAIN
    assert not reviewed.training_eligible and reviewed.parent_queue_sha256
    path = tmp_path / "review.json"
    save(path, reviewed)
    assert load(path, Queue) == reviewed
    with pytest.raises(ValueError, match="unavailable source geometry"):
        apply(queue, (Review(11, 999, 21, Relation.OWNING, "operator", "note"),))
    with pytest.raises(ValueError, match="remain diagnostic"):
        apply(replace(queue, frames=(replace(frame, review_role=Role.TRAIN),)), ())
