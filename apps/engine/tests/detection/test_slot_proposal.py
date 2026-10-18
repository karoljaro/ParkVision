from dataclasses import FrozenInstanceError

import pytest

from parkvision.detection.slot_proposal import SlotProposal
from parkvision.domain.geometry.point import Point
from parkvision.domain.geometry.polygon import Polygon
from parkvision.domain.types import CameraId


@pytest.fixture
def polygon() -> Polygon:
    return Polygon((Point(0.0, 0.0), Point(1.0, 0.0), Point(0.0, 1.0)))


def test_initialization_preserves_fields(polygon: Polygon) -> None:
    camera_id = CameraId("camera-1")

    proposal = SlotProposal(camera_id=camera_id, polygon=polygon)

    assert proposal.camera_id == camera_id
    assert proposal.polygon is polygon


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        pytest.param("camera_id", CameraId("camera-2"), id="camera-id"),
        pytest.param(
            "polygon",
            Polygon((Point(0.2, 0.2), Point(0.8, 0.2), Point(0.5, 0.8))),
            id="polygon",
        ),
    ],
)
def test_fields_cannot_be_reassigned(
    polygon: Polygon, field: str, replacement: object
) -> None:
    proposal = SlotProposal(CameraId("camera-1"), polygon)

    with pytest.raises(FrozenInstanceError):
        setattr(proposal, field, replacement)


@pytest.mark.parametrize("field", ["camera_id", "polygon"])
def test_fields_cannot_be_deleted(polygon: Polygon, field: str) -> None:
    proposal = SlotProposal(CameraId("camera-1"), polygon)

    with pytest.raises(FrozenInstanceError):
        delattr(proposal, field)


def test_equality_and_hash_depend_on_both_fields(polygon: Polygon) -> None:
    proposal = SlotProposal(CameraId("camera-1"), polygon)
    same_proposal = SlotProposal(
        CameraId("camera-1"),
        Polygon((Point(0.0, 0.0), Point(1.0, 0.0), Point(0.0, 1.0))),
    )
    different_camera = SlotProposal(CameraId("camera-2"), polygon)
    different_polygon = SlotProposal(
        proposal.camera_id,
        Polygon((Point(0.2, 0.2), Point(0.8, 0.2), Point(0.5, 0.8))),
    )

    assert proposal == same_proposal
    assert hash(proposal) == hash(same_proposal)
    assert proposal != different_camera
    assert proposal != different_polygon
    assert len({proposal, same_proposal, different_camera, different_polygon}) == 3


def test_one_camera_can_have_multiple_proposals(polygon: Polygon) -> None:
    camera_id = CameraId("camera-1")
    other_polygon = Polygon((Point(0.2, 0.2), Point(0.8, 0.2), Point(0.5, 0.8)))

    first_proposal = SlotProposal(camera_id, polygon)
    second_proposal = SlotProposal(camera_id, other_polygon)

    assert first_proposal.camera_id == second_proposal.camera_id == camera_id
    assert first_proposal.polygon is polygon
    assert second_proposal.polygon is other_polygon
    assert first_proposal.polygon != second_proposal.polygon
    assert first_proposal != second_proposal
    assert len({first_proposal, second_proposal}) == 2


def test_identical_polygons_can_be_proposed_for_different_cameras(
    polygon: Polygon,
) -> None:
    first_proposal = SlotProposal(CameraId("camera-1"), polygon)
    second_proposal = SlotProposal(CameraId("camera-2"), polygon)

    assert first_proposal.camera_id != second_proposal.camera_id
    assert first_proposal.polygon is second_proposal.polygon is polygon
    assert first_proposal != second_proposal
    assert len({first_proposal, second_proposal}) == 2
