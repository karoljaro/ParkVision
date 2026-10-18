from dataclasses import FrozenInstanceError

import pytest

from parkvision.domain.geometry.point import Point
from parkvision.domain.geometry.polygon import Polygon
from parkvision.domain.slot_projection import SlotProjection
from parkvision.domain.types import CameraId, ParkingSlotId


@pytest.fixture
def polygon() -> Polygon:
    return Polygon((Point(0.0, 0.0), Point(1.0, 0.0), Point(0.0, 1.0)))


def test_initialization_preserves_fields(polygon: Polygon) -> None:
    camera_id = CameraId("camera-1")
    parking_slot_id = ParkingSlotId("slot-1")

    projection = SlotProjection(
        camera_id=camera_id, parking_slot_id=parking_slot_id, polygon=polygon
    )

    assert projection.camera_id == camera_id
    assert projection.parking_slot_id == parking_slot_id
    assert projection.polygon is polygon


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        pytest.param("camera_id", CameraId("camera-2"), id="camera-id"),
        pytest.param("parking_slot_id", ParkingSlotId("slot-2"), id="parking-slot-id"),
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
    projection = SlotProjection(CameraId("camera-1"), ParkingSlotId("slot-1"), polygon)

    with pytest.raises(FrozenInstanceError):
        setattr(projection, field, replacement)


@pytest.mark.parametrize("field", ["camera_id", "parking_slot_id", "polygon"])
def test_fields_cannot_be_deleted(polygon: Polygon, field: str) -> None:
    projection = SlotProjection(CameraId("camera-1"), ParkingSlotId("slot-1"), polygon)

    with pytest.raises(FrozenInstanceError):
        delattr(projection, field)


def test_equality_and_hash_depend_on_all_fields(polygon: Polygon) -> None:
    projection = SlotProjection(CameraId("camera-1"), ParkingSlotId("slot-1"), polygon)
    same_projection = SlotProjection(
        CameraId("camera-1"),
        ParkingSlotId("slot-1"),
        Polygon((Point(0.0, 0.0), Point(1.0, 0.0), Point(0.0, 1.0))),
    )
    different_camera = SlotProjection(
        CameraId("camera-2"), projection.parking_slot_id, polygon
    )
    different_slot = SlotProjection(
        projection.camera_id, ParkingSlotId("slot-2"), polygon
    )
    different_polygon = SlotProjection(
        projection.camera_id,
        projection.parking_slot_id,
        Polygon((Point(0.2, 0.2), Point(0.8, 0.2), Point(0.5, 0.8))),
    )

    assert projection == same_projection
    assert hash(projection) == hash(same_projection)
    assert projection != different_camera
    assert projection != different_slot
    assert projection != different_polygon
    projections = {
        projection,
        same_projection,
        different_camera,
        different_slot,
        different_polygon,
    }
    assert len(projections) == 4


def test_same_parking_slot_can_have_different_camera_projections(
    polygon: Polygon,
) -> None:
    parking_slot_id = ParkingSlotId("slot-1")
    other_polygon = Polygon((Point(0.2, 0.2), Point(0.8, 0.2), Point(0.5, 0.8)))

    first_projection = SlotProjection(CameraId("camera-1"), parking_slot_id, polygon)
    second_projection = SlotProjection(
        CameraId("camera-2"), parking_slot_id, other_polygon
    )

    assert first_projection.parking_slot_id == second_projection.parking_slot_id
    assert first_projection.camera_id != second_projection.camera_id
    assert first_projection.polygon is polygon
    assert second_projection.polygon is other_polygon
    assert first_projection.polygon != second_projection.polygon
    assert first_projection != second_projection
