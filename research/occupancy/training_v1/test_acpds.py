"""ACPDS pairing and strict separation from the previously observed test."""

from copy import deepcopy
import json

import pytest

from .acpds import (
    add_official_split,
    official_annotations,
    slot_annotation,
    valid_annotations,
)
from .data import protected_session, source_identity


def annotations():
    return {
        "train": {"file_names": ["GOPR0001.JPG"]},
        "test": {"file_names": ["GOPR0002.JPG"]},
        "valid": {
            "file_names": ["GOPR0003.JPG"],
            "rois_list": [[[[0, 0], [1, 0], [1, 1], [0, 1]]]],
            "occupancy_list": [[False]],
        },
    }


def test_only_official_valid_names_are_returned():
    pairs = valid_annotations(json.dumps(annotations()).encode())
    assert list(pairs) == ["GOPR0003.JPG"]
    assert pairs["GOPR0003.JPG"][1] == [False]


@pytest.mark.parametrize("kind", ["overlap", "unpaired", "integer_label", "traversal"])
def test_corrupt_acpds_split_and_semantics_rejected(kind):
    data = deepcopy(annotations())
    if kind == "overlap":
        data["valid"]["file_names"][0] = "GOPR0002.JPG"
    elif kind == "unpaired":
        data["valid"]["occupancy_list"][0] = []
    elif kind == "integer_label":
        data["valid"]["occupancy_list"][0] = [0]
    else:
        data["valid"]["file_names"][0] = "../GOPR0003.JPG"
    with pytest.raises(ValueError):
        valid_annotations(json.dumps(data).encode())


def test_normalized_roi_maps_to_pixel_centers_without_relabeling():
    pair = valid_annotations(json.dumps(annotations()).encode())["GOPR0003.JPG"]
    record = slot_annotation(
        {"source_image_id": "GOPR0003.JPG", "width": 100, "height": 50}, pair, 0
    )
    assert record["category_id"] == 0
    assert record["segmentation"] == [[0, 0, 99, 0, 99, 49, 0, 49]]


def test_test_source_path_is_not_loadable():
    with pytest.raises(ValueError):
        source_identity("ACPDS/test/GOPR0002.JPG")


def test_train_is_importable_without_reassigning_validation():
    data = annotations()
    data["train"]["rois_list"] = data["valid"]["rois_list"]
    data["train"]["occupancy_list"] = [[True]]
    pairs = official_annotations(json.dumps(data).encode(), "train")
    assert list(pairs) == ["GOPR0001.JPG"]
    assert pairs["GOPR0001.JPG"][1] == [True]
    train = source_identity("ACPDS/train/GOPR0001.JPG")
    valid = source_identity("ACPDS/valid/GOPR0003.JPG")
    assert train[-1] == "train" and valid[-1] == "val"
    assert train[:3] != valid[:3]
    with pytest.raises(ValueError, match="test cannot be imported"):
        official_annotations(json.dumps(data).encode(), "test")


def test_acquisition_rejects_test_before_reading_or_fetching_anything():
    with pytest.raises(ValueError, match="test cannot be imported"):
        add_official_split(None, None, None, "test")


def test_protection_joins_days_across_other_cameras():
    assert protected_session("CNRPark-EXT/camera6/RAINY/2015-12-22/a.jpg") == (
        "pisa-cnr:2015-12-22"
    )
