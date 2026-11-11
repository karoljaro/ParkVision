"""Small synthetic format/integrity cases; no held-out real data is loaded."""

from copy import deepcopy
import io
import json
import tarfile

import numpy as np
import pytest

from .data import (
    COMMIT,
    RECIPE,
    VIEWS,
    archive_data,
    blob_hash,
    digest,
    encode_slot,
    file_hash,
    local_path,
    polygon,
    select_frames,
    source_identity,
    verify,
)


@pytest.mark.parametrize(
    "path",
    [
        "PKLot/PUCPR/Sunny/2012-01-01/a.jpg",
        "T-01/a.jpg",
        "CNRPark-EXT/UFPR04/Sunny/2012-01-01/a.jpg",
        "../../outside.jpg",
    ],
)
def test_protected_and_misrepresented_sources_rejected(path):
    with pytest.raises(ValueError):
        source_identity(path)


def test_previous_cnr_parking_cannot_become_validation():
    with pytest.raises(ValueError):
        source_identity("CNRPark-EXT/camera1/SUNNY/2015-11-22/a.jpg")


def test_same_parking_views_share_split():
    a = source_identity("PKLot/UFPR04/Sunny/2012-01-01/a.jpg")
    b = source_identity("PKLot/UFPR05/Sunny/2012-01-01/b.jpg")
    assert (a[0], a[2], a[3]) == (b[0], b[2], b[3])


def test_session_selection_is_deterministic_and_excludes_all_views_of_day():
    images = [
        {
            "file_name": f"PKLot/{camera}/Sunny/2012-01-0{day}/{hour}.jpg",
            "time": [hour, 0, 0],
        }
        for camera in ("UFPR04", "UFPR05")
        for day in (1, 2)
        for hour in range(8, 17)
    ]
    first = select_frames(images, {"ufpr:2012-01-01"}, 2, 42)
    assert len(first) == 4
    assert first == select_frames(list(reversed(images)), {"ufpr:2012-01-01"}, 2, 42)
    assert all("2012-01-02" in x["file_name"] for x in first)


def test_rgb_and_mask_stay_aligned_and_context_is_present():
    rgb = np.zeros((60, 100, 3), np.uint8)
    rgb[10:31, 40:61] = [255, 0, 0]
    array, box = encode_slot(rgb, [[40, 10], [60, 10], [60, 30], [40, 30]])
    assert array.shape == (224, 224, 4)
    assert set(np.unique(array[:, :, 3])) == {0, 255}
    assert (array[:, :, 0][array[:, :, 3] == 255] > 200).mean() > 0.9
    assert box[0] < 40 and box[2] > 60
    # Metadata must be JSON serializable, including NumPy-derived crop bounds.
    json.dumps(box)


@pytest.mark.parametrize(
    "segment",
    [
        [[0, 0, 8, 8, 0, 8, 8, 0]],
        [[-1, 0, 8, 0, 8, 8]],
        [[0, 0, 10, 0, 10, 8]],
        [[0, 0, 8, 0, 8, 8], [1, 1, 2, 2]],
    ],
)
def test_invalid_polygon_rejected(segment):
    with pytest.raises(ValueError):
        polygon({"segmentation": segment}, 10, 10)


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    # Synthetic second site lets the generic COCO validator exercise both roles.
    monkeypatch.setitem(
        VIEWS,
        "camera1",
        ("CNRPark-EXT", "synthetic-spots", "synthetic-site", "val", "synthetic"),
    )
    manifest = {
        "schema_version": 1,
        "source_commit": COMMIT,
        "recipe": RECIPE,
        "sources": {},
        "frames": [],
        "samples": [],
        "exclusions": {"sessions": [], "sha256": []},
    }
    import cv2

    for index, camera in enumerate(("UFPR04", "camera1")):
        dataset, member, parking, split, _ = VIEWS[camera]
        path = f"{dataset}/{camera}/Sunny/2012-01-01/frame.jpg"
        rgb = np.full((40, 50, 3), 30 + index * 50, dtype=np.uint8)
        raw = cv2.imencode(".jpg", rgb)[1].tobytes()
        (tmp_path / f"image{index}.jpg").write_bytes(raw)
        original = {"id": index, "file_name": path, "width": 50, "height": 40}
        annotations = [
            {
                "id": label,
                "image_id": index,
                "category_id": label,
                "segmentation": [
                    [
                        1 + label * 20,
                        1,
                        15 + label * 20,
                        1,
                        15 + label * 20,
                        20,
                        1 + label * 20,
                        20,
                    ]
                ],
            }
            for label in (0, 1)
        ]
        data = {
            "categories": [{"id": 0, "name": "empty"}, {"id": 1, "name": "occupied"}],
            "images": [original],
            "annotations": annotations,
        }
        archive = tmp_path / f"{member}.tar.xz"
        payload = json.dumps(data).encode()
        with tarfile.open(archive, "w:xz") as tar:
            entry = tarfile.TarInfo(member + ".json")
            entry.size = len(payload)
            tar.addfile(entry, io.BytesIO(payload))
        manifest["sources"][camera] = {
            "archive": archive.name,
            "sha256": file_hash(archive),
            "git_blob": blob_hash(archive.read_bytes()),
        }
        group = {
            "parking_id": parking,
            "camera_id": camera,
            "session_id": f"{parking}:2012-01-01",
            "split": split,
        }
        manifest["frames"].append(
            dict(
                group,
                source_path=path,
                source_image_id=index,
                width=50,
                height=40,
                image=f"image{index}.jpg",
                sha256=digest(raw),
                git_blob=blob_hash(raw),
            )
        )
        for annotation in annotations:
            label = annotation["category_id"]
            array, box = encode_slot(rgb, polygon(annotation, 50, 40))
            # Distinct source pixels per target footprint for dedup assertions.
            destination = tmp_path / f"{camera}{label}.npy"
            np.save(destination, array, allow_pickle=False)
            manifest["samples"].append(
                dict(
                    group,
                    id=f"{camera}{label}",
                    source_path=path,
                    source_annotation_id=label,
                    roi=polygon(annotation, 50, 40),
                    crop_box=box,
                    label=label,
                    visibility="unassessed",
                    array=destination.name,
                    sha256=file_hash(destination),
                    pixel_sha256=digest(array.tobytes()),
                )
            )
    return tmp_path, manifest


def test_valid_manifest_with_real_archive_and_arrays(dataset):
    root, manifest = dataset
    verify(root, manifest)


@pytest.mark.parametrize(
    "kind", ["split", "duplicate", "protected", "session", "parent", "label"]
)
def test_leakage_duplicate_and_annotation_tampering_rejected(dataset, kind):
    root, original = dataset
    manifest = deepcopy(original)
    if kind == "split":
        manifest["frames"][0]["split"] = "val"
    elif kind == "duplicate":
        manifest["samples"][1]["pixel_sha256"] = manifest["samples"][0]["pixel_sha256"]
    elif kind == "protected":
        manifest["exclusions"]["sha256"].append(manifest["frames"][0]["sha256"])
    elif kind == "session":
        manifest["exclusions"]["sessions"].append(manifest["frames"][0]["session_id"])
    elif kind == "parent":
        manifest["samples"][0]["split"] = "val"
    else:
        manifest["samples"][0]["label"] = 1
    with pytest.raises(ValueError):
        verify(root, manifest)


def test_corrupt_array_and_escaping_paths_rejected(dataset):
    root, manifest = dataset
    (root / manifest["samples"][0]["array"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        verify(root, manifest)
    with pytest.raises(ValueError):
        local_path(root, "../escape")


def test_reencoded_identical_frame_pixels_are_rejected_across_splits(dataset):
    import cv2

    root, manifest = dataset
    first, second = manifest["frames"]
    image = cv2.imread(str(root / first["image"]))
    different_encoding = cv2.imencode(".png", image)[1].tobytes()
    (root / second["image"]).write_bytes(different_encoding)
    second["sha256"] = digest(different_encoding)
    second["git_blob"] = blob_hash(different_encoding)
    assert first["sha256"] != second["sha256"]
    with pytest.raises(ValueError, match="Duplicate decoded frame pixels"):
        verify(root, manifest)


def test_detection_free_is_not_accepted_as_classification_format(tmp_path):
    raw = json.dumps(
        {"categories": [{"id": 0, "name": "car"}, {"id": 1, "name": "free"}]}
    ).encode()
    path = tmp_path / "detection.tar.xz"
    with tarfile.open(path, "w:xz") as tar:
        entry = tarfile.TarInfo("data.json")
        entry.size = len(raw)
        tar.addfile(entry, io.BytesIO(raw))
    with pytest.raises(ValueError):
        archive_data(path, "data")
