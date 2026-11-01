"""A complete synthetic pretraining -> transfer -> sealed-test fine-tuning run."""

import argparse

import pytest

torch = pytest.importorskip("torch")

import avm  # noqa: E402
import pretrain_avm  # noqa: E402
import train  # noqa: E402
from torch import nn  # noqa: E402
from prepare import read_json, sha256, write_json  # noqa: E402

pytest_plugins = ("test_avm", "test_train")


class Tiny(nn.Module):
    def __init__(self, classes=3):
        super().__init__()
        self.encoder = torch.nn.Conv2d(3, 4, 1)
        self.decoder = torch.nn.Conv2d(4, 4, 1)
        self.segmentation_head = torch.nn.Conv2d(4, classes, 1)

    def forward(self, x):
        return self.segmentation_head(self.decoder(self.encoder(x)))


def test_pretraining_transfer_and_finetuning(archive, frozen, tmp_path, monkeypatch):
    zipped, provenance = archive
    source = read_json(provenance)
    source["research_use_basis_accepted"] = True  # synthetic fixtures only
    write_json(provenance, source)
    dataset = tmp_path / "avm"
    avm.import_archive(zipped, dataset, provenance, count=2)
    monkeypatch.setattr(
        pretrain_avm, "build_model", lambda pretrained=None, classes=3: Tiny(classes)
    )
    monkeypatch.setattr(
        train, "build_model", lambda pretrained=None, classes=3: Tiny(classes)
    )
    pretrained = tmp_path / "synthetic-pretrained.json"
    write_json(pretrained, {"test_only": True})
    output = tmp_path / "pretrain"
    report = pretrain_avm.run(
        argparse.Namespace(
            dataset=dataset,
            pretrained=pretrained,
            output=output,
            seed=42,
            batch_size=2,
            epochs=1,
            device="cpu",
        )
    )
    assert report["checkpoint_reload_identical"] and not report["CNR_used"]
    assert report["steps"] == 1 and not report["PUCPR_used"]
    fine = tmp_path / "fine"
    args = argparse.Namespace(
        frozen=frozen,
        pretrained=pretrained,
        output=fine,
        seed=42,
        batch_size=2,
        epochs=2,
        patience=2,
        min_delta=0.001,
        device="cpu",
        tile_size=64,
        transfer_checkpoint=output / "last.pt",
        transfer_sha256=sha256(output / "last.pt"),
    )
    before = sha256(output / "last.pt")
    result = train.run(args)
    assert (
        not result["test_evaluated"]
        and result["checkpoint_reload_predictions_identical"]
    )
    assert sha256(output / "last.pt") == before
    assert read_json(fine / "transfer.json")["feature_tensors_transferred"] == 4
    assert set(result["validation"]["per_image"]) == {"val"}
    assert result["validation"]["classes"]["curb"]["support"] > 0
    assert not list((fine / "validation").glob("test*"))
    with pytest.raises(ValueError, match="NEW directory"):
        pretrain_avm.run(argparse.Namespace(output=output))
