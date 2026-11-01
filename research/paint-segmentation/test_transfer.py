"""AVM transfer scope and complete tensor provenance, including BN buffers."""

import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402
import train  # noqa: E402
from prepare import sha256  # noqa: E402


class Model(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4))
        self.decoder = nn.Sequential(nn.Conv2d(4, 4, 1), nn.BatchNorm2d(4))
        self.segmentation_head = nn.Conv2d(4, classes, 1)


@pytest.mark.parametrize("scope", ("encoder", "encoder_decoder"))
def test_scope_copies_only_selected_tensors_and_preserves_rng(tmp_path, scope):
    source = Model(2)
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.fill_(11.0)
        for module in (source.encoder, source.decoder):
            module[1].running_mean.fill_(7.0)
            module[1].running_var.fill_(9.0)
            module[1].num_batches_tracked.fill_(17)
    path = tmp_path / "avm.pt"
    torch.save(
        {
            "epoch": 3,
            "model": source.state_dict(),
            "config": {
                "schema": "parkvision-avm-pretraining-v1",
                "classes": {"background": 0, "paint": 1, "ignore": 255},
            },
        },
        path,
    )
    destination = Model(3)
    fresh = {key: value.clone() for key, value in destination.state_dict().items()}
    rng_before = torch.get_rng_state().clone()
    info = train.transfer_avm_features(destination, path, sha256(path), scope=scope)
    assert torch.equal(torch.get_rng_state(), rng_before)
    rows = {row["name"]: row for row in info["tensor_audit"]}
    assert set(rows) == set(fresh)
    for name, result in destination.state_dict().items():
        copied = name.startswith("encoder.") or (
            scope == "encoder_decoder" and name.startswith("decoder.")
        )
        expected = source.state_dict()[name] if copied else fresh[name]
        assert torch.equal(result, expected), name
        row = rows[name]
        assert row["action"] == (
            "copied_from_avm" if copied else "kept_fresh_initialization"
        )
        assert (
            row["result_sha256"] == row["source_sha256" if copied else "initial_sha256"]
        )
    assert rows["encoder.1.num_batches_tracked"]["kind"] == "buffer"
    assert info["all_retained_tensors_equal_fresh_initialization"]
    assert info["all_transferred_tensors_equal_source"]
    with pytest.raises(ValueError, match="scope"):
        train.transfer_avm_features(destination, path, sha256(path), scope="head")


def test_shape_mismatch_fails_without_changing_model(tmp_path):
    model = Model(3)
    fresh = {key: value.clone() for key, value in model.state_dict().items()}
    state = Model(2).state_dict()
    state["encoder.0.weight"] = torch.zeros(5, 3, 1, 1)
    path = tmp_path / "bad-shape.pt"
    torch.save(
        {
            "epoch": 3,
            "model": state,
            "config": {
                "schema": "parkvision-avm-pretraining-v1",
                "classes": {"background": 0, "paint": 1, "ignore": 255},
            },
        },
        path,
    )
    with pytest.raises(ValueError, match="shape"):
        train.transfer_avm_features(model, path, sha256(path), scope="encoder")
    assert all(
        torch.equal(value, fresh[key]) for key, value in model.state_dict().items()
    )
