"""A resumed training run must retain a valid best even without improvement."""

from pathlib import Path

import pytest
import torch

from .data import RECIPE, file_hash
from .model import ARCHITECTURE
from .train import inherit_best_checkpoint


def checkpoint():
    return {
        "manifest_sha256": "a" * 64,
        "architecture": ARCHITECTURE,
        "recipe": RECIPE,
        "epoch": 1,
        "best_loss": 0.8,
        "state_dict": {"example": torch.tensor([1.0])},
    }


def test_resume_retains_best_when_later_epochs_do_not_improve(tmp_path):
    previous, output = tmp_path / "previous", tmp_path / "resumed"
    previous.mkdir()
    output.mkdir()
    torch.save(checkpoint(), previous / "best.pt")
    provenance = inherit_best_checkpoint(previous / "last.pt", output, "a" * 64, 0.8, 1)
    assert file_hash(output / "best.pt") == file_hash(previous / "best.pt")
    assert Path(provenance["path"]) == previous / "best.pt"


@pytest.mark.parametrize(
    "field,value",
    [
        ("manifest_sha256", "b" * 64),
        ("architecture", "wrong"),
        ("best_loss", 0.9),
        ("epoch", 3),
    ],
)
def test_resume_rejects_unrelated_best(tmp_path, field, value):
    previous, output = tmp_path / "previous", tmp_path / "resumed"
    previous.mkdir()
    output.mkdir()
    state = checkpoint()
    state[field] = value
    torch.save(state, previous / "best.pt")
    with pytest.raises(ValueError, match="does not match resume"):
        inherit_best_checkpoint(previous / "last.pt", output, "a" * 64, 0.8, 1)
    assert not (output / "best.pt").exists()


def test_resume_does_not_silently_use_last_as_best(tmp_path):
    with pytest.raises(ValueError, match="sibling best.pt"):
        inherit_best_checkpoint(tmp_path / "last.pt", tmp_path, "a" * 64, 0.8, 1)
