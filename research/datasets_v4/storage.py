"""Strict JSON boundary and verified local/ZIP assets; no network access."""

from dataclasses import fields, is_dataclass
from enum import Enum
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import types
from typing import Any, TypeVar, Union, get_args, get_origin, get_type_hints
import zipfile

from .contracts import Asset

T = TypeVar("T")


@lru_cache(maxsize=64)
def hints(cls: type) -> dict[str, Any]:
    return get_type_hints(cls)


def encode(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: encode(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return [encode(x) for x in value]
    return value


def fingerprint(value: Any) -> str:
    raw = json.dumps(
        encode(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(raw.encode()).hexdigest()


def decode(cls: type[T], value: Any, at: str = "$") -> T:
    """Reject extra fields, coercions (including bool-as-int) and unknown enums."""
    origin, args = get_origin(cls), get_args(cls)
    if origin in (Union, types.UnionType):
        for choice in args:
            try:
                return decode(choice, value, at)
            except (TypeError, ValueError):
                pass
        raise ValueError(f"{at}: invalid union value")
    if cls is type(None):
        if value is not None:
            raise ValueError(f"{at}: expected null")
        return value
    if origin is tuple:
        if type(value) is not list:
            raise ValueError(f"{at}: expected array")
        return tuple(decode(args[0], v, f"{at}[{i}]") for i, v in enumerate(value))
    if isinstance(cls, type) and issubclass(cls, Enum):
        if type(value) is not str:
            raise ValueError(f"{at}: expected enum string")
        return cls(value)
    if is_dataclass(cls):
        if type(value) is not dict:
            raise ValueError(f"{at}: expected object")
        known = {f.name for f in fields(cls)}
        if value.keys() - known:
            raise ValueError(f"{at}: unknown fields {sorted(value.keys() - known)}")
        annotations = hints(cls)
        return cls(
            **{
                key: decode(annotations[key], val, f"{at}.{key}")
                for key, val in value.items()
            }
        )
    if cls in (str, int, bool) and type(value) is cls:
        return value
    raise ValueError(f"{at}: expected {cls}")


def load(path: Path, cls: type[T]) -> T:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"Duplicate JSON key {key}")
            result[key] = value
        return result

    return decode(cls, json.loads(path.read_text(), object_pairs_hook=pairs))


def save(path: Path, value: Any) -> None:
    with path.open("x") as stream:
        stream.write(
            json.dumps(encode(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
        )


class AssetVerifier:
    """Each gate invocation hashes fresh bytes; cache lifetime is one invocation."""

    def __init__(self) -> None:
        self.cache: dict[tuple[str, str | None], str] = {}

    def actual_hash(self, asset: Asset) -> str:
        key = (asset.path, asset.member)
        if key not in self.cache:
            digest = hashlib.sha256()
            if asset.member is not None:
                with zipfile.ZipFile(asset.path) as archive:
                    if len([n for n in archive.namelist() if n == asset.member]) != 1:
                        raise ValueError("Missing or ambiguous ZIP member")
                    with archive.open(asset.member) as stream:
                        while chunk := stream.read(1024 * 1024):
                            digest.update(chunk)
            else:
                with Path(asset.path).open("rb") as stream:
                    while chunk := stream.read(1024 * 1024):
                        digest.update(chunk)
            self.cache[key] = digest.hexdigest()
        return self.cache[key]

    def verify(self, asset: Asset) -> bool:
        try:
            return self.actual_hash(asset) == asset.sha256
        except (OSError, ValueError, KeyError, zipfile.BadZipFile, RuntimeError) as exc:
            raise ValueError(f"Cannot verify {asset.path}: {exc}") from exc
