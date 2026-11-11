"""Shared import state and integrity checks; importers never edit their inputs."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path

from .contracts import (
    Asset,
    Candidate,
    Dataset,
    Evidence,
    Fact,
    Knowledge,
    License,
    Lock,
    Method,
    Record,
    Relation,
    RelationKind,
    Verification,
)
from .storage import AssetVerifier


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def key(prefix: str, value: str) -> str:
    return prefix + ":" + hashlib.sha256(value.encode()).hexdigest()[:24]


def lines(path: Path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


class ImportContext:
    def __init__(self, base: Path, audit: Path, v3: Path, output: Path) -> None:
        self.base, self.audit, self.v3, self.output = base, audit, v3, output
        self.datasets: dict[str, Dataset] = {}
        self.records: dict[str, Record] = {}
        self.evidence: dict[str, Evidence] = {}
        self.relations: dict[str, Relation] = {}
        self.reviewed: dict[str, Candidate] = {}
        self.locks: dict[str, Lock] = {}
        self.by_path: dict[str, str] = {}
        self.by_source: dict[str, str] = {}
        self.raw: dict[tuple[str, str], str] = {}
        self.technical_ids: list[str] = []
        self.native_refs: dict[tuple[str, int], str] = {}
        self.inputs: dict[str, str] = {}
        self.original: dict[str, dict] = {}
        self.verifier = AssetVerifier()
        self.documents: dict[str, dict] = {}
        self.file_hashes: dict[str, str] = {}

    def asset(self, path: Path, expected: str | None = None) -> Asset:
        name = str(path.resolve())
        if expected is None:
            if name not in self.file_hashes:
                self.file_hashes[name] = file_hash(path)
            expected = self.file_hashes[name]
        asset = Asset(name, expected)
        if not self.verifier.verify(asset):
            raise ValueError(f"Changed input: {path}")
        self.inputs[asset.path] = asset.sha256
        return asset

    def ev(
        self,
        path: Path,
        locator: str,
        method: Method = Method.MANIFEST,
        status: Verification = Verification.CONFIRMED,
    ) -> str:
        asset = self.asset(path)
        identity = key("ev", f"{asset.path}:{locator}:{method}:{status}")
        self.evidence[identity] = Evidence(identity, asset, locator, method, status)
        return identity

    def fact(self, value: str | None, evidence: str) -> Fact:
        return Fact() if value is None else Fact(Knowledge.KNOWN, value, (evidence,))

    def dataset(
        self,
        identity: str,
        version: str | None,
        source: str,
        ev: str,
        license_name: str | None = None,
        authorized: bool = False,
        publisher: str | None = None,
    ) -> None:
        if identity not in self.datasets:
            self.datasets[identity] = Dataset(
                identity,
                version,
                source,
                License(
                    license_name,
                    publisher,
                    Verification.CONFIRMED if authorized else Verification.DECLARED,
                    authorized,
                    (ev,),
                ),
                (ev,),
            )

    def add(self, record: Record, *, lock: bool = False) -> None:
        if record.id in self.records:
            raise ValueError(f"Duplicate imported record: {record.id}")
        self.records[record.id] = record
        if record.asset is not None:
            self.by_path.setdefault(record.asset.path, record.id)
        if lock:
            self.locks[record.id] = Lock(
                record.id,
                record.historical_role,
                record.evidence,
                record.training_eligible,
                record.geometry_excluded,
                record.is_parking_slot,
            )

    def link(self, left: str, right: str, kind: RelationKind, ev: str) -> None:
        identity = key("relation", f"{left}:{right}:{kind}")
        self.relations[identity] = Relation(identity, left, right, kind, (ev,))

    def patch(self, identity: str, **changes) -> None:
        self.records[identity] = replace(self.records[identity], **changes)

    def document(self, path: Path) -> dict:
        self.asset(path)
        name = str(path.resolve())
        if name not in self.documents:
            self.documents[name] = json.loads(path.read_text())
        return self.documents[name]

    def verify_audit(self) -> None:
        manifest = self.document(self.audit / "artifact-manifest-final.json")
        for name, expected in manifest["files"].items():
            # All audit evidence, including reviewed images, remains pinned.
            self.asset(self.audit / name, expected)
