"""Single-writer occupancy journal and atomic current snapshot."""

import fcntl
import json
import os
import tempfile
from contextlib import ExitStack
from pathlib import Path


def _repair_tail(stream):
    """Keep completed records and recover at most the last unterminated line."""
    size = stream.seek(0, os.SEEK_END)
    if not size:
        return
    stream.seek(-1, os.SEEK_END)
    if stream.read(1) == b"\n":
        return

    end = size
    boundary = 0
    while end:
        start = max(0, end - 65536)
        stream.seek(start)
        index = stream.read(end - start).rfind(b"\n")
        if index >= 0:
            boundary = start + index + 1
            break
        end = start

    stream.seek(boundary)
    try:
        complete = isinstance(json.loads(stream.read()), dict)
    except UnicodeDecodeError, json.JSONDecodeError:
        complete = False
    if complete:
        stream.write(b"\n")
    else:
        stream.truncate(boundary)
    os.fsync(stream.fileno())


class LiveOutput:
    """Own a directory while publishing. Each restart leaves the journal intact.

    A journal event is synced before the corresponding snapshot is replaced.
    An interrupted publish can leave a newer journal event and an older snapshot,
    but readers of the snapshot always see a complete JSON document.
    """

    def __init__(self, directory: Path):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self._resources = ExitStack()
        try:
            lock = self._resources.enter_context(
                (directory / ".live-occupancy.lock").open("a+b")
            )
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise BlockingIOError(
                    f"Another live runner owns output directory: {directory}"
                ) from error
            self._journal = self._resources.enter_context(
                (directory / "events.jsonl").open("a+b", buffering=0)
            )
            _repair_tail(self._journal)
        except BaseException:
            self._resources.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()

    def close(self) -> None:
        self._resources.close()

    def _append(self, data: bytes) -> None:
        offset = self._journal.seek(0, os.SEEK_END)
        pending = memoryview(data)
        try:
            while pending:
                written = self._journal.write(pending)
                if not written:
                    raise OSError("Could not append to the live occupancy journal")
                pending = pending[written:]
            os.fsync(self._journal.fileno())
        except OSError:
            self._journal.truncate(offset)
            raise

    def publish(self, event: dict, snapshot: dict) -> None:
        if self._journal.closed:
            raise ValueError("Live output is closed")
        record = json.dumps(
            event, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        )
        current = json.dumps(snapshot, ensure_ascii=False, allow_nan=False, indent=2)
        staged = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=self.directory, prefix=".snapshot.", suffix=".tmp", delete=False
            ) as stream:
                staged = Path(stream.name)
                stream.write((current + "\n").encode("utf-8"))
                stream.flush()
                os.fsync(stream.fileno())
            self._append((record + "\n").encode("utf-8"))
            os.replace(staged, self.directory / "snapshot.json")
            directory_fd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if staged is not None:
                staged.unlink(missing_ok=True)
