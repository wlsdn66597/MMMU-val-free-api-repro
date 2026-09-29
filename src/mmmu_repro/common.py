import hashlib
import json
import os
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def file_hash(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    result = []
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Incomplete/corrupt JSONL at {path.name}:{n}; restore the last complete line before resuming") from e
    return result


def append_jsonl(path, row):
    with Path(path).open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def lock_manifest(directory, filename, identity):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != identity:
            raise ValueError(f"Settings/source changed: choose a NEW output directory ({path.name})")
    else:
        save_json(path, identity)


class RunLock:
    """Prevent two processes from writing the same run, without recording secrets."""
    def __init__(self, directory):
        self.path = Path(directory) / ".running.lock"

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as e:
            raise ValueError(f"Run is locked at {self.path}. If its process has ended, remove only this lock and resume.") from e
        with os.fdopen(fd, "w") as f:
            f.write(str(os.getpid()))
        return self

    def __exit__(self, *args):
        self.path.unlink(missing_ok=True)


def source_hash():
    root = Path(__file__).parent
    return digest({str(p.relative_to(root)): file_hash(p) for p in sorted(root.rglob("*.py"))})
