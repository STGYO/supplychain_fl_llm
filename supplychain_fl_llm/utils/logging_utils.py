from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from threading import Lock
from typing import Any

_JSONL_LOCK = Lock()


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def configure_logger(name: str, log_file: str | Path | None = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file is not None:
        target = Path(log_file)
        ensure_dir(target.parent)
        file_handler = logging.FileHandler(target, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def append_jsonl(file_path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(file_path)
    ensure_dir(target.parent)

    safe_payload = dict(payload)
    safe_payload.setdefault("timestamp", utc_now_iso())

    with _JSONL_LOCK:
        with target.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(safe_payload, sort_keys=True) + "\n")


def read_jsonl(file_path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    target = Path(file_path)
    if not target.exists():
        return []

    rows: list[dict[str, Any]] = []
    with target.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    if limit is not None and limit > 0:
        return rows[-limit:]

    return rows


def write_json(file_path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(file_path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, sort_keys=True)


def read_json(file_path: str | Path) -> dict[str, Any] | None:
    target = Path(file_path)
    if not target.exists():
        return None

    with target.open("r", encoding="utf-8") as fp:
        return json.load(fp)
