"""Single-worker file monitoring, stable-image retries, and durable inspection logs."""
import csv
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from c4_runtime import Predictor, inspect_frame, save_inspection


EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class FileInbox:
    def __init__(self, source, recursive=False, include_existing=True, stable_seconds=2):
        self.source = source
        self.recursive = recursive
        self.include_existing = include_existing
        self.stable_seconds = stable_seconds
        self.pending = {}
        self.finished = {}
        self.initial = True

    def paths(self):
        if isinstance(self.source, (list, tuple)):
            return [Path(path) for path in self.source]
        root = Path(self.source)
        if not root.is_dir():
            raise OSError("Input folder is unavailable. Reconnect it before restarting inspection.")
        def visit(folder):
            with os.scandir(folder) as entries:
                for entry in entries:
                    path = Path(entry.path)
                    if entry.is_symlink() or getattr(path, "is_junction", lambda: False)():
                        continue
                    if entry.is_dir(follow_symlinks=False) and self.recursive:
                        yield from visit(path)
                    elif entry.is_file() and path.suffix.lower() in EXTENSIONS:
                        yield path
        return list(visit(root))

    def scan(self, now=None):
        now = time.monotonic() if now is None else now
        present = set()
        for path in sorted(self.paths()):
            path = path.resolve()
            present.add(path)
            try:
                stat = path.stat()
            except FileNotFoundError:
                if isinstance(self.source, (list, tuple)):
                    raise
                continue
            signature = (stat.st_size, stat.st_mtime_ns)
            if self.initial and not self.include_existing:
                self.finished[path] = signature
            if self.finished.get(path) == signature:
                continue
            previous = self.pending.get(path)
            if previous is None or previous["signature"] != signature:
                self.pending[path] = {"signature": signature, "since": now, "attempts": 0}
        self.initial = False
        for path in set(self.pending) - present:
            del self.pending[path]
        for path in set(self.finished) - present:
            del self.finished[path]
        return [(path, row["signature"]) for path, row in sorted(self.pending.items())
                if now - row["since"] >= self.stable_seconds]

    def finish(self, path, signature):
        self.finished[path] = signature
        self.pending.pop(path, None)

    def retry(self, path, now=None):
        row = self.pending[path]
        row["attempts"] += 1
        row["since"] = time.monotonic() if now is None else now
        return row["attempts"] < 5


def validate_folders(source, output):
    output = Path(output).resolve()
    if not output.name:
        raise ValueError("Choose a results folder.")
    if not isinstance(source, (list, tuple)):
        source = Path(source).resolve()
        if not source.is_dir():
            raise ValueError("Choose an existing image folder.")
        if source == output or source.is_relative_to(output) or output.is_relative_to(source):
            raise ValueError("Image and results folders must be separate, not inside one another.")
    elif not source:
        raise ValueError("Select at least one image.")


class InspectionWorker:
    def __init__(self, model_dir, source, output, events, stop, watch=False,
                 include_existing=True, recursive=False, predictor_factory=Predictor, stable_seconds=2):
        validate_folders(source, output)
        self.model_dir, self.source, self.output = Path(model_dir), source, Path(output)
        self.events, self.stop = events, stop
        self.watch, self.recursive = watch, recursive
        self.predictor_factory = predictor_factory
        self.inbox = FileInbox(source, recursive, include_existing, stable_seconds)
        self.counts = {"processed": 0, "triggers": 0, "errors": 0, "reviews": 0}

    def emit(self, kind, payload):
        self.events.put((kind, payload))

    def run(self):
        try:
            self.inbox.scan()
            if not self.watch:
                self.inbox.source = list(self.inbox.pending)
            predictor = self.predictor_factory(self.model_dir)
            self.output.mkdir(parents=True, exist_ok=True)
            for decision in ("Good", "Bad", "Review"):
                (self.output / "Result" / decision).mkdir(parents=True, exist_ok=True)
            session = self.output / (datetime.now().strftime("session_%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8])
            session.mkdir()
            settings = {"created_utc": datetime.now(timezone.utc).isoformat(), "model": predictor.metadata,
                        "source": [str(path) for path in self.source] if isinstance(self.source, (list, tuple)) else str(self.source),
                        "watch": self.watch, "recursive": self.recursive,
                        "include_existing": self.inbox.include_existing}
            (session / "session.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
            self.emit("ready", {"session_dir": str(session), "metadata": predictor.metadata})
            with (session / "inspection.csv").open("w", newline="", encoding="utf-8-sig") as stream:
                fields = ["time_utc", "source_path", "result", "unit_1", "unit_2", "unit_3", "threshold_raw",
                          "elapsed_seconds", "result_dir", "error", "tray_result", "tray_image_path"]
                fields.extend(f"{name}_{unit}" for unit in range(1, 4)
                              for name in ("vid", "vid_status", "outcome", "shift_x", "shift_y"))
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                while not self.stop.is_set():
                    ready = self.inbox.scan()
                    for path, signature in ready:
                        if self.stop.is_set():
                            break
                        log = {"time_utc": datetime.now(timezone.utc).isoformat(), "source_path": str(path)}
                        try:
                            stat = path.stat()
                            if (stat.st_size, stat.st_mtime_ns) != signature:
                                continue
                            inspection = inspect_frame(path, predictor)
                        except OSError as exc:
                            if self.inbox.retry(path):
                                continue
                            log.update(result="ERROR", error=str(exc))
                        except ValueError as exc:
                            log.update(result="ERROR", error=str(exc))
                        else:
                            result = save_inspection(inspection, session, predictor.metadata)
                            self.counts["processed"] += 1
                            self.counts["triggers"] += int(result["triggered"])
                            self.counts["reviews"] += int(result["tray_result"] == "Review")
                            log.update(result="TRIGGER" if result["triggered"] else "NO TRIGGER",
                                       threshold_raw=result["threshold_raw"], elapsed_seconds=result["elapsed_seconds"],
                                       result_dir=result["result_dir"], tray_result=result["tray_result"],
                                       tray_image_path=result["tray_image_path"])
                            log.update({f"unit_{row['unit']}": row["raw_score"] for row in result["units"]})
                            for row in result["units"]:
                                unit = row["unit"]
                                log.update({f"vid_{unit}": row["vid"], f"vid_status_{unit}": row["vid_status"],
                                            f"outcome_{unit}": row["result"],
                                            f"shift_x_{unit}": row["alignment"]["offset_xy"][0],
                                            f"shift_y_{unit}": row["alignment"]["offset_xy"][1]})
                        writer.writerow(log)
                        stream.flush()
                        os.fsync(stream.fileno())
                        self.inbox.finish(path, signature)
                        if log["result"] == "ERROR":
                            self.counts["errors"] += 1
                            self.emit("file_error", log)
                        else:
                            self.emit("result", result)
                    self.emit("counts", {**self.counts, "pending": len(self.inbox.pending)})
                    if not self.watch and not self.inbox.pending:
                        break
                    self.stop.wait(0.25)
        except Exception as exc:
            self.emit("fatal", f"Inspection stopped: {type(exc).__name__}: {exc}")
        finally:
            self.emit("done", self.counts.copy())