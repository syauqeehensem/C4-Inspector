"""CPU-only C4 inference and inspection evidence; no training imports."""
import hashlib
import io
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from c4_alignment import C4Aligner
from c4_identity import read_unit_vid
from heatmaps import render_heatmap, save_map


def sha256_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class Predictor:
    def __init__(self, model_dir, threads=4):
        import torch
        self.torch = torch
        torch.set_num_threads(max(1, min(threads, os.cpu_count() or 1)))
        model_dir = Path(model_dir)
        self.metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
        metadata = self.metadata
        if metadata.get("format") != "iom-c4-raw-v1":
            raise ValueError("Select the exported C4 model folder, not a training checkpoint or bonded model.")
        for key in ("raw_threshold", "pixel_min", "pixel_max"):
            if not isinstance(metadata.get(key), (int, float)) or not math.isfinite(metadata[key]):
                raise ValueError(f"Invalid model calibration: {key}")
        if metadata["pixel_max"] <= metadata["pixel_min"]:
            raise ValueError("Invalid heatmap display bounds.")
        for key in ("input_size", "frame_size_wh"):
            if len(metadata[key]) != 2 or any(type(value) is not int or value <= 0 for value in metadata[key]):
                raise ValueError(f"Invalid model dimensions: {key}")
        width, height = metadata["frame_size_wh"]
        if len(metadata["rois"]) != 3:
            raise ValueError("C4 model must have three unit rectangles.")
        for box in metadata["rois"]:
            if (len(box) != 4 or any(type(value) is not int for value in box)
                    or not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height)):
                raise ValueError("Invalid C4 rectangle in model metadata.")
        path = model_dir / "patchcore.pt"
        if sha256_file(path) != metadata["model_sha256"]:
            raise ValueError("C4 model checksum mismatch. Restore the supplied bundle.")
        self.aligner = None
        if metadata.get("alignment_sha256"):
            alignment_path = model_dir / "alignment.json"
            if sha256_file(alignment_path) != metadata["alignment_sha256"]:
                raise ValueError("Alignment reference checksum mismatch. Restore the supplied bundle.")
            self.aligner = C4Aligner(alignment_path)
        self.model = torch.jit.load(str(path), map_location="cpu").eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def locate_unit(self, image, roi, unit):
        if self.aligner is not None:
            return self.aligner.locate(image, roi, unit)
        return list(roi), {"method": "fixed_roi", "offset_xy": [0, 0], "reference_roi_xyxy": list(roi)}

    def read_vid(self, image, roi):
        return read_unit_vid(image, roi)

    def tensor(self, image):
        torch = self.torch
        array = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).float().div_(255)
        tensor = torch.nn.functional.interpolate(tensor, size=self.metadata["input_size"],
                                                mode="bilinear", align_corners=False, antialias=True)
        return (tensor - self.mean) / self.std

    def predict(self, image):
        with self.torch.inference_mode():
            score, anomaly_map = self.model(self.tensor(image))
        score = float(score.item())
        if (not math.isfinite(score) or tuple(anomaly_map.shape) != (1, 1, *self.metadata["input_size"])
                or not self.torch.isfinite(anomaly_map).all()):
            raise ValueError("Model returned an invalid score or anomaly map.")
        return score, anomaly_map[0, 0].cpu().numpy().copy()


@dataclass
class Inspection:
    record: dict
    image: Image.Image
    units: list
    source_bytes: bytes


def inspect_frame(path, predictor):
    path = Path(path).resolve()
    start = time.monotonic()
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise OSError("Image changed while being read; retry after writing finishes.")
    with Image.open(io.BytesIO(raw)) as opened:
        if getattr(opened, "n_frames", 1) != 1:
            raise ValueError("Multi-page images are unsupported.")
        if list(opened.size) != predictor.metadata["frame_size_wh"]:
            raise ValueError(f"Expected {predictor.metadata['frame_size_wh']} pixels; got {list(opened.size)}.")
        opened.load()
        image = opened.convert("RGB")
    threshold = predictor.metadata["raw_threshold"]
    rows, units = [], []
    for unit, roi in enumerate(predictor.metadata["rois"], 1):
        aligned_roi, alignment = predictor.locate_unit(image, roi, unit)
        identity = predictor.read_vid(image, roi)
        crop = image.crop(aligned_roi)
        score, raw_map = predictor.predict(crop)
        if not math.isfinite(score) or not np.isfinite(raw_map).all():
            raise ValueError("Non-finite inference output; frame has not passed inspection.")
        rows.append({"unit": unit, "roi_xyxy": aligned_roi, "reference_roi_xyxy": list(roi),
                     "alignment": alignment, **identity, "raw_score": score, "triggered": score > threshold,
                     "model_result": "Bad" if score > threshold else "Good"})
        units.append((crop, raw_map))
    identifiers = [row["vid"] for row in rows if row.get("vid")]
    for row in rows:
        if row.get("vid") and identifiers.count(row["vid"]) > 1:
            row["vid_status"] = "DUPLICATE"
        row["artifact_name"] = row["vid"] if row["vid_status"] == "READ" else f"REVIEW-position_{row['unit']}"
        row["result"] = row["model_result"] if row["vid_status"] == "READ" else "Review"
    tray_result = "Review" if any(row["result"] == "Review" for row in rows) else (
        "Bad" if any(row["triggered"] for row in rows) else "Good")
    record = {"source_path": str(path), "source_sha256": hashlib.sha256(raw).hexdigest(),
              "source_size": after.st_size, "source_mtime_ns": after.st_mtime_ns,
              "model_sha256": predictor.metadata.get("model_sha256"),
              "alignment_sha256": predictor.metadata.get("alignment_sha256"), "threshold_raw": threshold,
              "tray_result": tray_result, "vids_left_to_right": [row.get("vid") for row in rows],
              "triggered": any(row["triggered"] for row in rows), "units": rows,
              "elapsed_seconds": time.monotonic() - start}
    return Inspection(record, image, units, raw)


def save_inspection(inspection, session_dir, metadata):
    import tempfile
    record = inspection.record
    identity = hashlib.sha256((record["source_path"] + record["source_sha256"]
                               + str(record["source_mtime_ns"])).encode()).hexdigest()[:24]
    destination = Path(session_dir) / "frames" / identity
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Inspection already exists: {destination}")
    with tempfile.TemporaryDirectory(prefix=".pending_", dir=destination.parent) as temporary:
        folder = Path(temporary)
        preview = inspection.image.copy()
        draw = ImageDraw.Draw(preview)
        for row in record["units"]:
            color = "#b36b00" if row["result"] == "Review" else ("#d6273f" if row["triggered"] else "#19855a")
            draw.rectangle(row["roi_xyxy"], outline=color, width=10)
            left, top, _, _ = row["roi_xyxy"]
            draw.text((left, top - 24), f"{row.get('vid') or row['vid_status']}: {row['result']} {row['raw_score']:.4f}",
                      fill=color, font_size=20)
        preview.thumbnail((1325, 444))
        preview.save(folder / "preview.jpg")
        if record["triggered"]:
            (folder / ("original" + Path(record["source_path"]).suffix.lower())).write_bytes(inspection.source_bytes)
        for row, (crop, raw_map) in zip(record["units"], inspection.units):
            if not row["triggered"]:
                continue
            name = row["artifact_name"]
            crop.save(folder / f"{name}.png")
            _, overlay, _ = render_heatmap(crop, raw_map, metadata["pixel_min"], metadata["pixel_max"])
            overlay.save(folder / f"{name}_overlay.jpg")
            save_map(folder / f"{name}.npz", raw_map, roi=row["roi_xyxy"], source_size=inspection.image.size,
                     display_min=metadata["pixel_min"], display_max=metadata["pixel_max"],
                     score=row["raw_score"], raw_score=row["raw_score"], threshold=record["threshold_raw"])
        tray_folder = Path(session_dir).parent / "Result" / record["tray_result"]
        tray_folder.mkdir(parents=True, exist_ok=True)
        stem = "_".join(row["artifact_name"] for row in record["units"])
        suffix = Path(record["source_path"]).suffix.lower()
        sequence = 0
        while True:
            collision = "" if sequence == 0 else f"__{identity}_{sequence}"
            tray_path = tray_folder / (stem + collision + suffix)
            sidecar = tray_path.with_suffix(".json")
            if sidecar.exists():
                sequence += 1
                continue
            try:
                with tray_path.open("xb") as stream:
                    try:
                        stream.write(inspection.source_bytes)
                        stream.flush()
                        os.fsync(stream.fileno())
                    except OSError:
                        stream.close()
                        tray_path.unlink(missing_ok=True)
                        raise
                break
            except FileExistsError:
                sequence += 1
        record = {**record, "tray_image_path": str(tray_path), "result_dir": str(destination),
                  "preview_path": str(destination / "preview.jpg")}
        serialized = json.dumps(record, indent=2, allow_nan=False)
        (folder / "result.json").write_text(serialized, encoding="utf-8")
        with sidecar.open("x", encoding="utf-8") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(folder, destination)
    return record