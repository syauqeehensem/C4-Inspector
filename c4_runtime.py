"""CPU-only multi-product inference and inspection evidence; no training imports."""
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
from c4_identity import read_vid
from c4_products import fit_box, identify_product, shift_box
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
        if metadata.get("format") not in ("iom-c4-raw-v1", "iom-area-raw-v1"):
            raise ValueError("Select an exported inspection model, not a training checkpoint or bonded model.")
        for key in ("raw_threshold", "pixel_min", "pixel_max"):
            if not isinstance(metadata.get(key), (int, float)) or not math.isfinite(metadata[key]):
                raise ValueError(f"Invalid model calibration: {key}")
        if metadata["raw_threshold"] <= 0 or metadata["pixel_max"] <= metadata["pixel_min"]:
            raise ValueError("Invalid model threshold or heatmap display bounds.")
        for key in ("input_size", "frame_size_wh"):
            if len(metadata[key]) != 2 or any(type(value) is not int or value <= 0 for value in metadata[key]):
                raise ValueError(f"Invalid model dimensions: {key}")
        width, height = metadata["frame_size_wh"]
        if not metadata.get("rois"):
            raise ValueError("Model must define unit rectangles.")
        for box in metadata["rois"]:
            if (len(box) != 4 or any(type(value) is not int for value in box)
                    or not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height)):
                raise ValueError("Invalid unit rectangle in model metadata.")
        path = model_dir / "patchcore.pt"
        if sha256_file(path) != metadata["model_sha256"]:
            raise ValueError("Model checksum mismatch. Restore the supplied bundle.")
        self.model = torch.jit.load(str(path), map_location="cpu").eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

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


def bundle_path(bundle, relative):
    path = (bundle / relative).resolve()
    if not path.is_relative_to(bundle):
        raise ValueError("Model bundle path escapes its folder.")
    return path


class ProductPredictor:
    def __init__(self, bundle_dir, threads=4):
        bundle = Path(bundle_dir).resolve()
        self.metadata = json.loads((bundle / "products.json").read_text(encoding="utf-8"))
        metadata = self.metadata
        detection = metadata.get("detection") or {}
        if metadata.get("format") != "iom-inspector-products-v1" or not metadata.get("products"):
            raise ValueError("Select the supplied multi-product model folder.")
        if (not all(type(detection.get(key)) is int and detection[key] >= 0 for key in ("downscale", "maximum_shift"))
                or not all(isinstance(detection.get(key), (int, float)) and math.isfinite(detection[key])
                           for key in ("minimum_correlation", "minimum_margin"))):
            raise ValueError("Invalid product detection settings.")
        self.products = {}
        for product in metadata["products"]:
            name = product["name"]
            if not product.get("units") or not product.get("areas"):
                raise ValueError(f"{name} must define units and inspection areas.")
            alignment = bundle_path(bundle, product["alignment"])
            if sha256_file(alignment) != product["alignment_sha256"]:
                raise ValueError(f"{name} alignment checksum mismatch. Restore the supplied bundle.")
            aligner = C4Aligner(alignment)
            anchors = [unit["anchor_xyxy"] for unit in product["units"]]
            if [unit["roi_xyxy"] for unit in aligner.config["units"]] != anchors:
                raise ValueError(f"{name} alignment does not match its unit geometry.")
            areas = {}
            for area in product["areas"]:
                if area.get("mode", "alert") not in ("alert", "monitor"):
                    raise ValueError(f"{name} {area['name']} has an unsupported mode.")
                folder = bundle_path(bundle, area["model"])
                if sha256_file(folder / "metadata.json") != area["metadata_sha256"]:
                    raise ValueError(f"{name} {area['name']} metadata checksum mismatch. Restore the supplied bundle.")
                model = Predictor(folder, threads)
                if (len(model.metadata["rois"]) != len(anchors)
                        or model.metadata["frame_size_wh"] != metadata["frame_size_wh"]):
                    raise ValueError(f"{name} {area['name']} model does not match the product layout.")
                areas[area["name"]] = (model, area.get("mode", "alert"))
                self.torch = model.torch
            if not any(mode == "alert" for _, mode in areas.values()):
                raise ValueError(f"{name} needs at least one alerting inspection area.")
            self.products[name] = (product, aligner, areas)

    def detect(self, image):
        return identify_product(image, [entry[0] for entry in self.products.values()], self.metadata["detection"])

    def units(self, product):
        return self.products[product][0]["units"]

    def areas(self, product):
        return [(name, {**model.metadata, "mode": mode}) for name, (model, mode) in self.products[product][2].items()]

    def locate_unit(self, image, product, unit):
        _, record = self.products[product][1].locate(image, self.units(product)[unit - 1]["anchor_xyxy"], unit)
        return record["offset_xy"], record

    def read_vid(self, image, box):
        return read_vid(image, box)

    def predict(self, product, area, crop):
        return self.products[product][2][area][0].predict(crop)


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
    product, product_scores = predictor.detect(image)
    rows, crops = [], []
    for index, unit in enumerate(predictor.units(product), 1):
        try:
            offset, alignment = predictor.locate_unit(image, product, index)
            alignment = {**alignment, "status": "ALIGNED"}
        except ValueError as error:
            offset, alignment = (0, 0), {"status": "FAILED", "error": str(error), "offset_xy": [0, 0]}
        identity = predictor.read_vid(image, shift_box(unit["vid_xyxy"], offset))
        areas, unit_crops = [], []
        for name, metadata in predictor.areas(product):
            reference = metadata["rois"][index - 1]
            box = fit_box(shift_box(reference, offset), image.size)
            crop = image.crop(box)
            score, raw_map = predictor.predict(product, name, crop)
            if not math.isfinite(score) or not np.isfinite(raw_map).all():
                raise ValueError("Non-finite inference output; frame has not passed inspection.")
            threshold = metadata["raw_threshold"]
            mode = metadata.get("mode", "alert")
            areas.append({"area": name, "mode": mode, "roi_xyxy": box, "reference_roi_xyxy": list(reference),
                          "raw_score": score, "threshold_raw": threshold, "score_ratio": score / threshold,
                          "above_threshold": score > threshold, "triggered": score > threshold and mode == "alert",
                          "model_sha256": metadata.get("model_sha256")})
            unit_crops.append((crop, raw_map, metadata))
        triggered = any(area["triggered"] for area in areas)
        rows.append({"unit": index, "alignment": alignment, **identity, "areas": areas,
                     "score_ratio": max(area["score_ratio"] for area in areas if area["mode"] == "alert"),
                     "triggered": triggered, "model_result": "Bad" if triggered else "Good"})
        crops.append(unit_crops)
    identifiers = [row["vid"] for row in rows if row.get("vid")]
    for row in rows:
        if row.get("vid") and identifiers.count(row["vid"]) > 1:
            row["vid_status"] = "DUPLICATE"
        row["artifact_name"] = row["vid"] if row["vid_status"] == "READ" else f"REVIEW-position_{row['unit']}"
        review = row["vid_status"] != "READ" or row["alignment"]["status"] != "ALIGNED"
        row["result"] = "Review" if review else row["model_result"]
    tray_result = "Review" if any(row["result"] == "Review" for row in rows) else (
        "Bad" if any(row["triggered"] for row in rows) else "Good")
    record = {"source_path": str(path), "source_sha256": hashlib.sha256(raw).hexdigest(),
              "source_size": after.st_size, "source_mtime_ns": after.st_mtime_ns,
              "product": product, "product_scores": product_scores, "tray_result": tray_result,
              "vids_in_order": [row.get("vid") for row in rows],
              "triggered": any(row["triggered"] for row in rows), "units": rows,
              "elapsed_seconds": time.monotonic() - start}
    return Inspection(record, image, crops, raw)


def save_inspection(inspection, session_dir):
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
            for area in row["areas"]:
                draw.rectangle(area["roi_xyxy"], outline="#d6273f" if area["triggered"] else color, width=10)
            left, top = max((area["roi_xyxy"] for area in row["areas"]),
                            key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))[:2]
            draw.text((left + 18, top + 14), f"{row.get('vid') or row['vid_status']}\n{row['result']} {row['score_ratio']:.2f}",
                      fill=color, font_size=56)
        preview.thumbnail((1325, 444))
        preview.save(folder / "preview.jpg")
        if record["triggered"]:
            (folder / ("original" + Path(record["source_path"]).suffix.lower())).write_bytes(inspection.source_bytes)
        for row, unit_crops in zip(record["units"], inspection.units, strict=True):
            for area, (crop, raw_map, metadata) in zip(row["areas"], unit_crops, strict=True):
                if not area["triggered"]:
                    continue
                name = f"{row['artifact_name']}_{area['area']}"
                crop.save(folder / f"{name}.png")
                _, overlay, _ = render_heatmap(crop, raw_map, metadata["pixel_min"], metadata["pixel_max"])
                overlay.save(folder / f"{name}_overlay.jpg")
                save_map(folder / f"{name}.npz", raw_map, roi=area["roi_xyxy"], source_size=inspection.image.size,
                         display_min=metadata["pixel_min"], display_max=metadata["pixel_max"],
                         score=area["raw_score"], raw_score=area["raw_score"], threshold=area["threshold_raw"])
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