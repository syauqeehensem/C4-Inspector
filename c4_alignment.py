"""Bounded integer-pixel registration using die-boundary gradient profiles."""
import json

import numpy as np


def edge_profiles(image, roi, margin=80):
    import cv2

    left, top, right, bottom = roi
    box = (left - margin, top - margin, right + margin, bottom + margin)
    if not (0 <= box[0] < box[2] <= image.width and 0 <= box[1] < box[3] <= image.height):
        raise ValueError("Alignment search area is outside the image.")
    pixels = np.asarray(image.crop(box).convert("L"), dtype=np.float32) / 255
    pixels = cv2.GaussianBlur(pixels, (5, 5), 0)
    gradient_x = np.abs(cv2.Sobel(pixels, cv2.CV_32F, 1, 0, ksize=3))
    gradient_y = np.abs(cv2.Sobel(pixels, cv2.CV_32F, 0, 1, ksize=3))
    band = margin + 100
    return gradient_x[band:-band, :].mean(axis=0), gradient_y[:, band:-band].mean(axis=1)


def axis_offset(reference, observed, maximum):
    reference = np.asarray(reference, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    if reference.shape != observed.shape or reference.size <= 2 * maximum + 4:
        raise ValueError("Invalid alignment profile dimensions.")
    if not np.isfinite(reference).all() or not np.isfinite(observed).all():
        raise ValueError("Non-finite alignment profile.")
    template = reference[maximum:-maximum]
    template = template - template.mean()
    norm = np.linalg.norm(template)
    if norm < 1e-8:
        raise ValueError("Alignment reference has no usable edges.")
    candidates = []
    for offset in range(-maximum, maximum + 1):
        segment = observed[maximum + offset:observed.size - maximum + offset]
        segment = segment - segment.mean()
        denominator = norm * np.linalg.norm(segment)
        score = float(np.dot(template, segment) / denominator) if denominator > 1e-8 else -1.0
        candidates.append((score, offset))
    score, offset = max(candidates)
    runner_up = max(value for value, shift in candidates if abs(shift - offset) > 3)
    if score < 0.70 or score - runner_up < 0.01 or abs(offset) >= maximum:
        raise ValueError(f"Uncertain alignment: correlation={score:.4f}, shift={offset}.")
    return offset, score


class C4Aligner:
    def __init__(self, path):
        self.config = json.loads(path.read_text(encoding="utf-8"))
        if self.config.get("format") != "iom-c4-edge-profiles-v1":
            raise ValueError("Unsupported alignment reference.")
        if self.config.get("maximum_shift") != 64 or self.config.get("margin") != 80:
            raise ValueError("Unsupported alignment search limits.")
        if len(self.config.get("units", [])) != 3:
            raise ValueError("Alignment requires three unit profiles.")

    def locate(self, image, roi, unit):
        reference = self.config["units"][unit - 1]
        if list(roi) != reference["roi_xyxy"]:
            raise ValueError("Alignment reference does not match model geometry.")
        _, observed_y = edge_profiles(image, roi)
        vertical, quality_y = axis_offset(reference["y"], observed_y, 64)
        left, top, right, bottom = roi
        observed_x, _ = edge_profiles(image, [left, top + vertical, right, bottom + vertical])
        horizontal, quality_x = axis_offset(reference["x"], observed_x, 64)
        aligned = [left + horizontal, top + vertical, right + horizontal, bottom + vertical]
        if not (0 <= aligned[0] < aligned[2] <= image.width and 0 <= aligned[1] < aligned[3] <= image.height):
            raise ValueError("Aligned C4 rectangle is outside the image.")
        return aligned, {"method": "integer_boundary_profiles", "offset_xy": [horizontal, vertical],
                         "correlation_xy": [quality_x, quality_y], "reference_roi_xyxy": list(roi)}