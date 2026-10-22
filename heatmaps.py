"""Render measured anomaly maps and preserve their values and spatial reference."""
import numpy as np
from PIL import Image


def render_heatmap(image, raw_map, display_min, display_max):
    """Use a fixed model-wide scale; never stretch each image's contrast separately."""
    raw_map = np.asarray(raw_map, dtype=np.float32)
    if raw_map.ndim != 2 or not raw_map.size or not np.isfinite(raw_map).all():
        raise ValueError("Model returned an invalid anomaly map.")
    if not np.isfinite([display_min, display_max]).all() or display_max <= display_min:
        raise ValueError("Model has invalid heatmap display bounds.")
    # Maps use the same unrotated image axes as the crop. Resize scalar values
    # before applying color so the overlay stays aligned with native crop pixels.
    native = np.asarray(Image.fromarray(raw_map).resize(image.size, Image.Resampling.BILINEAR))
    intensity = np.clip((native - display_min) / (display_max - display_min), 0, 1)
    stops = np.array([0, 1 / 3, 2 / 3, 1])
    colors = np.array([[0, 0, 255], [0, 255, 255], [255, 255, 0], [255, 0, 0]])
    rgb = np.stack([np.interp(intensity, stops, colors[:, channel]) for channel in range(3)], axis=-1)
    heatmap = Image.fromarray(np.rint(rgb).astype(np.uint8))
    overlay = Image.blend(image.convert("RGB"), heatmap, 0.45)
    peak_y, peak_x = np.unravel_index(np.argmax(native), native.shape)
    return heatmap, overlay, {"peak_crop_x": int(peak_x), "peak_crop_y": int(peak_y),
                              "peak_raw_value": float(native[peak_y, peak_x])}


def save_map(path, raw_map, *, roi, source_size, display_min, display_max, score, raw_score, threshold):
    """Lossless model-resolution float32 data, with everything needed to align it."""
    np.savez_compressed(
        path, schema_version=np.int32(1), raw_anomaly_map=np.asarray(raw_map, dtype=np.float32),
        roi_xyxy=np.asarray(roi, dtype=np.int64), source_size_wh=np.asarray(source_size, dtype=np.int64),
        crop_size_wh=np.asarray([roi[2] - roi[0], roi[3] - roi[1]], dtype=np.int64),
        display_min=np.float32(display_min), display_max=np.float32(display_max),
        score=np.float64(score), raw_score=np.float64(raw_score), image_threshold=np.float64(threshold),
    )
