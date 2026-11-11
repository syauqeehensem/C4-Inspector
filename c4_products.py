"""Product identification and frame-bounded unit geometry for mixed camera folders."""
import numpy as np


def layout_profiles(image, downscale):
    import cv2

    pixels = np.asarray(image.convert("L"), dtype=np.float32)
    small = cv2.resize(pixels, (pixels.shape[1] // downscale, pixels.shape[0] // downscale),
                       interpolation=cv2.INTER_AREA)
    return small.mean(axis=0), small.mean(axis=1)


def profile_match(reference, observed, maximum):
    reference = np.asarray(reference, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    if (reference.ndim != 1 or reference.shape != observed.shape or reference.size <= 2 * maximum + 4
            or not np.isfinite(reference).all() or not np.isfinite(observed).all()):
        raise ValueError("Invalid layout profile.")
    template = reference[maximum:reference.size - maximum]
    template = template - template.mean()
    best = -1.0
    for offset in range(-maximum, maximum + 1):
        segment = observed[maximum + offset:observed.size - maximum + offset]
        segment = segment - segment.mean()
        denominator = np.linalg.norm(template) * np.linalg.norm(segment)
        if denominator > 1e-8:
            best = max(best, float(np.dot(template, segment) / denominator))
    return best


def layout_score(layout, columns, rows, maximum):
    # EMRXCP row profiles vary between frames, so neither axis alone decides the product.
    return (profile_match(layout["x"], columns, maximum) + profile_match(layout["y"], rows, maximum)) / 2


def identify_product(image, products, detection):
    columns, rows = layout_profiles(image, detection["downscale"])
    scores = {product["name"]: layout_score(product["layout"], columns, rows, detection["maximum_shift"])
              for product in products}
    ranked = sorted(scores.values(), reverse=True) + [-1.0]
    if ranked[0] < detection["minimum_correlation"] or ranked[0] - ranked[1] < detection["minimum_margin"]:
        summary = ", ".join(f"{name} {value:.3f}" for name, value in scores.items())
        raise ValueError(f"Unknown product layout ({summary}); frame not inspected.")
    return max(scores, key=scores.get), scores


def shift_box(box, offset):
    return [box[0] + offset[0], box[1] + offset[1], box[2] + offset[0], box[3] + offset[1]]


def fit_box(box, size):
    left, top, right, bottom = box
    width, height = size
    if right - left > width or bottom - top > height:
        raise ValueError("Inspection rectangle is larger than the frame.")
    dx = -left if left < 0 else min(0, width - right)
    dy = -top if top < 0 else min(0, height - bottom)
    return [left + dx, top + dy, right + dx, bottom + dy]
