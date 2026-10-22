"""Read the operator-confirmed lower-left unit Data Matrix."""
import re

import numpy as np
from PIL import Image


def read_unit_vid(image, roi):
    import zxingcpp

    left, _, _, bottom = roi
    box = (left - 270, bottom - 320, left + 60, bottom + 260)
    if not (0 <= box[0] < box[2] <= image.width and 0 <= box[1] < box[3] <= image.height):
        return {"vid": None, "vid_status": "OUT_OF_BOUNDS", "vid_roi_xyxy": list(box)}
    crop = image.crop(box).convert("L")
    payloads = set()
    for scale in (1, 2, 4):
        resized = crop.resize((crop.width * scale, crop.height * scale), Image.Resampling.NEAREST)
        symbols = zxingcpp.read_barcodes(np.asarray(resized), formats=zxingcpp.BarcodeFormat.DataMatrix,
                                        try_rotate=True, try_downscale=True, try_invert=True)
        payloads.update(bytes(symbol.bytes) for symbol in symbols if symbol.valid)
    result = {"vid": None, "vid_status": "UNREADABLE", "vid_roi_xyxy": list(box),
              "vid_payloads_hex": [payload.hex() for payload in sorted(payloads)]}
    if len(payloads) > 1:
        result["vid_status"] = "AMBIGUOUS"
    elif payloads:
        try:
            text = next(iter(payloads)).decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            result["vid_status"] = "INVALID_ENCODING"
        else:
            reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{number}" for number in range(1, 10)),
                        *(f"LPT{number}" for number in range(1, 10))}
            if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]{0,63}", text) and text.upper() not in reserved:
                result.update(vid=text, vid_status="READ")
            else:
                result["vid_status"] = "INVALID_FILENAME"
    return result