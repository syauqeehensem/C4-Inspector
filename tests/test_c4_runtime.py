import csv
import json
import queue
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from c4_alignment import C4Aligner, axis_offset, edge_profiles
from c4_identity import read_vid
from c4_monitor import FileInbox, InspectionWorker, validate_folders
from c4_products import fit_box, identify_product, layout_profiles
from c4_runtime import inspect_frame, save_inspection


class FakePredictor:
    def __init__(self, scores=(10, 11, 9), units=3, areas=("Unit",)):
        self.scores = iter(scores)
        boxes = [[4 * index, 0, 4 * index + 4, 4] for index in range(units)]
        self.names = ("VIDLEFT", "VIDMIDDLE", "VIDRIGHT") if units == 3 else [f"VID{index}" for index in range(1, units + 1)]
        self.unit_list = [{"anchor_xyxy": box, "vid_xyxy": box} for box in boxes]
        self.area_list = [(name, {"rois": boxes, "raw_threshold": 10, "pixel_min": 0, "pixel_max": 20,
                                  "model_sha256": "test"}) for name in areas]
        self.metadata = {"frame_size_wh": [4 * units, 4], "products": [{"name": "FAKE"}]}

    def detect(self, image):
        return "FAKE", {"FAKE": 1.0}

    def units(self, product):
        return self.unit_list

    def areas(self, product):
        return self.area_list

    def locate_unit(self, image, product, unit):
        return [0, 0], {"method": "fixed_roi", "offset_xy": [0, 0]}

    def read_vid(self, image, box):
        return {"vid": self.names[box[0] // 4], "vid_status": "READ"}

    def predict(self, product, area, crop):
        return next(self.scores), np.ones((4, 4), dtype=np.float32)


class C4RuntimeTests(unittest.TestCase):
    def test_product_detection_requires_a_clear_layout_match(self):
        image = Image.new("L", (400, 200))
        image.paste(255, (40, 20, 180, 120))
        image.paste(160, (220, 70, 360, 180))
        band = Image.new("L", (400, 200))
        band.paste(255, (0, 60, 400, 140))
        band.paste(128, (300, 0, 340, 200))
        products = []
        for name, source in (("TWO", image), ("BAND", band)):
            columns, rows = layout_profiles(source, 10)
            products.append({"name": name, "layout": {"x": columns.tolist(), "y": rows.tolist()}})
        detection = {"downscale": 10, "maximum_shift": 2, "minimum_correlation": 0.55, "minimum_margin": 0.15}
        shifted = Image.fromarray(np.roll(np.asarray(image), 10, axis=1))
        self.assertEqual(identify_product(shifted, products, detection)[0], "TWO")
        with self.assertRaisesRegex(ValueError, "Unknown product"):
            identify_product(Image.new("L", (400, 200), 128), products, detection)

    def test_fit_box_keeps_inspection_size_inside_frame(self):
        self.assertEqual(fit_box([-5, 1760, 95, 1790], (5300, 1775)), [0, 1745, 100, 1775])
        self.assertEqual(fit_box([10, 10, 20, 20], (100, 100)), [10, 10, 20, 20])
        with self.assertRaises(ValueError):
            fit_box([0, 0, 200, 10], (100, 100))
    def test_alignment_resolves_vertical_shift_before_horizontal_boundary_match(self):
        image = Image.new("L", (700, 700))
        image.paste(220, (220, 220, 480, 480))
        roi = [200, 200, 500, 500]
        horizontal, vertical = edge_profiles(image, roi)
        config = {"format": "iom-c4-edge-profiles-v1", "maximum_shift": 64, "margin": 80,
                  "units": [{"roi_xyxy": roi, "x": horizontal.tolist(), "y": vertical.tolist()}] * 3}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "alignment.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            shifted = Image.fromarray(np.roll(np.roll(np.asarray(image), 50, axis=0), -7, axis=1))
            aligned, record = C4Aligner(path).locate(shifted, roi, 1)
        self.assertEqual(aligned, [193, 250, 493, 550])
        self.assertEqual(record["offset_xy"], [-7, 50])

    def test_integer_alignment_recovers_shift_and_rejects_blank_or_out_of_limit_profiles(self):
        generator = np.random.default_rng(42)
        reference = generator.normal(size=500)
        for shift in (-30, 0, 23):
            offset, correlation = axis_offset(reference, np.roll(reference, shift), 64)
            self.assertEqual(offset, shift)
            self.assertAlmostEqual(correlation, 1)
        with self.assertRaisesRegex(ValueError, "Uncertain"):
            axis_offset(reference, np.zeros(500), 64)
        with self.assertRaisesRegex(ValueError, "Uncertain"):
            axis_offset(reference, np.roll(reference, 64), 64)
        with self.assertRaisesRegex(ValueError, "reference"):
            axis_offset(np.zeros(500), reference, 64)

    def test_vid_decodes_inside_product_box_and_unsafe_or_ambiguous_values_never_become_identifiers(self):
        image = Image.new("RGB", (5300, 1775), "white")
        box = [3495, 800, 3825, 1380]
        symbol = SimpleNamespace(valid=True, bytes=b"TESTVID000001")
        with patch("zxingcpp.read_barcodes", return_value=[symbol]) as decoder:
            result = read_vid(image, box)
        self.assertEqual(result["vid"], "TESTVID000001")
        self.assertEqual(result["vid_status"], "READ")
        self.assertEqual(result["vid_roi_xyxy"], box)
        self.assertEqual(decoder.call_args_list[0].args[0].shape, (580, 330))
        for payload in (b"../escape", b"=FORMULA", b"", b"bad\xff", b"CON", b"LPT9"):
            with patch("zxingcpp.read_barcodes", return_value=[SimpleNamespace(valid=True, bytes=payload)]):
                self.assertIsNone(read_vid(image, box)["vid"])
        with patch("zxingcpp.read_barcodes", return_value=[symbol, SimpleNamespace(valid=True, bytes=b"OTHER")]):
            self.assertEqual(read_vid(image, box)["vid_status"], "AMBIGUOUS")
        self.assertEqual(read_vid(image, [-270, -310, 60, 270])["vid_status"], "OUT_OF_BOUNDS")
        with patch("zxingcpp.read_barcodes", side_effect=[[], [], [], [symbol], []]) as decoder:
            self.assertEqual(read_vid(image, box)["vid"], "TESTVID000001")
        self.assertEqual(decoder.call_count, 5)
        with patch("zxingcpp.read_barcodes", return_value=[]):
            self.assertEqual(read_vid(image, box)["vid_status"], "UNREADABLE")

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.path = self.root / "frame.png"
        Image.new("RGB", (12, 4), "blue").save(self.path)

    def test_raw_threshold_is_strict_and_any_unit_triggers_frame(self):
        result = inspect_frame(self.path, FakePredictor())
        self.assertEqual([row["triggered"] for row in result.record["units"]], [False, True, False])
        self.assertTrue(result.record["triggered"])
        self.assertEqual(result.image.getpixel((0, 0)), (0, 0, 255))

    def test_wrong_dimensions_corrupt_images_and_nonfinite_outputs_never_pass(self):
        Image.new("RGB", (4, 4)).save(self.path)
        with self.assertRaisesRegex(ValueError, "Expected"):
            inspect_frame(self.path, FakePredictor())
        self.path.write_bytes(b"incomplete camera file")
        with self.assertRaises(OSError):
            inspect_frame(self.path, FakePredictor())
        Image.new("RGB", (12, 4)).save(self.path)
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            inspect_frame(self.path, FakePredictor((float("nan"), 2, 3)))

    def test_saved_evidence_preserves_original_pixels_maps_and_coordinates(self):
        original_bytes = self.path.read_bytes()
        predictor = FakePredictor()
        result = save_inspection(inspect_frame(self.path, predictor), self.root / "session")
        folder = Path(result["result_dir"])
        self.assertEqual((folder / "original.png").read_bytes(), original_bytes)
        self.assertEqual(self.path.read_bytes(), original_bytes)
        self.assertTrue((folder / "VIDMIDDLE_Unit_overlay.jpg").exists())
        self.assertFalse((folder / "VIDLEFT_Unit.png").exists())
        with np.load(folder / "VIDMIDDLE_Unit.npz") as data:
            self.assertEqual(data["roi_xyxy"].tolist(), [4, 0, 8, 4])
        self.assertTrue(json.loads((folder / "result.json").read_text())["triggered"])
        tray = Path(result["tray_image_path"])
        self.assertEqual(tray.name, "VIDLEFT_VIDMIDDLE_VIDRIGHT.png")
        self.assertEqual(tray.parent, self.root / "Result" / "Bad")
        self.assertEqual(tray.read_bytes(), original_bytes)
        self.assertEqual(json.loads(tray.with_suffix(".json").read_text())["vids_in_order"],
                 ["VIDLEFT", "VIDMIDDLE", "VIDRIGHT"])

    def test_eight_unit_tray_name_follows_configured_unit_order(self):
        path = self.root / "eight.png"
        Image.new("RGB", (32, 4), "gray").save(path)
        result = save_inspection(inspect_frame(path, FakePredictor([5] * 8, units=8)), self.root / "session")
        self.assertEqual(Path(result["tray_image_path"]).name, "_".join(f"VID{index}" for index in range(1, 9)) + ".png")
        self.assertEqual(result["tray_result"], "Good")

    def test_any_area_trigger_marks_unit_bad_and_names_evidence_by_vid_and_area(self):
        predictor = FakePredictor((5, 5, 5, 11, 5, 5), areas=("C4", "Unit"))
        result = save_inspection(inspect_frame(self.path, predictor), self.root / "session")
        self.assertEqual([row["result"] for row in result["units"]], ["Good", "Bad", "Good"])
        self.assertAlmostEqual(result["units"][1]["score_ratio"], 1.1)
        folder = Path(result["result_dir"])
        self.assertTrue((folder / "VIDMIDDLE_Unit_overlay.jpg").exists())
        self.assertFalse((folder / "VIDMIDDLE_C4_overlay.jpg").exists())

    def test_monitor_area_records_score_without_alert_or_bad_routing(self):
        predictor = FakePredictor((5, 5, 5, 11, 5, 5), areas=("C4", "Unit"))
        predictor.area_list[1][1]["mode"] = "monitor"
        result = save_inspection(inspect_frame(self.path, predictor), self.root / "session")
        middle = result["units"][1]
        self.assertFalse(result["triggered"])
        self.assertEqual(result["tray_result"], "Good")
        self.assertTrue(middle["areas"][1]["above_threshold"])
        self.assertFalse(middle["areas"][1]["triggered"])
        self.assertAlmostEqual(middle["score_ratio"], 0.5)
        self.assertFalse((Path(result["result_dir"]) / "VIDMIDDLE_Unit_overlay.jpg").exists())

    def test_untriggered_frame_has_no_defect_evidence_and_existing_result_is_not_overwritten(self):
        predictor = FakePredictor((10, 10, 10))
        inspection = inspect_frame(self.path, predictor)
        result = save_inspection(inspection, self.root / "session")
        self.assertFalse(result["triggered"])
        self.assertFalse((Path(result["result_dir"]) / "original.png").exists())
        self.assertEqual(Path(result["tray_image_path"]).parent.name, "Good")
        with self.assertRaises(FileExistsError):
            save_inspection(inspection, self.root / "session")

    def test_repeat_tray_never_overwrites_and_unreadable_or_duplicate_vid_is_review(self):
        predictor = FakePredictor((10, 10, 10))
        inspection = inspect_frame(self.path, predictor)
        first = save_inspection(inspection, self.root / "session1")
        second = save_inspection(inspection, self.root / "session2")
        self.assertNotEqual(first["tray_image_path"], second["tray_image_path"])
        self.assertEqual(Path(first["tray_image_path"]).read_bytes(), self.path.read_bytes())
        for identity in ({"vid": None, "vid_status": "UNREADABLE"}, {"vid": "SAMEVID", "vid_status": "READ"}):
            predictor = FakePredictor((10, 10, 10))
            predictor.read_vid = lambda image, box, identity=identity: dict(identity)
            inspection = inspect_frame(self.path, predictor)
            self.assertEqual(inspection.record["tray_result"], "Review")
            result = save_inspection(inspection, self.root / identity["vid_status"])
            self.assertEqual(Path(result["tray_image_path"]).parent.name, "Review")

    def test_alignment_failure_routes_tray_to_review_never_good(self):
        predictor = FakePredictor((5, 5, 5))
        predictor.locate_unit = lambda *args: (_ for _ in ()).throw(ValueError("Uncertain alignment"))
        inspection = inspect_frame(self.path, predictor)
        self.assertEqual(inspection.record["tray_result"], "Review")
        self.assertEqual({row["alignment"]["status"] for row in inspection.record["units"]}, {"FAILED"})
        result = save_inspection(inspection, self.root / "session")
        self.assertEqual(Path(result["tray_image_path"]).parent.name, "Review")

    def test_monitor_waits_for_stability_ignores_existing_and_detects_replacements(self):
        inbox = FileInbox(self.root, include_existing=False, stable_seconds=2)
        self.assertEqual(inbox.scan(now=0), [])
        Image.new("RGB", (12, 4), "red").save(self.path)
        self.assertEqual(inbox.scan(now=1), [])
        self.assertEqual(inbox.scan(now=2), [])
        ready = inbox.scan(now=3)
        self.assertEqual(len(ready), 1)
        inbox.finish(*ready[0])
        self.assertEqual(inbox.scan(now=6), [])

    def test_monitor_retries_are_bounded_and_outputs_cannot_feed_back_into_input(self):
        inbox = FileInbox(self.root)
        inbox.scan(now=0)
        self.assertEqual([inbox.retry(self.path, now=1) for _ in range(5)], [True, True, True, True, False])
        with self.assertRaises(ValueError):
            validate_folders(self.root, self.root / "results")
        with self.assertRaises(ValueError):
            validate_folders(self.root, self.root.parent)

    def test_background_worker_emits_trigger_and_durable_log_without_touching_source(self):
        original = self.path.read_bytes()
        events = queue.Queue()
        worker = InspectionWorker(self.root, [self.path], self.root / "output", events, threading.Event(),
                                  predictor_factory=lambda path: FakePredictor(), stable_seconds=0)
        worker.run()
        messages = list(events.queue)
        self.assertNotIn("fatal", [kind for kind, _ in messages])
        result = next(payload for kind, payload in messages if kind == "result")
        self.assertTrue(result["triggered"])
        session = next(payload["session_dir"] for kind, payload in messages if kind == "ready")
        self.assertIn("TRIGGER", (Path(session) / "inspection.csv").read_text(encoding="utf-8-sig"))
        with (Path(session) / "inspection.csv").open(encoding="utf-8-sig", newline="") as stream:
            row = next(csv.DictReader(stream))
        self.assertEqual([row[f"vid_{unit}"] for unit in range(1, 4)], ["VIDLEFT", "VIDMIDDLE", "VIDRIGHT"])
        self.assertEqual([row[f"outcome_{unit}"] for unit in range(1, 4)], ["Good", "Bad", "Good"])
        self.assertEqual(row["tray_result"], "Bad")
        self.assertEqual(self.path.read_bytes(), original)

    def test_bad_image_emits_error_not_a_pass(self):
        Image.new("RGB", (5, 4)).save(self.path)
        events = queue.Queue()
        worker = InspectionWorker(self.root, [self.path], self.root / "output", events, threading.Event(),
                                  predictor_factory=lambda path: FakePredictor(), stable_seconds=0)
        worker.run()
        self.assertIn("file_error", [kind for kind, _ in list(events.queue)])
        self.assertNotIn("result", [kind for kind, _ in list(events.queue)])

    def test_batch_does_not_include_images_arriving_after_start(self):
        events = queue.Queue()
        def load_predictor(path):
            Image.new("RGB", (12, 4)).save(self.root / "later.png")
            return FakePredictor()
        worker = InspectionWorker(self.root, self.root, self.root.parent / (self.root.name + "_results"),
                                  events, threading.Event(), predictor_factory=load_predictor, stable_seconds=0)
        worker.output = self.root / "batch_output"
        worker.run()
        messages = list(events.queue)
        self.assertNotIn("fatal", [kind for kind, _ in messages])
        self.assertEqual(sum(kind == "result" for kind, _ in messages), 1)

    def test_gui_coalesces_popup_alerts_and_acknowledgement_does_not_erase_history(self):
        import tkinter as tk

        from C4Inspector import App
        root = tk.Tk()
        root.withdraw()
        self.addCleanup(root.destroy)
        app = App(root, model_dir=self.root, settings_path=self.root / "settings.json")
        self.addCleanup(lambda: root.after_cancel(app.poll_id))
        self.assertEqual(app.history.heading("image", "text"), "Product: unit VIDs in order")
        app.sound_var.set(False)
        predictor = FakePredictor()
        result = save_inspection(inspect_frame(self.path, predictor), self.root / "session")
        app.handle_event("result", result)
        popup = app.alert
        self.assertIsNotNone(popup)
        self.assertTrue(hasattr(app, "alert_preview"))
        self.assertEqual(app.history.item(app.history.get_children()[0], "values")[0],
                 "FAKE: VIDLEFT / VIDMIDDLE / VIDRIGHT")
        app.handle_event("result", result)
        self.assertIs(app.alert, popup)
        self.assertEqual(app.pending_alerts, 2)
        self.assertEqual(len(app.history.get_children()), 2)
        popup.geometry("580x350")
        popup.update()
        self.assertLessEqual(app.alert_ack.winfo_rooty() + app.alert_ack.winfo_height(),
                     popup.winfo_rooty() + popup.winfo_height())
        app.acknowledge()
        self.assertIsNone(app.alert)
        self.assertEqual(app.pending_alerts, 0)
        self.assertEqual(len(app.history.get_children()), 2)


if __name__ == "__main__":
    unittest.main()