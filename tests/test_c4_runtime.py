import json
import queue
import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np
from c4_monitor import FileInbox, InspectionWorker, validate_folders
from c4_runtime import inspect_frame, save_inspection
from PIL import Image


class FakePredictor:
    def __init__(self, scores=(10, 11, 9)):
        self.scores = iter(scores)
        self.metadata = {"frame_size_wh": [12, 4], "rois": [[0, 0, 4, 4], [4, 0, 8, 4], [8, 0, 12, 4]],
                         "raw_threshold": 10, "pixel_min": 0, "pixel_max": 20, "model_sha256": "test"}

    def predict(self, crop):
        return next(self.scores), np.ones((4, 4), dtype=np.float32)


class C4RuntimeTests(unittest.TestCase):
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
        result = save_inspection(inspect_frame(self.path, predictor), self.root / "session", predictor.metadata)
        folder = Path(result["result_dir"])
        self.assertEqual((folder / "original.png").read_bytes(), original_bytes)
        self.assertEqual(self.path.read_bytes(), original_bytes)
        self.assertTrue((folder / "unit_2_overlay.jpg").exists())
        self.assertFalse((folder / "unit_1.png").exists())
        with np.load(folder / "unit_2.npz") as data:
            self.assertEqual(data["roi_xyxy"].tolist(), [4, 0, 8, 4])
        self.assertTrue(json.loads((folder / "result.json").read_text())["triggered"])

    def test_untriggered_frame_has_no_defect_evidence_and_existing_result_is_not_overwritten(self):
        predictor = FakePredictor((10, 10, 10))
        inspection = inspect_frame(self.path, predictor)
        result = save_inspection(inspection, self.root / "session", predictor.metadata)
        self.assertFalse(result["triggered"])
        self.assertFalse((Path(result["result_dir"]) / "original.png").exists())
        with self.assertRaises(FileExistsError):
            save_inspection(inspection, self.root / "session", predictor.metadata)

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
        app.sound_var.set(False)
        predictor = FakePredictor()
        result = save_inspection(inspect_frame(self.path, predictor), self.root / "session", predictor.metadata)
        app.handle_event("result", result)
        popup = app.alert
        self.assertIsNotNone(popup)
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