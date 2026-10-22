"""Standalone desktop C4 inspection with CPU inference and acknowledged trigger alerts."""
import argparse
import json
import os
import queue
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from c4_monitor import InspectionWorker, validate_folders


def app_directory():
    return Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).parent


def configure_display():
    if os.name == "nt":
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (AttributeError, OSError):
            ctypes.windll.user32.SetProcessDPIAware()


class App:
    def __init__(self, root, model_dir=None, settings_path=None):
        self.root = root
        self.model_dir = Path(model_dir) if model_dir else app_directory() / "c4_model"
        self.settings_path = Path(settings_path) if settings_path else Path(
            os.environ.get("LOCALAPPDATA", str(Path.home()))) / "IOMC4Inspector/settings.json"
        self.events = queue.Queue(maxsize=16)
        self.thread = None
        self.stop_event = threading.Event()
        self.closing = False
        self.running = False
        self.failed = False
        self.session_dir = None
        self.results = {}
        self.pending_alerts = 0
        self.alert = None
        self.preview = self.photo = self.alert_photo = None
        self.root.title("Input Shuttle | C4 Inspection")
        self.root.geometry("1120x700")
        self.root.minsize(900, 600)
        self.root.configure(background="#f3f5f4")
        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure("TFrame", background="#f3f5f4")
        style.configure("TLabel", background="#f3f5f4", foreground="#243832", font=("Segoe UI", 10))
        style.configure("TButton", font=("Segoe UI", 10), padding=(10, 5))
        style.configure("TCheckbutton", background="#f3f5f4", font=("Segoe UI", 10))
        style.configure("TRadiobutton", background="#f3f5f4", font=("Segoe UI", 10))
        style.configure("Title.TLabel", font=("Segoe UI", 18, "bold"))
        style.configure("Muted.TLabel", foreground="#56665f", font=("Segoe UI", 9))
        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar(value=str(Path.home() / "Documents/IOM C4 Results"))
        self.mode_var = tk.StringVar(value="watch")
        self.existing_var = tk.BooleanVar(value=True)
        self.recursive_var = tk.BooleanVar(value=False)
        self.popup_var = tk.BooleanVar(value=True)
        self.sound_var = tk.BooleanVar(value=True)
        self.status = tk.StringVar(value="Idle")
        self.metrics = tk.StringVar(value="Frames: 0    Triggers: 0    Errors: 0    Pending: 0")
        self.model_text = tk.StringVar(value="C4 model unavailable")
        self.load_settings()
        self.mutable = []
        self.build()
        try:
            metadata = json.loads((self.model_dir / "metadata.json").read_text(encoding="utf-8"))
            self.model_text.set(f"{metadata['name']}   |   CPU   |   Raw threshold {metadata['raw_threshold']:.6f}")
        except (OSError, ValueError, KeyError, TypeError):
            pass
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.poll_id = self.root.after(100, self.poll)

    def editable(self, widget):
        self.mutable.append(widget)
        return widget

    def build(self):
        outer = ttk.Frame(self.root, padding=16)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(5, weight=1)
        heading = ttk.Frame(outer)
        heading.grid(row=0, column=0, sticky="ew")
        ttk.Label(heading, text="C4 Inspection", style="Title.TLabel").pack(side="left")
        ttk.Label(heading, textvariable=self.model_text, style="Muted.TLabel").pack(side="right")
        setup = ttk.Frame(outer)
        setup.grid(row=1, column=0, sticky="ew", pady=(12, 8))
        setup.columnconfigure(1, weight=1)
        for row, (label, variable) in enumerate((("Image folder", self.input_var), ("Results folder", self.output_var))):
            ttk.Label(setup, text=label).grid(row=row, column=0, sticky="w", padx=(0, 12), pady=4)
            self.editable(ttk.Entry(setup, textvariable=variable)).grid(row=row, column=1, sticky="ew", pady=4)
            self.editable(ttk.Button(setup, text="Browse...", command=lambda target=variable: self.browse(target))).grid(
                row=row, column=2, padx=(8, 0))
        options = ttk.Frame(outer)
        options.grid(row=2, column=0, sticky="ew")
        for text, value in (("Watch folder", "watch"), ("Batch folder", "batch")):
            self.editable(ttk.Radiobutton(options, text=text, variable=self.mode_var, value=value)).pack(side="left", padx=(0, 14))
        for text, variable in (("Include existing", self.existing_var), ("Subfolders", self.recursive_var)):
            self.editable(ttk.Checkbutton(options, text=text, variable=variable)).pack(side="left", padx=(0, 14))
        ttk.Checkbutton(options, text="Popup alerts", variable=self.popup_var).pack(side="right", padx=(10, 0))
        ttk.Checkbutton(options, text="Sound", variable=self.sound_var).pack(side="right")
        toolbar = ttk.Frame(outer)
        toolbar.grid(row=3, column=0, sticky="ew", pady=10)
        self.start_button = self.editable(ttk.Button(toolbar, text="Start inspection", command=self.start))
        self.start_button.pack(side="left")
        self.editable(ttk.Button(toolbar, text="Inspect files...", command=self.inspect_files)).pack(side="left", padx=6)
        self.stop_button = ttk.Button(toolbar, text="Stop", command=self.stop, state="disabled")
        self.stop_button.pack(side="left")
        self.ack_button = ttk.Button(toolbar, text="Acknowledge alerts", command=self.acknowledge, state="disabled")
        self.ack_button.pack(side="right")
        ttk.Button(toolbar, text="Open results", command=self.open_results).pack(side="right", padx=6)
        self.banner = tk.Label(outer, text="IDLE", background="#dfe7e3", foreground="#243832",
                               font=("Segoe UI", 12, "bold"), anchor="w", padx=12, pady=8)
        self.banner.grid(row=4, column=0, sticky="ew")
        review = ttk.Frame(outer)
        review.grid(row=5, column=0, sticky="nsew", pady=(8, 4))
        review.columnconfigure(0, weight=1)
        review.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(review, background="#e6eae7", highlightthickness=0, height=180)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", lambda event: self.draw_preview())
        table = ttk.Frame(review)
        table.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        table.columnconfigure(0, weight=1)
        self.history = ttk.Treeview(table, columns=("image", "decision", "scores", "seconds"), show="headings", height=4)
        for name, text, width in (("image", "Unit VIDs (left / middle / right)", 300), ("decision", "Decision", 110),
                      ("scores", "Raw unit scores (1 / 2 / 3)", 290), ("seconds", "Seconds", 85)):
            self.history.heading(name, text=text)
            self.history.column(name, width=width, minwidth=65, stretch=name == "image")
        self.history.tag_configure("trigger", foreground="#b42037")
        self.history.tag_configure("error", foreground="#94520d")
        self.history.grid(row=0, column=0, sticky="ew")
        scrollbar = ttk.Scrollbar(table, orient="vertical", command=self.history.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.history.configure(yscrollcommand=scrollbar.set)
        self.history.bind("<<TreeviewSelect>>", self.select_result)
        self.history.bind("<Double-1>", self.open_selected)
        ttk.Label(outer, textvariable=self.metrics).grid(row=6, column=0, sticky="w", pady=(6, 2))
        self.status_label = ttk.Label(outer, textvariable=self.status, wraplength=1040)
        self.status_label.grid(row=7, column=0, sticky="ew")
        self.root.bind("<Configure>", lambda event: self.status_label.configure(wraplength=max(300, self.root.winfo_width() - 40)))

    def browse(self, variable):
        selected = filedialog.askdirectory(parent=self.root, initialdir=variable.get() or str(Path.home()))
        if selected:
            variable.set(selected)

    def load_settings(self):
        try:
            saved = json.loads(self.settings_path.read_text(encoding="utf-8"))
            for key, variable in self.variables().items():
                if key in saved:
                    variable.set(saved[key])
        except (OSError, ValueError, TypeError, tk.TclError):
            pass

    def variables(self):
        return {"input": self.input_var, "output": self.output_var, "mode": self.mode_var,
                "existing": self.existing_var, "recursive": self.recursive_var,
                "popup": self.popup_var, "sound": self.sound_var}

    def start(self, files=None):
        if self.thread and self.thread.is_alive():
            return
        try:
            if not self.output_var.get().strip() or (files is None and not self.input_var.get().strip()):
                raise ValueError("Select image and results folders first.")
            source = files if files is not None else Path(self.input_var.get().strip())
            validate_folders(source, self.output_var.get().strip())
            self.settings_path.parent.mkdir(parents=True, exist_ok=True)
            self.settings_path.write_text(json.dumps({key: variable.get() for key, variable in self.variables().items()}, indent=2), encoding="utf-8")
            self.stop_event = threading.Event()
            worker = InspectionWorker(self.model_dir, source, self.output_var.get().strip(), self.events, self.stop_event,
                                      watch=files is None and self.mode_var.get() == "watch",
                                      include_existing=files is not None or self.mode_var.get() == "batch" or self.existing_var.get(),
                                      recursive=self.recursive_var.get())
        except (OSError, ValueError) as exc:
            messagebox.showerror("Cannot start C4 inspection", str(exc), parent=self.root)
            return
        self.failed = False
        self.running = True
        for widget in self.mutable:
            widget.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.status.set("Loading C4 model on CPU...")
        self.refresh_banner()
        self.thread = threading.Thread(target=worker.run, name="C4-inference", daemon=False)
        self.thread.start()

    def inspect_files(self):
        files = filedialog.askopenfilenames(parent=self.root, filetypes=[("Camera frames", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")])
        if files:
            self.start([Path(path) for path in files])

    def stop(self):
        self.stop_event.set()
        self.stop_button.configure(state="disabled")
        self.status.set("Stopping after the current frame...")

    def refresh_banner(self):
        if self.pending_alerts:
            self.banner.configure(text=f"TRIGGER | {self.pending_alerts} frame(s) awaiting acknowledgement",
                                  background="#b42037", foreground="white")
        elif self.failed:
            self.banner.configure(text="INSPECTION ERROR", background="#f3d6ad", foreground="#713c0c")
        else:
            self.banner.configure(text="INSPECTING" if self.running else "IDLE", background="#dfe7e3", foreground="#243832")
        self.ack_button.configure(state="normal" if self.pending_alerts else "disabled")

    def handle_event(self, kind, payload):
        if kind == "ready":
            self.session_dir = Path(payload["session_dir"])
            self.status.set(f"CPU inspection active | {self.session_dir}")
        elif kind == "result":
            scores = " / ".join(f"{row['raw_score']:.4f}" for row in payload["units"])
            decision = payload.get("tray_result", "TRIGGER" if payload["triggered"] else "NO TRIGGER")
            identifiers = " / ".join(row.get("vid") or row.get("vid_status", "UNREADABLE") for row in payload["units"])
            tags = ("error",) if decision == "Review" else (("trigger",) if payload["triggered"] else ())
            item = self.history.insert("", 0, values=(identifiers, decision, scores,
                                                      f"{payload['elapsed_seconds']:.2f}"), tags=tags)
            self.results[item] = payload
            self.trim_history()
            self.history.selection_set(item)
            self.select_result()
            if payload["triggered"]:
                self.show_alert(payload)
            if decision == "Review":
                self.failed = True
                self.status.set(f"VID needs review: {payload['source_path']}")
                self.refresh_banner()
        elif kind == "file_error":
            self.failed = True
            self.history.insert("", 0, values=(Path(payload["source_path"]).name, "ERROR", payload["error"], ""), tags=("error",))
            self.trim_history()
            self.status.set(f"Image not inspected: {payload['source_path']} | {payload['error']}")
            self.refresh_banner()
        elif kind == "counts":
            self.metrics.set(f"Frames: {payload['processed']}    Triggers: {payload['triggers']}    Review: {payload.get('reviews', 0)}    Errors: {payload['errors']}    Pending: {payload['pending']}")
        elif kind == "fatal":
            self.failed = True
            self.status.set(payload)
            self.refresh_banner()
            if not self.closing:
                messagebox.showerror("C4 inspection stopped", payload, parent=self.root)
        elif kind == "done":
            self.running = False
            self.stop_button.configure(state="disabled")
            for widget in self.mutable:
                widget.configure(state="normal")
            if not self.failed:
                self.status.set("Stopped" if self.stop_event.is_set() else "Batch complete")
            self.refresh_banner()

    def trim_history(self):
        for item in self.history.get_children()[500:]:
            self.results.pop(item, None)
            self.history.delete(item)

    def show_alert(self, result):
        self.pending_alerts += 1
        self.refresh_banner()
        if self.sound_var.get():
            try:
                import winsound
                winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
            except (ImportError, RuntimeError):
                self.root.bell()
        if not self.popup_var.get():
            return
        if self.alert is None or not self.alert.winfo_exists():
            self.alert = tk.Toplevel(self.root)
            self.alert.title("C4 TRIGGER")
            self.alert.geometry("660x380")
            self.alert.minsize(580, 350)
            self.alert.attributes("-topmost", True)
            self.alert.protocol("WM_DELETE_WINDOW", self.acknowledge)
            self.alert.columnconfigure(0, weight=1)
            self.alert.rowconfigure(2, weight=1)
            self.alert_heading = tk.Label(self.alert, background="#b42037", foreground="white",
                                          font=("Segoe UI", 14, "bold"), anchor="w", padx=14, pady=12)
            self.alert_heading.grid(row=0, column=0, sticky="ew")
            self.alert_detail = ttk.Label(self.alert, wraplength=610, padding=12)
            self.alert_detail.grid(row=1, column=0, sticky="ew")
            self.alert.bind("<Configure>", lambda event: self.alert_detail.configure(
                wraplength=max(200, self.alert.winfo_width() - 40)) if event.widget is self.alert else None)
            self.alert_canvas = tk.Canvas(self.alert, background="#f3f5f4", highlightthickness=0, height=150)
            self.alert_canvas.grid(row=2, column=0, sticky="nsew", padx=10)
            self.alert_canvas.bind("<Configure>", lambda event: self.draw_alert_preview())
            self.alert_ack = ttk.Button(self.alert, text="Acknowledge", command=self.acknowledge)
            self.alert_ack.grid(row=3, column=0, pady=10)
            self.alert.lift()
        flagged = [row for row in result["units"] if row["triggered"]]
        self.alert_heading.configure(text=f"C4 TRIGGER | {self.pending_alerts} unacknowledged frame(s)")
        detail = " / ".join(f"{row.get('vid') or row.get('vid_status', 'UNREADABLE')}: {row['raw_score']:.4f}" for row in flagged)
        self.alert_detail.configure(text=f"{Path(result['source_path']).name}\n{detail} | Threshold {result['threshold_raw']:.6f}")
        overlay = Path(result["result_dir"]) / f"{flagged[0]['artifact_name']}_overlay.jpg"
        try:
            with Image.open(overlay) as opened:
                image = opened.convert("RGB")
        except OSError as exc:
            self.status.set(f"Triggered; overlay unavailable: {exc}")
            return
        self.alert_preview = image
        self.draw_alert_preview()

    def draw_alert_preview(self):
        if self.alert is None or not hasattr(self, "alert_preview"):
            return
        image = self.alert_preview.copy()
        image.thumbnail((max(1, self.alert_canvas.winfo_width()), max(1, self.alert_canvas.winfo_height())))
        self.alert_photo = ImageTk.PhotoImage(image, master=self.alert)
        self.alert_canvas.delete("all")
        self.alert_canvas.create_image(self.alert_canvas.winfo_width() // 2, self.alert_canvas.winfo_height() // 2,
                                       image=self.alert_photo)

    def acknowledge(self):
        self.pending_alerts = 0
        if self.alert is not None:
            self.alert.destroy()
            self.alert = None
        self.alert_photo = None
        self.refresh_banner()

    def select_result(self, event=None):
        selected = self.history.selection()
        result = self.results.get(selected[0]) if selected else None
        if result:
            try:
                with Image.open(result["preview_path"]) as opened:
                    self.preview = opened.convert("RGB")
                self.draw_preview()
            except OSError as exc:
                self.status.set(f"Preview unavailable: {exc}")

    def draw_preview(self):
        self.canvas.delete("all")
        if self.preview is None:
            return
        image = self.preview.copy()
        image.thumbnail((max(1, self.canvas.winfo_width() - 12), max(1, self.canvas.winfo_height() - 12)))
        self.photo = ImageTk.PhotoImage(image, master=self.root)
        self.canvas.create_image(self.canvas.winfo_width() // 2, self.canvas.winfo_height() // 2, image=self.photo)

    def open_selected(self, event=None):
        selected = self.history.selection()
        if selected and selected[0] in self.results:
            self.open_path(self.results[selected[0]]["result_dir"])

    def open_results(self):
        path = Path(self.output_var.get()) / "Result"
        if not path.is_dir():
            path = self.session_dir or Path(self.output_var.get())
        if path.is_dir():
            self.open_path(path)

    def open_path(self, path):
        try:
            os.startfile(str(path))
        except OSError as exc:
            messagebox.showerror("Cannot open results", str(exc), parent=self.root)

    def poll(self):
        try:
            for _ in range(20):
                kind, payload = self.events.get_nowait()
                self.handle_event(kind, payload)
        except queue.Empty:
            pass
        if self.closing and (self.thread is None or not self.thread.is_alive()):
            self.root.destroy()
            return
        self.poll_id = self.root.after(100, self.poll)

    def close(self):
        self.closing = True
        self.stop()
        if self.alert is not None:
            self.alert.destroy()
            self.alert = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=app_directory() / "c4_model")
    parser.add_argument("--check-image", type=Path, help="Inspect one full frame without opening the GUI.")
    parser.add_argument("--report", type=Path, help="Write the command-line image check to JSON.")
    arguments = parser.parse_args()
    if arguments.check_image:
        from c4_runtime import Predictor, inspect_frame
        try:
            predictor = Predictor(arguments.model_dir)
            record = inspect_frame(arguments.check_image, predictor).record
            report = {"ok": True, "device": "cpu", "torch_version": predictor.torch.__version__, **record}
        except Exception as exc:
            report = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        text = json.dumps(report, indent=2, allow_nan=False)
        if arguments.report:
            arguments.report.write_text(text, encoding="utf-8")
        if sys.stdout is not None:
            print(text)
        return 0 if report["ok"] else 1
    configure_display()
    root = tk.Tk()
    App(root, arguments.model_dir)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())