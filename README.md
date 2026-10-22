# C4 Inspector

Windows desktop inspection tool for three C4 die-surface regions, using
PatchCore anomaly detection and CPU-only inference.

## Downloads

Download the portable Windows application or the separate model bundle from
[Releases](https://github.com/syauqeehensem/C4-Inspector/releases).

- `C4Inspector-v1.1.0-windows-x64.zip`: complete application, CPU runtime, and aligned model.
- `C4Inspector-model-v1.1.0.zip`: weights, metadata, and alignment profiles for source use.

For the portable application, extract the entire archive and run
`C4Inspector.exe`. Keep `_internal` and `c4_model` beside it. Python, CUDA, and
internet access are not required to run the extracted application.

## Inspection

Select an image folder and a separate results folder, then start inspection.
The application supports live folder monitoring, snapshot batch inspection,
individual files, subfolders, raw unit scores, and source-frame previews.
Triggered frames raise a topmost heatmap popup with optional sound. Additional
triggers update the same popup; acknowledgement does not erase saved results.

Images must be unrotated **5300 x 1775** full frames in the trained three-unit
layout. This model is not a general-purpose detector for other camera layouts.
Supported formats are JPEG, PNG, BMP, and single-page TIFF.

| Unit | Full-frame rectangle (left, top, right, bottom) |
| --- | --- |
| 1 | (640, 590, 1490, 1120) |
| 2 | (2200, 590, 3060, 1120) |
| 3 | (3765, 590, 4615, 1120) |

Each crop is resized to 512 x 512 and processed sequentially on CPU. A frame
triggers if any unit's raw score is **strictly greater than 42.89402770996094**.
Scores are distances, not probabilities. Model weights are checked against the
SHA-256 hash in the metadata when loaded.

Version 1.1.0 uses bounded integer-pixel translation to register each die before
scoring. Reference profiles are one-dimensional gradient summaries, not factory
photographs. The profile hash is verified at load time. No interpolation is used
for alignment. Weak, ambiguous, out-of-range, or out-of-bounds matches produce an
ERROR, not a Good result. This does not correct rotation, scale, or perspective.

## Tray Results

The selected output folder contains `Result/Good`, `Result/Bad`, and `Result/Review`.
Original-resolution tray copies are named `leftVID_middleVID_rightVID.jpg`, retaining
the source extension and exact original bytes. Repeated captures receive a suffix;
existing results are never overwritten. Each VID comes only from the lower-left
rectangular Data Matrix. A tray enters Bad if any unit triggers, Good if none do,
or Review if any VID is unreadable, conflicting, duplicate, or filename-unsafe.
Good is a model routing label, not certification that a tray has no defects.

JSON sidecars and the session CSV link each left-to-right VID with its score and
outcome. JSON also retains decode status, source hash, actual/reference rectangles,
alignment offsets and confidence, and evidence paths. Triggered-unit evidence is
named by VID. Raw anomaly maps use translated original-frame crop coordinates.
Source images and historical results are never changed or moved.

## Important Limitations

**NO TRIGGER does not mean verified defect-free.** The chosen development model
flagged one of two selected defective units and still missed a small single-spot
defect. This is not an independent accuracy estimate. The model only inspects C4;
surrounding boards/components are excluded. Translation registration is limited
to the supplied layout and is not general tracking. Heatmaps are not calibrated
pixel-level defect classifications.

This is an operator alert, not a PLC signal, machine interlock, hard-real-time
system, or guaranteed defect detector. Popups require an interactive desktop.
Verify independent labelled examples, image alignment, processing throughput,
disk space, and alert visibility before operational use.

## Source Setup

Use Python 3.13 x64 on Windows. Extract the separate model download into the
repository so `c4_model/metadata.json`, `c4_model/patchcore.pt`, and
`c4_model/alignment.json` exist. Use the version 1.1.0 model bundle for alignment.

```powershell
py -3.13 -m venv .c4-runtime-venv
.\.c4-runtime-venv\Scripts\python.exe -m pip install -r .\requirements-c4-gui.txt
.\.c4-runtime-venv\Scripts\python.exe C4Inspector.py
```

Run synthetic-fixture tests without any factory images or model download:

```powershell
.\.c4-runtime-venv\Scripts\python.exe -m unittest discover -s tests -p test_c4_runtime.py -v
```

To build a new portable folder:

```powershell
.\.c4-runtime-venv\Scripts\python.exe -m pip install 'pyinstaller>=6.16,<7'
.\build_c4_inspector.ps1
```

The build refuses to overwrite an existing release folder. It creates a package
inventory and SHA-256 manifest. Use `-DistRoot .\release_next` for another build.
No training libraries, torchvision, Lightning, anomalib, or GPU are needed at runtime.

The updated builder explicitly bundles Microsoft's signed x64 C++ runtime DLLs,
including `vcruntime140_threads.dll`. Install the current Microsoft Visual C++
x64 runtime on the build PC, or use `-VCRuntimeDirectory` to select its x64
redistributable DLL directory. The original v1.0.0 archive omitted the threading
DLL and could fail with WinError 126 on a PC without the current runtime. A
corrected complete build is needed; moving the old package to C: is not a fix.
Portable builds still require supported x64 Windows, a compatible CPU, RAM, and
disk space. They do not make the model independent of hardware or OS support.

## Hardware and Verification

Intended evaluation hardware: Windows 10 Enterprise LTSC x64, Intel Core
i5-9500TE at 2.20 GHz, 8 GB RAM. The operator confirmed the corrected version
1.0.1 runs there; version 1.1.0 still needs target-PC acceptance. Local validation
ran on Windows 11.

- 15 runtime/monitor/GUI tests are included here, including VID and alignment checks.
- CPU export scores and maps matched the source CPU model on six checked crops.
- Aligned CPU scoring of 204 normal calibration crops / 68 frames produced no model
  flags; three frames required VID review. This is not independent false-alarm evidence.
- All five saved production trigger frames (10 previously red units) aligned with
  no model triggers at the unchanged threshold. The other production frames lacked
  full-resolution originals in the result package and were not rescored.
- The previously detected multi-spot defect still triggered after alignment;
  the known small single-spot defect remains missed. No model retraining was done.
- Local compute averaged about 0.85 seconds per three-unit frame, with about
  416 MiB peak process working memory, excluding GUI/evidence overhead.
- Portable ZIP extraction to a different path, model loading, a known trigger,
  rejection of corrupt input, and GUI startup/clean close were checked locally.

Verify the actual Windows LTSC build, approved runtime dependencies, camera
arrival rate, sound, and operator session before using the tool on that PC.

## Data and Licensing

Factory images, datasets, internal evidence, training checkpoints, and local
environments are deliberately excluded. Release model metadata omits local
training paths and image-level development records. The model is specific to
the documented layout and retains the limitations above.

No open-source license has been selected for the project code or model.
Third-party runtime components retain their own licenses; their notices are
included in the portable application.