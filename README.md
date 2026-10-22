# C4 Inspector

Windows desktop inspection tool for three fixed C4 die-surface regions, using
PatchCore anomaly detection and CPU-only inference.

## Downloads

Download the portable Windows application or the separate model bundle from
[Releases](https://github.com/syauqeehensem/C4-Inspector/releases).

- `C4Inspector-v1.0.0-windows-x64.zip`: complete application, CPU runtime, and model.
- `C4Inspector-model-v1.0.0.zip`: model weights and runtime metadata for source use.

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

## Important Limitations

**NO TRIGGER does not mean verified defect-free.** The chosen development model
flagged one of two selected defective units and still missed a small single-spot
defect. This is not an independent accuracy estimate. The model only inspects C4;
surrounding boards/components are excluded. Fixed rectangles are not alignment
or tracking. Heatmaps are not calibrated pixel-level defect classifications.

This is an operator alert, not a PLC signal, machine interlock, hard-real-time
system, or guaranteed defect detector. Popups require an interactive desktop.
Verify independent labelled examples, image alignment, processing throughput,
disk space, and alert visibility before operational use.

## Source Setup

Use Python 3.13 x64 on Windows. Extract the separate model download into the
repository so `c4_model/metadata.json` and `c4_model/patchcore.pt` exist.

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

## Hardware and Verification

Intended evaluation hardware: Windows 10 Enterprise LTSC x64, Intel Core
i5-9500TE at 2.20 GHz, 8 GB RAM. That exact PC's compatibility and throughput
remain unverified. Local validation ran on a Windows 11 development workstation.

- 25 experiment and runtime/GUI tests passed; 10 runtime/GUI tests are included here.
- CPU export scores and maps matched the source CPU model on six checked crops.
- CPU scoring of 204 normal calibration crops / 68 frames produced no frame flags
  at the unchanged threshold. This is not independent false-alarm evidence.
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