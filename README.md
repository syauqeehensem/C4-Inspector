# C4 Inspector

Windows desktop inspection tool for EMRXCP and M4G camera frames, using PatchCore
anomaly detection and CPU-only inference. EMRXCP units are checked on the C4 die
surface and across the whole unit; M4G units are checked across the whole unit.
The product is identified automatically from each frame's layout.

## Downloads

Download the portable Windows application or the separate model bundle from
[Releases](https://github.com/syauqeehensem/C4-Inspector/releases).

- `C4Inspector-v1.2.0-windows-x64.zip`: complete application, CPU runtime, and model bundle.
- `C4Inspector-models-v1.2.0.zip`: EMRXCP and M4G models, metadata, and alignment profiles for source use.

For the portable application, extract the entire archive into a new folder and
run `C4Inspector.exe`. Keep `_internal` and `models` beside it. Python, CUDA, and
internet access are not required to run the extracted application.

## Inspection

Select an image folder and a separate results folder, then start inspection.
The application supports live folder monitoring, snapshot batch inspection,
individual files, subfolders, raw unit scores, and source-frame previews.
Triggered frames raise a topmost heatmap popup with optional sound. Additional
triggers update the same popup; acknowledgement does not erase saved results.

Images must be unrotated **5300 x 1775** full frames in a supported layout:
EMRXCP (three units in one row) or M4G (eight units in two rows of four). Mixed
folders are supported. Each frame is classified from coarse intensity profiles;
an unclear or unknown layout is an ERROR, never Good. This is not a general-purpose
detector for other products or camera layouts. Supported formats are JPEG, PNG,
BMP, and single-page TIFF.

| Product | Inspection areas | Unit order (VIDs and filenames) |
| --- | --- | --- |
| EMRXCP | C4 die surface and whole unit, separate models | Left to right |
| M4G | Whole unit | Top row left to right, then bottom row left to right |

EMRXCP C4 reference rectangles are unchanged:

| Unit | Full-frame rectangle (left, top, right, bottom) |
| --- | --- |
| 1 | (640, 590, 1490, 1120) |
| 2 | (2200, 590, 3060, 1120) |
| 3 | (3765, 590, 4615, 1120) |

Whole-unit rectangles for both products are stored in the model bundle. Every
area crop is resized to 512 x 512 and processed sequentially on CPU. A unit
triggers if **any** of its areas has a raw score strictly greater than that area's
own threshold; a frame triggers if any unit triggers. The GUI shows the highest
score/threshold ratio per unit, so values above 1 trigger. Scores are distances,
not probabilities. Model, metadata, and alignment hashes are checked when the
bundle loads.

Each unit is registered by bounded integer-pixel translation before scoring:
EMRXCP from its C4 die edges, M4G from internal unit-structure edges. Reference
profiles are one-dimensional gradient summaries, not factory photographs. No
interpolation is used. Weak, ambiguous, or out-of-range registration sends the
tray to Review instead of Good. Rotation, scale, and perspective are not corrected.

## Tray Results

A VID identifies one unit, not a tray. A tray image filename joins the individual
unit VIDs in the product's unit order: three for EMRXCP
(`leftVID_middleVID_rightVID.jpg`) and eight for M4G (`VID1_..._VID8.jpg`, top row
left to right, then bottom row left to right). It is not a tray-level identifier.

The selected output folder contains `Result/Good`, `Result/Bad`, and `Result/Review`.
Tray copies retain the source extension and exact original bytes. Repeated captures
receive a suffix; existing results are never overwritten. EMRXCP VIDs come from the
lower-left rectangular Data Matrix. M4G VIDs come from the code below the left
square grid, beside the small pads; the upper square code is not used. A tray enters
Bad if any unit triggers, Good if none do, or Review if any VID is unreadable,
conflicting, duplicate, or filename-unsafe, or if any unit cannot be registered.
Good is a model routing label, not certification that a tray has no defects.

JSON sidecars and the session CSV link each VID with its product, per-area scores,
thresholds, and outcome. JSON also retains decode status, source hash,
actual/reference rectangles, alignment offsets and confidence, and evidence paths.
Triggered-area evidence is named `VID_Area` (for example `VID_C4_overlay.jpg`).
Raw anomaly maps use translated original-frame crop coordinates. Source images and
historical results are never changed or moved.

## Important Limitations

**NO TRIGGER does not mean verified defect-free.** The EMRXCP C4 model flagged one
of two selected defective development units and still misses a small single-spot
defect. This is not an independent accuracy estimate. C4-area defects are covered
by the C4 model; the whole-unit model adds coverage outside C4 and is not relied on
for C4. Its retraining accepted test-set alarms as normal without unit-by-unit
review, so a real non-C4 defect among them would now be scored as normal.

**The M4G model is provisional.** It was built from one operator-supplied golden
frame (eight good units) with no defective M4G examples. Its threshold, 50.316, is
1.10 x the highest leave-one-unit-out normal score. On 329 unlabelled M4G frames it
marked 3 trays Bad (unit ratios 1.010 to 1.162) and 1 Review (one unit could not be
registered). Those results have not been confirmed, so false-alarm and
defect-detection rates are unknown. Confirm the Bad trays and recalibrate after the
first production week.

Translation registration is limited to the supplied layouts and is not general
tracking. Heatmaps are not calibrated pixel-level defect classifications.

This is an operator alert, not a PLC signal, machine interlock, hard-real-time
system, or guaranteed defect detector. Popups require an interactive desktop.
Verify independent labelled examples, image alignment, processing throughput,
disk space, and alert visibility before operational use.

## Source Setup

Use Python 3.13 x64 on Windows. Extract the separate model download into the
repository so `models/products.json` and its product folders exist. The source
app and builder load that bundle by default.

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
i5-9500TE at 2.20 GHz, 8 GB RAM. The operator confirmed version 1.0.1 runs there;
version 1.2.0 still needs target-PC acceptance. Local validation ran on Windows 11.

- 20 runtime/monitor/GUI tests are included here, covering product detection,
  eight-unit VID order, per-area triggering, monitor-only areas, VID decoding, and
  registration failure routing.
- Product detection identified all 892 held-out EMRXCP frames and the M4G golden
  frame plus shifted and brightness-varied copies; a blank frame is rejected.
- EMRXCP whole unit: trained on the normal training frames plus 367 test-set
  false-alarm units the operator accepted as normal. Threshold 50.064 is the
  highest CPU score on 68 normal validation frames (0 of 68 flagged). On test hours
  held out from retraining, it flagged 2.0% and 3.6% of frames, compared with 39%
  before retraining. Alone, it flags 9 of the 13 units the C4 model flags; combined
  with C4, all 13 remain Bad. Five saved production frames remained Good.
- EMRXCP C4 behavior is unchanged: the multi-spot development defect still triggers
  and the small single-spot defect is still missed.
- M4G: 329 unlabelled test frames produced 325 Good, 3 Bad, 1 Review, and 0 errors.
  All 2,632 unit VIDs were read.
- On the Windows 11 workstation, CPU compute took about 2.7 seconds per EMRXCP
  tray and 2.6 seconds per M4G tray, using four threads. Expect several times longer
  on the i5-9500TE; check camera arrival rate on the target.
- Portable extraction, model loading, product detection, known triggers, invalid
  input, and GUI startup/clean close are checked on the built package before release.

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