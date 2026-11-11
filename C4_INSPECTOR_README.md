# C4 Inspector Operator Guide

Extract the complete portable archive into a new folder and run `C4Inspector.exe`.
Keep `_internal` and `models` beside the executable. No Python installation or GPU
is required.

Corrected builds include Microsoft's signed x64 C++ runtime DLLs in `_internal`,
including `vcruntime140_threads.dll`. The original v1.0.0 package omitted that
DLL and could report WinError 126 while loading PyTorch's `shm.dll` on PCs
without the current C++ runtime. Extract the complete corrected build into a
new folder rather than merging it into an older installation. No separate C++
runtime installation is intended to be needed with the corrected build.

The application still requires supported 64-bit Windows, a compatible x64 CPU,
sufficient RAM, and writable disk space. It is not hardware-independent. Do not
download individual DLLs from unofficial websites or modify Windows system DLLs.

1. Select the camera image folder and a separate writable results folder.
2. Choose Watch folder for incoming frames or Batch folder for a snapshot.
3. Select Include existing and Subfolders as required, then Start inspection.
4. A trigger opens a topmost heatmap popup; sound is optional. New triggers update
   the same popup. Acknowledge clears the alert without deleting evidence.
5. Select a result for its source-frame preview; double-click it for evidence.
6. Stop finishes the current frame. Restarting creates a new session, not a resume.

Use unrotated 5300 x 1775 full frames. Each frame is identified as EMRXCP (three
units in one row) or M4G (eight units in two rows of four), so one folder may mix
both products. An unknown or unclear layout is an error, not a pass. EMRXCP units
are scored on the C4 die and the whole unit with separate models; M4G units are
scored on the whole unit. A unit triggers if any of its areas scores strictly above
that area's threshold. History shows each unit's highest score/threshold ratio;
values above 1 trigger.
Files settle for two seconds; incomplete reads retry up to five times.
Unresolved, wrong-size, or invalid images are errors, not passes.

Each unit is registered with bounded integer-pixel translation before scoring.
It does not correct rotation, scale, perspective, or another layout. Ambiguous,
low-confidence, or out-of-range registration sends the tray to Review, never
Good. Model, metadata, and alignment hashes are checked before inspection.

The selected output folder contains `Result/Good`, `Result/Bad`, and `Result/Review`.
Tray images retain exact source bytes and the original extension. Each VID
identifies one unit, not a tray; the filename joins the unit VIDs in product order.
EMRXCP: `leftVID_middleVID_rightVID`, read from each unit's lower-left rectangular
Data Matrix. M4G: eight VIDs, top row left to right then bottom row left to right,
read from the code below the left square grid beside the small pads (not the upper
square code).
Repeated captures get a suffix instead of overwriting existing files. A tray is
Bad if any unit triggers, Good if none trigger, or Review if any VID is unreadable,
conflicting, duplicate, or filename-unsafe, or any unit cannot be registered.
Good is not defect-free certification.

Each tray has a JSON sidecar linking product, positions, VIDs, decode status,
per-area scores and thresholds, unit decisions, source hash, alignment
offsets/correlations, and evidence paths. The session CSV records every VID,
outcome, score ratio, and per-area score. Triggered-area crops, maps, and heatmap
overlays are named `VID_Area`. Map rectangles are the actual translated crops in
original-frame coordinates. Open results opens the sorted tray folders;
double-click a history row for its preview and detailed evidence.

CSV/JSON records, previews, and triggered-frame originals, unit crops, heatmaps,
and raw maps are saved under each results session. Source images are not changed,
moved, or deleted. Monitor free disk space; evidence is not automatically pruned.
The GUI keeps the latest 500 rows while saved evidence remains on disk.

**NO TRIGGER is not a defect-free guarantee.** The EMRXCP C4 model missed one of
two selected defective development units; the known small single-spot defect
remains missed. **M4G is provisional**: it was built from one golden frame with no
defective M4G examples, so its false-alarm and detection rates are unknown until
production Bad/Good results are confirmed. Raw scores are not probabilities. It is
not a machine interlock.
Popups require an interactive Windows session and may be hidden when disconnected
or minimized in RDP. Verify labelled examples and alert visibility before use.

The pictured evaluation target was Windows 10 Enterprise LTSC x64, i5-9500TE,
8 GB RAM; version 1.0.1 was reported working there, but version 1.2.0 still needs
target acceptance. EMRXCP now runs two models per unit and M4G eight units per
tray, so each frame takes longer than in version 1.1. Camera
throughput, LTSC/runtime compatibility, output permissions, and sound must be
checked on the actual PC. If the pending count grows, processing is behind.

For a single-frame check, run PowerShell from the extracted application folder:

```powershell
.\C4Inspector.exe --check-image 'C:\Camera\sample.jpg' --report 'C:\Temp\c4-check.json'
```

The report parent must already exist and be writable. An invalid image produces
an error report and a nonzero exit code, not a passing decision.