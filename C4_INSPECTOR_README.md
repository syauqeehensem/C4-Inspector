# C4 Inspector Operator Guide

Extract the complete portable archive and run `C4Inspector.exe`. Keep `_internal`
and `c4_model` beside the executable. No Python installation or GPU is required.

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

Use unrotated 5300 x 1775 full frames in the trained three-unit layout. A frame
triggers if any C4 unit raw score is strictly greater than 42.89402770996094.
Files settle for two seconds; incomplete reads retry up to five times.
Unresolved, wrong-size, or invalid images are errors, not passes.

Version 1.1.0 registers each die with bounded integer-pixel translation before
scoring. It does not correct rotation, scale, perspective, or another layout.
Ambiguous, low-confidence, or out-of-range registration produces an ERROR, never
Good. The model and alignment-profile hashes are checked before inspection.

The selected output folder contains `Result/Good`, `Result/Bad`, and `Result/Review`.
Tray images retain exact source bytes and are named
`leftVID_middleVID_rightVID.jpg`, using the original extension. The confirmed VID
identifies one unit, not a tray, and is read from that unit's lower-left
rectangular Data Matrix, not the upper-left square code. The filename combines
the three independent unit VIDs in left-to-right order. Version 1.1.1 labels
the GUI history column Unit VIDs; inspection and sorting behavior are unchanged.
Repeated captures get a suffix instead of overwriting existing files. A tray is
Bad if any unit triggers, Good if none trigger, or Review if any VID is unreadable,
conflicting, duplicate, or filename-unsafe. Good is not defect-free certification.

Each tray has a JSON sidecar linking positions, VIDs, decode status, model/unit
decisions, scores, source hash, alignment offsets/correlations, and evidence paths.
The session CSV records all three VIDs and outcomes. Triggered-unit crops, maps,
and heatmap overlays use the VID as their name. Map rectangles are the actual
translated crops in original-frame coordinates. Open results opens the sorted
tray folders; double-click a history row for its preview and detailed evidence.

CSV/JSON records, previews, and triggered-frame originals, unit crops, heatmaps,
and raw maps are saved under each results session. Source images are not changed,
moved, or deleted. Monitor free disk space; evidence is not automatically pruned.
The GUI keeps the latest 500 rows while saved evidence remains on disk.

**NO TRIGGER is not a defect-free guarantee.** This model missed one of two
selected defective development units and excludes areas outside C4. The known
small single-spot defect remains missed after alignment. Raw scores are not
probabilities. It is not a machine interlock.
Popups require an interactive Windows session and may be hidden when disconnected
or minimized in RDP. Verify labelled examples and alert visibility before use.

The pictured evaluation target was Windows 10 Enterprise LTSC x64, i5-9500TE,
8 GB RAM; version 1.0.1 was reported working there, but version 1.1.0 still needs
target acceptance. Local Windows 11 checks passed. Camera
throughput, LTSC/runtime compatibility, output permissions, and sound must be
checked on the actual PC. If the pending count grows, processing is behind.

For a single-frame check, run PowerShell from the extracted application folder:

```powershell
.\C4Inspector.exe --check-image 'C:\Camera\sample.jpg' --report 'C:\Temp\c4-check.json'
```

The report parent must already exist and be writable. An invalid image produces
an error report and a nonzero exit code, not a passing decision.