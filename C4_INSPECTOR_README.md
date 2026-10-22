# C4 Inspector Operator Guide

Extract the complete portable archive and run `C4Inspector.exe`. Keep `_internal`
and `c4_model` beside the executable. No Python installation or GPU is required.

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

CSV/JSON records, previews, and triggered-frame originals, unit crops, heatmaps,
and raw maps are saved under each results session. Source images are not changed,
moved, or deleted. Monitor free disk space; evidence is not automatically pruned.
The GUI keeps the latest 500 rows while saved evidence remains on disk.

**NO TRIGGER is not a defect-free guarantee.** This model missed one of two
selected defective development units, excludes areas outside C4, and cannot track
shifted dies. Raw scores are not probabilities. It is not a machine interlock.
Popups require an interactive Windows session and may be hidden when disconnected
or minimized in RDP. Verify labelled examples and alert visibility before use.

The pictured evaluation target was Windows 10 Enterprise LTSC x64, i5-9500TE,
8 GB RAM; that target remains unverified. Local Windows 11 checks passed. Camera
throughput, LTSC/runtime compatibility, output permissions, and sound must be
checked on the actual PC. If the pending count grows, processing is behind.

For a single-frame check, run PowerShell from the extracted application folder:

```powershell
.\C4Inspector.exe --check-image 'C:\Camera\sample.jpg' --report 'C:\Temp\c4-check.json'
```

The report parent must already exist and be writable. An invalid image produces
an error report and a nonzero exit code, not a passing decision.