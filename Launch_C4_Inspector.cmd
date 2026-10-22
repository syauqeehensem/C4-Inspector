@echo off
setlocal
if exist "%~dp0release\C4Inspector\C4Inspector.exe" (
    start "" "%~dp0release\C4Inspector\C4Inspector.exe"
    exit /b 0
)
if exist "%~dp0.c4-runtime-venv\Scripts\pythonw.exe" (
    start "" "%~dp0.c4-runtime-venv\Scripts\pythonw.exe" "%~dp0C4Inspector.py"
    exit /b 0
)
echo C4 Inspector is not built. See C4_INSPECTOR_README.md.
pause
exit /b 1