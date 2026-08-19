@echo off
setlocal
title AZMO Voice Setup - Python 3.12 + Azmodan Clone
cd /d "%~dp0"

echo.
echo ============================================================
echo      AZMO VOICE SETUP  (Python 3.12, DSP, Azmodan clone)
echo ============================================================
echo.
echo This runs unattended. You can walk away. It will:
echo   1. Install Python 3.12 if you don't have it
echo   2. Rebuild AZMO's environment on 3.12
echo   3. Install the voice modulation + cloning stack
echo   4. Download the XTTS voice model (about 2 GB, first run only)
echo   5. Speak a test line in the cloned Azmodan voice
echo.
echo The first run can take 20-40 minutes, mostly the download.
echo.
pause

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup_py312.ps1" -WithClone

set EXIT_CODE=%ERRORLEVEL%
echo.
if not "%EXIT_CODE%"=="0" (
    echo Setup stopped with code %EXIT_CODE%. Read the messages above.
    echo If it mentions Python 3.12, install it from python.org
    echo (tick "Add Python to PATH"^) and run this again.
) else (
    echo Voice setup complete. Launch AZMO with START_AZMO.bat.
)
echo.
pause
exit /b %EXIT_CODE%
