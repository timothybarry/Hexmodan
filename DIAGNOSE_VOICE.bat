@echo off
rem ============================================================
rem  VOICE STACK DIAGNOSTIC
rem  Finds exactly which step crashes the process.
rem  NOTE: keep this file pure ASCII.
rem ============================================================
setlocal
title AZMO - voice diagnostic
cd /d "%~dp0"

set PY=%~dp0.venv312\Scripts\python.exe
if not exist "%PY%" set PY=%~dp0.venv\Scripts\python.exe
if not exist "%PY%" (
    echo No environment found. Run SETUP_VOICE.bat first.
    pause
    exit /b 1
)

set COQUI_TOS_AGREED=1

echo.
echo Running the GPU path. Watch for the last [step N] line.
echo.
"%PY%" scripts\diagnose_voice.py %*
set CODE=%ERRORLEVEL%

echo.
echo ------------------------------------------------------------
if "%CODE%"=="0" (
    echo All steps passed. The voice stack is healthy.
) else (
    echo Exited with code %CODE%.
    echo   -1073740791 means a hard native abort - the step printed
    echo   last is the one that killed it. Copy everything above.
    echo.
    echo Now trying the CPU path to confirm it is GPU-specific...
    echo ------------------------------------------------------------
    "%PY%" scripts\diagnose_voice.py --cpu
    echo.
    echo If the CPU run passed and the GPU run did not, set
    echo   speech.clone_device: cpu
    echo in config\azmo.yaml to get a working voice today.
)
echo ------------------------------------------------------------
echo.
pause
exit /b %CODE%
