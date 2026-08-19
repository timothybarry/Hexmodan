@echo off
rem ============================================================
rem  AZMO - hands-free mode (listen . think . speak)
rem  NOTE: keep this file pure ASCII. A single non-ASCII byte
rem  (an em-dash) has broken Windows PowerShell parsing before.
rem ============================================================
setlocal
title AZMO - hands-free (ears + brain + voice)
cd /d "%~dp0"

rem --- Ask for admin once, so the temporary GPU power cap can be applied. ---
rem The cap clips the current transients implicated in the reboots. Declining
rem is fine: AZMO still runs, just at full GPU power.
if /i "%~1"=="-noelevate" goto run
net session >nul 2>&1
if not errorlevel 1 goto run

echo.
echo Requesting administrator rights so the temporary GPU power cap
echo can be applied. Decline and AZMO still runs, just uncapped.
echo.
powershell.exe -NoLogo -NoProfile -Command "try { Start-Process -FilePath '%~f0' -ArgumentList '-noelevate' -Verb RunAs; exit 0 } catch { exit 1 }"
if not errorlevel 1 exit /b 0
echo Elevation declined - continuing without the GPU cap.
echo.

:run
echo.
echo ============================================================
echo        AZMO - HANDS-FREE MODE  (listen . think . speak)
echo ============================================================
echo.
echo This starts the LLM, checks that the brain, voice, and ears
echo are all ready, then listens for the wake word "Azmodan".
echo.
echo The microphone is muted while AZMO thinks and speaks, so he
echo cannot hear his own voice and answer himself.
echo.

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\one_click_listen.ps1"

set EXIT_CODE=%ERRORLEVEL%
echo.
if not "%EXIT_CODE%"=="0" (
    echo AZMO stopped with code %EXIT_CODE%. Read the messages above.
) else (
    echo AZMO has closed.
)
echo.
pause
exit /b %EXIT_CODE%
