@echo off
rem ============================================================
rem  RESTORE FULL GPU POWER  -  run this before gaming
rem  NOTE: keep this file pure ASCII.
rem ============================================================
setlocal
title AZMO - restore full GPU power
cd /d "%~dp0"

if /i "%~1"=="-noelevate" goto run
net session >nul 2>&1
if not errorlevel 1 goto run
echo Requesting administrator rights (changing the power limit needs them)...
powershell.exe -NoLogo -NoProfile -Command "try { Start-Process -FilePath '%~f0' -ArgumentList '-noelevate' -Verb RunAs; exit 0 } catch { exit 1 }"
if not errorlevel 1 exit /b 0
echo Elevation declined. Nothing changed.
pause
exit /b 1

:run
echo ============================================================
echo   RESTORE FULL GPU POWER
echo ============================================================
echo.
echo Before (limit / stock default / max):
nvidia-smi --query-gpu=power.limit,power.default_limit,power.max_limit --format=csv
echo.

for /f "usebackq tokens=*" %%D in (`nvidia-smi --query-gpu^=power.default_limit --format^=csv^,noheader^,nounits`) do set DEFAULT_W=%%D
if "%DEFAULT_W%"=="" (
    echo Could not read the stock power limit. Is nvidia-smi on PATH?
    pause
    exit /b 1
)

echo Restoring to the stock limit of %DEFAULT_W% W ...
nvidia-smi -pl %DEFAULT_W%
echo.
echo After:
nvidia-smi --query-gpu=power.limit --format=csv
echo.
echo Full power restored. Go and play.
echo (A reboot also restores it, so this is only needed mid-session.)
echo.
pause
