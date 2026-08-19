@echo off
setlocal
title AZMO - collect crash logs
cd /d "%~dp0"
echo Gathering Windows + GPU + Ollama logs into azmo_crash_report.txt ...
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\collect_logs.ps1"
echo.
pause
