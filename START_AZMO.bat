@echo off
setlocal
title AZMO Mind Launcher
cd /d "%~dp0"

echo.
echo ============================================================
echo                    AZMO MIND 0.2.5 LAUNCHER - CLAUDE EDITION
echo ============================================================
echo.
echo The first launch may install Python and Ollama, then download
echo the approximately 6.6 GB qwen3.5:9b model.
echo.

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\one_click_start.ps1"

set EXIT_CODE=%ERRORLEVEL%
echo.
if not "%EXIT_CODE%"=="0" (
    echo AZMO stopped with error code %EXIT_CODE%.
    echo Read the message above, then press any key to close.
    pause >nul
) else (
    echo AZMO has closed normally.
    pause
)
exit /b %EXIT_CODE%
