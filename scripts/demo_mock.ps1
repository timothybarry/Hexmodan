$ErrorActionPreference = "Stop"
if (-not (Test-Path ".venv\Scripts\Activate.ps1")) {
    throw "Virtual environment not found. Run .\scripts\setup_windows.ps1 first."
}
. .\.venv\Scripts\Activate.ps1
azmo once "Awaken, Azmo." --config config/mock.yaml --json
