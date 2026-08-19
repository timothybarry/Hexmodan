$ErrorActionPreference = "Stop"
Write-Host "AZMO Mind 0.1 setup" -ForegroundColor Cyan

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "Python was not found. Install Python 3.11 or newer and add it to PATH."
}

if (-not (Test-Path ".venv")) {
    python -m venv .venv
}

& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -e ".[dev]"

if (Get-Command ollama -ErrorAction SilentlyContinue) {
    Write-Host "Ollama detected." -ForegroundColor Green
    ollama pull qwen3.5:9b
} else {
    Write-Warning "Ollama is not installed."
    Write-Host "Install: irm https://ollama.com/install.ps1 | iex"
    Write-Host "Then:    ollama pull qwen3.5:9b"
}

Write-Host "Setup complete." -ForegroundColor Green
Write-Host "Activate: .\.venv\Scripts\Activate.ps1"
Write-Host "Diagnose: azmo doctor"
Write-Host "Chat:     azmo chat"
