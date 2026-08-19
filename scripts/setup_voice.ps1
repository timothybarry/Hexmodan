# AZMO voice setup: install the clone + DSP + prep extras, isolate a clean
# reference from source audio, download XTTS, and speak a test line.
# Run from the repo root after START_AZMO.bat has created the .venv:
#   .\scripts\setup_voice.ps1 -Sources "Diablo 3- All Azmodan Voice Lines.mp3"

param(
    [string[]]$Sources = @(),
    [string]$Reference = "data/voices/azmo_reference.wav",
    [double]$Seconds = 25.0,
    [switch]$SkipReference
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$venvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    throw "Virtual environment not found. Run START_AZMO.bat once first."
}

function Step([string]$m) { Write-Host "`n==> $m" -ForegroundColor Cyan }

Step "Installing voice extras (clone + dsp + prep)"
Write-Host "This pulls torch and coqui-tts and is a large download."
& $venvPython -m pip install -e ".[clone,prep]"
if ($LASTEXITCODE -ne 0) { throw "Voice extras failed to install." }

if (-not $SkipReference) {
    if ($Sources.Count -eq 0) {
        # Default to any Azmodan MP3s sitting next to the repo.
        $Sources = Get-ChildItem -Path $RepoRoot, (Split-Path -Parent $RepoRoot) `
            -Filter "*.mp3" -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -match "Azmo" } | ForEach-Object { $_.FullName }
    }
    if ($Sources.Count -eq 0) {
        Write-Warning "No source audio given and no Azmodan MP3s found. Skipping reference build."
        Write-Host "Re-run with:  .\scripts\setup_voice.ps1 -Sources 'path\to\voice.mp3'"
    } else {
        Step "Isolating a clean clone reference (Demucs)"
        Write-Host "Sources: $($Sources -join ', ')"
        & $venvPython "scripts\prepare_reference.py" @Sources --out $Reference --seconds $Seconds
        if ($LASTEXITCODE -ne 0) { throw "Reference preparation failed." }
    }
}

Step "Downloading XTTS and speaking a test line"
Write-Host "First run downloads the ~2 GB XTTS v2 model and accepts its license."
$env:COQUI_TOS_AGREED = "1"
& $venvPython -m azmo_mind.cli say "Nephalem. The vessel now speaks with my voice." `
    --config config/azmo.yaml
if ($LASTEXITCODE -ne 0) {
    Write-Warning "Test line failed. Confirm data/voices/azmo_reference.wav exists and engine is clone/auto."
    exit 1
}

Step "Voice ready"
Write-Host "AZMO will now use the cloned voice in 'azmo chat'." -ForegroundColor Green
