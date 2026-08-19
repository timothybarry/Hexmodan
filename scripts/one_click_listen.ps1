# One-click hands-free AZMO: start Ollama, pre-flight all three subsystems
# (brain, voice, ears), then run the live listen loop. Reuses the voice-ready
# .venv312 environment. Launched by START_AZMO_VOICE.bat.
# NOTE: keep this file pure ASCII - Windows PowerShell mis-reads non-ASCII bytes.

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

function Step([string]$m) { Write-Host "`n==> $m" -ForegroundColor Cyan }

$venvPython = Join-Path $RepoRoot ".venv312\Scripts\python.exe"
if (-not (Test-Path $venvPython)) { $venvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe" }
if (-not (Test-Path $venvPython)) {
    throw "No environment found. Run SETUP_VOICE.bat first (installs Python 3.12 + the voice stack)."
}
Write-Host "Environment: $venvPython"
$env:COQUI_TOS_AGREED = "1"

# Keep CUDA allocations from fragmenting across the LLM/XTTS handoff. Fewer
# large re-allocations means fewer sharp current ramps on the 4080.
if (-not $env:PYTORCH_CUDA_ALLOC_CONF) { $env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True" }

# Ollama defaults allow several models resident and several requests in flight.
# On a 12 GB card that means the LLM can be doing concurrent work while XTTS is
# loading - two peak loads at once, which is what we are trying to avoid. AZMO
# only ever needs one model answering one request at a time.
if (-not $env:OLLAMA_MAX_LOADED_MODELS) { $env:OLLAMA_MAX_LOADED_MODELS = "1" }
if (-not $env:OLLAMA_NUM_PARALLEL) { $env:OLLAMA_NUM_PARALLEL = "1" }

$isAdmin = ([Security.Principal.WindowsPrincipal] `
    [Security.Principal.WindowsIdentity]::GetCurrent()
).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

Step "GPU power"
if ($isAdmin) {
    Write-Host "Running elevated - AZMO will apply the temporary power cap from config/azmo.yaml." -ForegroundColor Green
} else {
    Write-Warning "Not running as administrator. The GPU power cap will be SKIPPED."
    Write-Warning "Close this and run START_AZMO_VOICE.bat again (accept the UAC prompt) to enable it."
}

Step "Starting the LLM brain (Ollama)"
$ready = $false
try { $null = Invoke-RestMethod "http://127.0.0.1:11434/api/tags" -TimeoutSec 3; $ready = $true } catch {}
if (-not $ready) {
    $ollama = (Get-Command ollama.exe -ErrorAction SilentlyContinue).Source
    if (-not $ollama) {
        foreach ($c in @("$env:LOCALAPPDATA\Programs\Ollama\ollama.exe", "$env:ProgramFiles\Ollama\ollama.exe")) {
            if (Test-Path $c) { $ollama = $c; break }
        }
    }
    if ($ollama) {
        Start-Process -FilePath $ollama -ArgumentList "serve" -WindowStyle Hidden
        for ($i = 0; $i -lt 20; $i++) {
            Start-Sleep -Milliseconds 750
            try { $null = Invoke-RestMethod "http://127.0.0.1:11434/api/tags" -TimeoutSec 2; $ready = $true; break } catch {}
        }
    }
}
if ($ready) { Write-Host "Ollama is up." -ForegroundColor Green }
else { Write-Warning "Ollama is not reachable. The brain step will fail. Start Ollama and re-run." }

Step "Pre-flight: brain + voice + ears + GPU"
& $venvPython -m azmo_mind.cli check

Step "AZMO is listening"
Write-Host 'Say:  Azmodan, introduce yourself.   (press Ctrl+C to stop)' -ForegroundColor Yellow
Write-Host 'The mic is muted while he thinks and speaks, then reopens after a short cooldown.' -ForegroundColor DarkGray
& $venvPython -m azmo_mind.cli listen
$listenExit = $LASTEXITCODE

# AZMO restores the power limit on a clean exit. If the window was killed, or
# elevation was declined mid-session, say so plainly rather than leaving the
# card quietly capped the next time the machine is used for gaming.
Step "GPU power on exit"
& $venvPython -m azmo_mind.cli gpu status

exit $listenExit
