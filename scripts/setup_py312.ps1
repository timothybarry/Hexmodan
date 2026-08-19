# Create AZMO's voice environment on Python 3.12 so the audio wheels
# (pedalboard, pyworld, soundfile, numpy - and coqui-tts) install cleanly.
# Brand-new Python builds (3.13/3.14) often have no prebuilt audio wheels yet.
#
# It builds a SEPARATE .venv312 folder and never touches your existing .venv,
# so a still-open AZMO window can't block it. START_AZMO.bat automatically
# prefers .venv312 when it exists.
#
# Run once from the repo root:   .\scripts\setup_py312.ps1  (or -WithClone)

param([switch]$WithClone)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

function Step([string]$m) { Write-Host "`n==> $m" -ForegroundColor Cyan }

# The interpreter is kept as an executable plus a separate argument list.
#
# It used to be one array, invoked as `& $py312[0] $py312[1..($py312.Count-1)]`.
# That is correct for @("py","-3.12") and quietly catastrophic for a
# one-element @("C:\...\python.exe"): PowerShell's range operator counts DOWN
# when the start exceeds the end, so `1..0` is @(1, 0) rather than the empty
# list the code assumed. The interpreter path was therefore passed to itself as
# a script argument, and Python tried to parse python.exe as source:
#
#     SyntaxError: Non-UTF-8 code starting with '\x90'
#
# The one-element branch only runs when `py -3.12` is missing — that is, on a
# machine where this script has just installed Python — so the bug stayed hidden
# on any machine that had already been set up once.

function Test-Python312 {
    param([string]$Exe, [string[]]$Prefix = @())
    if (-not $Exe) { return $false }
    try {
        $reported = & $Exe @Prefix -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
        return ($LASTEXITCODE -eq 0 -and $reported -eq "3.12")
    } catch { return $false }
}

function Resolve-Python312 {
    # The launcher first: it is the most reliable way to ask for a specific version.
    if (Get-Command py -ErrorAction SilentlyContinue) {
        if (Test-Python312 -Exe "py" -Prefix @("-3.12")) {
            return [pscustomobject]@{ Exe = "py"; Arguments = @("-3.12") }
        }
    }
    # Anything on PATH that turns out to actually be 3.12. `python` is checked
    # too, because the Store stub answers to the name without being Python — the
    # version probe is what rejects it.
    foreach ($name in @("python3.12", "python")) {
        $cmd = Get-Command $name -ErrorAction SilentlyContinue
        if ($cmd -and (Test-Python312 -Exe $cmd.Source)) {
            return [pscustomobject]@{ Exe = $cmd.Source; Arguments = @() }
        }
    }
    # The standard install locations, for the case where PATH was never updated
    # (a fresh winget install in this same session will not be on PATH yet).
    foreach ($root in @("$env:LOCALAPPDATA\Programs\Python\Python312",
                        "$env:ProgramFiles\Python312",
                        "${env:ProgramFiles(x86)}\Python312")) {
        $exe = Join-Path $root "python.exe"
        if ((Test-Path $exe) -and (Test-Python312 -Exe $exe)) {
            return [pscustomobject]@{ Exe = $exe; Arguments = @() }
        }
    }
    return $null
}

Step "Locating Python 3.12"
$python = Resolve-Python312
if (-not $python) {
    $winget = Get-Command winget.exe -ErrorAction SilentlyContinue
    if (-not $winget) {
        Start-Process "https://www.python.org/downloads/release/python-3120/"
        throw "Python 3.12 not found and winget is unavailable. Install Python 3.12 (tick 'Add to PATH'), then re-run."
    }
    Step "Installing Python 3.12 via winget"
    & $winget.Source install --id Python.Python.3.12 -e --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -ne 0) { throw "Python 3.12 installation failed." }
    $python = Resolve-Python312
}
if (-not $python) {
    throw "Python 3.12 was installed but could not be located afterwards. Open a new terminal and re-run, or install from python.org with 'Add Python to PATH' ticked."
}
$pyExe = $python.Exe
$pyArgs = $python.Arguments
Write-Host "Using: $pyExe $($pyArgs -join ' ')" -ForegroundColor Green

$venvDir = Join-Path $RepoRoot ".venv312"
$venvPython = Join-Path $venvDir "Scripts\python.exe"

Step "Creating .venv312 (leaving your existing .venv untouched)"
if (Test-Path $venvDir) {
    try { Remove-Item -Recurse -Force $venvDir -ErrorAction Stop }
    catch {
        throw "Could not clear an old .venv312 (a running AZMO/PowerShell window may be using it). Close all AZMO windows and run this again."
    }
}
& $pyExe @pyArgs -m venv $venvDir
if ($LASTEXITCODE -ne 0) { throw "Could not create the 3.12 virtual environment." }
if (-not (Test-Path $venvPython)) {
    throw "The virtual environment reported success but $venvPython is missing."
}

& $venvPython -m pip install --disable-pip-version-check --upgrade pip

if ($WithClone) {
    # Install PyTorch explicitly and FIRST. coqui-tts needs it, and the torch
    # download is the biggest and most failure-prone step - installing the CUDA
    # build up front (for the RTX 4080) is what makes the clone actually work.
    #
    # Several CUDA channels are tried in turn: a pinned index eventually stops
    # carrying wheels for newer torch releases, and the failure looks like an
    # unrelated resolver error. The CPU build is the last resort and is called
    # out loudly, because XTTS on CPU is slow enough to look broken.
    Step "Installing PyTorch (CUDA build for your GPU) - large download"
    $cudaInstalled = $false
    foreach ($channel in @("cu124", "cu121")) {
        Write-Host "  trying $channel"
        & $venvPython -m pip install --disable-pip-version-check torch torchaudio `
            --index-url "https://download.pytorch.org/whl/$channel"
        if ($LASTEXITCODE -eq 0) { $cudaInstalled = $true; break }
        Write-Warning "  $channel did not resolve; trying the next channel."
    }
    if (-not $cudaInstalled) {
        Write-Warning "No CUDA build installed. Falling back to the default (likely CPU) build."
        Write-Warning "The voice clone will be very slow until this is corrected."
        & $venvPython -m pip install --disable-pip-version-check torch torchaudio
        if ($LASTEXITCODE -ne 0) { throw "PyTorch install failed. See the pip output above." }
    }
}

$extras = if ($WithClone) { ".[dev,dsp,clone,listen]" } else { ".[dev,dsp]" }
Step "Installing AZMO with voice extras: $extras"
& $venvPython -m pip install --disable-pip-version-check -e $extras
if ($LASTEXITCODE -ne 0) { throw "Install failed. See the pip output above." }

if ($WithClone) {
    Step "Verifying PyTorch + the clone engine import"
    & $venvPython -c "import torch, TTS; print('  torch', torch.__version__, 'CUDA', torch.cuda.is_available())"
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "torch/TTS did not import cleanly - see output above."
    } else {
        # A CPU-only torch installs and imports perfectly and then makes the
        # voice unusable, so it is worth stating plainly rather than leaving it
        # in a line of version output.
        & $venvPython -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 3)"
        if ($LASTEXITCODE -eq 3) {
            Write-Warning "torch cannot see your GPU. XTTS will run on the CPU and will be far too slow."
            Write-Warning "Check your NVIDIA driver, then re-run this script to reinstall the CUDA build."
        }
    }
}

Step "Verifying the voice DSP is available"
& $venvPython -c "from azmo_mind.voice_dsp import dsp_available, world_available; import sys; print('  DSP (pedalboard):', dsp_available()); print('  Formants (pyworld):', world_available()); sys.exit(0 if dsp_available() else 1)"
if ($LASTEXITCODE -ne 0) {
    Write-Warning "DSP libraries still not importable. Check the pip output above."
    exit 1
}

if ($WithClone) {
    Step "Waking the cloned Azmodan voice (first run downloads XTTS, ~2 GB)"
    Write-Host "Uses data/voices/azmo_reference.wav. This can take a few minutes the first time."
    $env:COQUI_TOS_AGREED = "1"
    & $venvPython -m azmo_mind.cli say "Nephalem. The vessel now speaks with my voice." --config config/azmo.yaml
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "The clone test did not complete. Check that config speech.engine is 'auto' or 'clone'"
        Write-Warning "and that data/voices/azmo_reference.wav exists. A CUDA GPU is strongly recommended."
    } else {
        Write-Host "That was the cloned Azmodan voice + demonic DSP." -ForegroundColor Green
        Write-Host "Optional: run scripts/setup_voice.ps1 to rebuild a cleaner reference with Demucs."
    }
}

Step "Done"
Write-Host "AZMO's voice environment is ready in .venv312 (Python 3.12)." -ForegroundColor Green
Write-Host "Launch as usual with START_AZMO.bat - it now prefers .venv312 automatically." -ForegroundColor Green
if (-not $WithClone) {
    Write-Host "For the cloned Azmodan voice, re-run with:  .\scripts\setup_py312.ps1 -WithClone"
}
