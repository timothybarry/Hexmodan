$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

function Step([string]$Message) {
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Refresh-Path {
    $machine = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $user = [Environment]::GetEnvironmentVariable("Path", "User")
    $env:Path = "$machine;$user"
}

function Find-Python {
    $command = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($command) {
        try {
            $valid = & $command.Source -c "import sys; print(int(sys.version_info >= (3, 10)))"
            if ($valid.Trim() -eq "1") { return $command.Source }
        } catch {}
    }

    $roots = @(
        "$env:LOCALAPPDATA\Programs\Python",
        "$env:ProgramFiles\Python"
    )
    foreach ($folder in $roots) {
        $items = Get-ChildItem $folder -Filter python.exe -Recurse -ErrorAction SilentlyContinue |
            Sort-Object FullName -Descending
        foreach ($item in $items) {
            try {
                $valid = & $item.FullName -c "import sys; print(int(sys.version_info >= (3, 10)))"
                if ($valid.Trim() -eq "1") { return $item.FullName }
            } catch {}
        }
    }
    return $null
}

function Find-Ollama {
    $command = Get-Command ollama.exe -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }

    $candidates = @(
        "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe",
        "$env:LOCALAPPDATA\Ollama\ollama.exe",
        "$env:ProgramFiles\Ollama\ollama.exe"
    )
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) { return $candidate }
    }
    return $null
}

try {
    Write-Host "============================================================" -ForegroundColor DarkRed
    Write-Host "                    AZMO MIND 0.2.5 - CLAUDE EDITION" -ForegroundColor Red
    Write-Host "============================================================" -ForegroundColor DarkRed
    Write-Host "Hardware output is disabled. Gestures remain simulations. AZMO will speak if a local voice engine exists."

    Step "Checking Python"
    $pythonExe = Find-Python
    if (-not $pythonExe) {
        $winget = Get-Command winget.exe -ErrorAction SilentlyContinue
        if (-not $winget) {
            Start-Process "https://www.python.org/downloads/windows/"
            throw "Python 3.10+ is missing. Install it with 'Add Python to PATH', then run START_AZMO.bat again."
        }

        Write-Host "Installing Python 3.11..."
        & $winget.Source install --id Python.Python.3.11 -e `
            --accept-package-agreements --accept-source-agreements
        if ($LASTEXITCODE -ne 0) {
            throw "Python installation failed with exit code $LASTEXITCODE."
        }
        Refresh-Path
        $pythonExe = Find-Python
        if (-not $pythonExe) {
            throw "Python installed, but Windows needs a restart. Restart and run START_AZMO.bat again."
        }
    }
    Write-Host "Python ready: $pythonExe" -ForegroundColor Green

    Step "Checking Ollama"
    $ollamaExe = Find-Ollama
    if (-not $ollamaExe) {
        Write-Host "Installing Ollama from its official installer..."
        $installer = Invoke-RestMethod -Uri "https://ollama.com/install.ps1"
        Invoke-Expression $installer
        Refresh-Path
        $ollamaExe = Find-Ollama
        if (-not $ollamaExe) {
            throw "Ollama installed, but Windows needs a restart. Restart and run START_AZMO.bat again."
        }
    }
    Write-Host "Ollama ready: $ollamaExe" -ForegroundColor Green

    Step "Starting Ollama"
    $ready = $false
    try {
        $null = Invoke-RestMethod "http://127.0.0.1:11434/api/tags" -TimeoutSec 3
        $ready = $true
    } catch {}

    if (-not $ready) {
        Start-Process -FilePath $ollamaExe -ArgumentList "serve" -WindowStyle Hidden
        for ($i = 0; $i -lt 20; $i++) {
            Start-Sleep -Milliseconds 750
            try {
                $null = Invoke-RestMethod "http://127.0.0.1:11434/api/tags" -TimeoutSec 2
                $ready = $true
                break
            } catch {}
        }
    }
    if (-not $ready) { throw "The Ollama service did not start." }

    Step "Downloading or verifying qwen3.5:9b"
    Write-Host "The first download is approximately 6.6 GB."
    & $ollamaExe pull qwen3.5:9b
    if ($LASTEXITCODE -ne 0) {
        throw "Model download failed with exit code $LASTEXITCODE."
    }

    Step "Preparing AZMO's Python environment"
    # Prefer the voice-ready 3.12 environment if SETUP_VOICE.bat created it.
    $venv312 = Join-Path $RepoRoot ".venv312\Scripts\python.exe"
    if (Test-Path $venv312) {
        $venvPython = $venv312
        Write-Host "Using the voice environment (.venv312)." -ForegroundColor Green
        & $venvPython -m pip install --disable-pip-version-check -e ".[dev]" | Out-Null
    } else {
        $venvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
        if (-not (Test-Path $venvPython)) {
            & $pythonExe -m venv (Join-Path $RepoRoot ".venv")
            if ($LASTEXITCODE -ne 0) { throw "Could not create the virtual environment." }
        }
        & $venvPython -m pip install --disable-pip-version-check --upgrade pip
        if ($LASTEXITCODE -ne 0) { throw "Could not update pip." }
        & $venvPython -m pip install --disable-pip-version-check -e ".[dev]"
        if ($LASTEXITCODE -ne 0) { throw "Could not install AZMO Mind." }

        Step "Installing voice modulation (azmo-voice DSP)"
        Write-Host "Adds the demonic voice processing so AZMO is modulated, not a plain robot."
        & $venvPython -m pip install --disable-pip-version-check -e ".[dsp]"
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "Voice DSP extras did not install."
            Write-Warning "This is common on brand-new Python builds without prebuilt audio wheels."
            Write-Warning "AZMO will still run and speak, but WITHOUT modulation."
            Write-Warning "For the demonic voice, run SETUP_VOICE.bat (installs Python 3.12)."
        }
    }

    Step "Running diagnostics"
    & $venvPython -m azmo_mind.cli doctor --warmup
    if ($LASTEXITCODE -ne 0) { throw "Diagnostics failed." }

    Step "Awakening AZMO"
    Write-Host "Type /help for commands and /quit when finished." -ForegroundColor Yellow
    & $venvPython -m azmo_mind.cli chat --skip-warmup
    exit 0
}
catch {
    Write-Host ""
    Write-Host "AZMO SETUP ERROR" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Hardware control remains disabled."
    exit 1
}
