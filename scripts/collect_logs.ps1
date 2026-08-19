# Collect crash diagnostics into azmo_crash_report.txt for review.
# Pure ASCII only. Read-only: gathers logs, changes nothing.

$ErrorActionPreference = "SilentlyContinue"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$out = Join-Path $RepoRoot "azmo_crash_report.txt"

"AZMO crash report  -  $(Get-Date)" | Out-File $out -Encoding utf8
function Section([string]$t) { "`n===== $t =====" | Out-File $out -Append -Encoding utf8 }

Section "System + CPU + GPU"
"OS: $((Get-CimInstance Win32_OperatingSystem).Caption)" | Out-File $out -Append -Encoding utf8
Get-CimInstance Win32_Processor | Select-Object Name, NumberOfCores | Format-List | Out-File $out -Append -Encoding utf8
"RAM(GB): $([math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB,1))" | Out-File $out -Append -Encoding utf8
nvidia-smi 2>&1 | Out-File $out -Append -Encoding utf8

Section "Kernel-Power 41 (unexpected power loss / hard reset)"
Get-WinEvent -FilterHashtable @{LogName='System'; Id=41} -MaxEvents 5 | Format-List TimeCreated, Message | Out-File $out -Append -Encoding utf8

Section "WHEA hardware errors (power / thermal / bus)"
Get-WinEvent -FilterHashtable @{LogName='System'; ProviderName='Microsoft-Windows-WHEA-Logger'} -MaxEvents 8 | Format-List TimeCreated, Message | Out-File $out -Append -Encoding utf8

Section "WHEA error detail (which component - CPU / PCIe / memory)"
Get-WinEvent -FilterHashtable @{LogName='System'; ProviderName='Microsoft-Windows-WHEA-Logger'} -MaxEvents 4 | ForEach-Object {
    "`n--- $($_.TimeCreated)  (Event Id $($_.Id)) ---" | Out-File $out -Append -Encoding utf8
    try {
        $x = [xml]$_.ToXml()
        $x.Event.EventData.Data | ForEach-Object {
            if ($_.'#text') { "  $($_.Name) = $($_.'#text')" | Out-File $out -Append -Encoding utf8 }
        }
    } catch {}
}

Section "Correlate: do reboot times match WHEA times?"
"Kernel-Power 41 times:" | Out-File $out -Append -Encoding utf8
Get-WinEvent -FilterHashtable @{LogName='System'; Id=41} -MaxEvents 6 | Select-Object -ExpandProperty TimeCreated | Out-File $out -Append -Encoding utf8
"WHEA times:" | Out-File $out -Append -Encoding utf8
Get-WinEvent -FilterHashtable @{LogName='System'; ProviderName='Microsoft-Windows-WHEA-Logger'} -MaxEvents 8 | Select-Object -ExpandProperty TimeCreated | Out-File $out -Append -Encoding utf8

Section "Display driver / TDR (nvlddmkm resets)"
Get-WinEvent -LogName System -MaxEvents 500 | Where-Object { $_.Message -match 'nvlddmkm|display driver|Timeout Detected|TDR' } | Select-Object TimeCreated, Id, ProviderName -First 8 | Format-List | Out-File $out -Append -Encoding utf8

Section "Thermal / temperature events"
Get-WinEvent -LogName System -MaxEvents 500 | Where-Object { $_.Message -match 'thermal|temperature|overheat' } | Select-Object TimeCreated, Message -First 8 | Format-List | Out-File $out -Append -Encoding utf8

Section "Recent critical/error System events"
Get-WinEvent -FilterHashtable @{LogName='System'; Level=1,2} -MaxEvents 30 | Format-Table TimeCreated, Id, ProviderName, LevelDisplayName -Auto | Out-File $out -Append -Encoding utf8

Section "Ollama server log (tail)"
$olog = "$env:LOCALAPPDATA\Ollama\server.log"
if (Test-Path $olog) { Get-Content $olog -Tail 60 | Out-File $out -Append -Encoding utf8 }
else { "Ollama log not found at $olog" | Out-File $out -Append -Encoding utf8 }

Write-Host ""
Write-Host "Wrote crash report to:" -ForegroundColor Green
Write-Host "  $out"
Write-Host "Send that file (or tell Claude it's ready) so it can read it."
