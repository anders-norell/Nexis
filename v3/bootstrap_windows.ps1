# Nexis v3 – hämtar senaste koden och kör ett steg. Klistra in i PowerShell:
#   $s='install'; irm https://raw.githubusercontent.com/anders-norell/Nexis/main/v3/bootstrap_windows.ps1 | iex
# där $s är install | test | baseline | chat.
# Koden hamnar i Dokument\nexis3 (din venv och nedladdad modell behålls mellan körningar).
$ErrorActionPreference = "Stop"
if (-not $s) { $s = "install" }
$root = Join-Path ([Environment]::GetFolderPath("MyDocuments")) "nexis3"
New-Item -ItemType Directory -Force $root | Out-Null
Set-Location $root

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
  Write-Host "Python saknas - installerar Python 3.12 med winget..."
  winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements
  Write-Host "Stang och oppna PowerShell igen och kor samma rad en gang till."
  return
}

# Hämta senaste koden (behåll .venv och results)
$zip = Join-Path $env:TEMP "nexis_main.zip"
Invoke-WebRequest https://github.com/anders-norell/Nexis/archive/refs/heads/main.zip -OutFile $zip
$tmp = Join-Path $env:TEMP "nexis_main"
if (Test-Path $tmp) { Remove-Item -Recurse -Force $tmp }
Expand-Archive $zip $tmp
$src = Get-ChildItem $tmp | Select-Object -First 1
Get-ChildItem (Join-Path $src.FullName "v3") | Where-Object { $_.Name -ne "results" } |
  ForEach-Object { Copy-Item -Recurse -Force $_.FullName $root }
New-Item -ItemType Directory -Force (Join-Path $root "results") | Out-Null

& powershell -ExecutionPolicy Bypass -File (Join-Path $root "setup_windows.ps1") $s
