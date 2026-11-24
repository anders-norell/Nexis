# Nexis v3 – installation och körning på Windows (PowerShell).
# Kör från katalogen v3:   powershell -ExecutionPolicy Bypass -File setup_windows.ps1 [steg]
#   steg: install | test | baseline | baseline2 | chat | all   (default: install)
param([string]$step = "install")
$ErrorActionPreference = "Continue"  # PS 5.1: stderr från python får inte avbryta
$env:PYTHONUTF8 = "1"
if (!(Test-Path .venv)) { py -3 -m venv .venv }
$py = ".\.venv\Scripts\python.exe"
if ($step -in @("install","all")) {
  & $py -m pip install --upgrade pip
  & $py -m pip install torch --index-url https://download.pytorch.org/whl/cpu
  & $py -m pip install -r requirements.txt
}
if ($step -in @("test","all")) {
  & $py -m pytest -q                                   # testmodellen
  $env:NEXIS_REAL_MODEL = "Qwen/Qwen2.5-1.5B-Instruct" # laddar ner ca 3 GB första gången
  & $py -m pytest -q tests/test_invariants.py | Tee-Object results\tests_real_model.txt
  Remove-Item Env:\NEXIS_REAL_MODEL
}
if ($step -in @("baseline","all")) {
  & $py baseline.py --n 100 --seed 0 --out results\baseline_s0.json | Tee-Object results\baseline_s0.log
}
if ($step -eq "baseline2") {
  & $py baseline.py --texts 2 --n 100 --seed 1000 --out results\baseline2_s1000.json | Tee-Object results\baseline2_s1000.log
}
if ($step -eq "chat") { & $py chat.py }
