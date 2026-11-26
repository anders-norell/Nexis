# Nexis v3 – installation och körning på Windows (PowerShell).
# Kör från katalogen v3:   powershell -ExecutionPolicy Bypass -File setup_windows.ps1 [steg]
#   steg: install | test | baseline | baseline2 | extract | steer_beta | steer_test | chat | all   (default: install)
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
if ($step -eq "extract") {
  & $py extract.py --split test  | Tee-Object -Append results\extract.log
  & $py extract.py --split train | Tee-Object -Append results\extract.log
}
if ($step -eq "steer_beta") { & $py steer_eval.py beta | Tee-Object -Append results\steer.log }
if ($step -eq "steer_test") { & $py steer_eval.py test | Tee-Object -Append results\steer.log }
if ($step -eq "steer_dose") { & $py steer_eval.py dose | Tee-Object -Append results\steer.log }
if ($step -eq "extract3") {
  & $py extract.py --split test3  | Tee-Object -Append results\extract3.log
  & $py extract.py --split train3 | Tee-Object -Append results\extract3.log
}
if ($step -eq "extract3b") { & $py extract.py --split test3b | Tee-Object -Append results\extract3b.log }
if ($step -eq "extract_tok") {
  & $py extract_tokens.py --split tok_test  | Tee-Object -Append results\extract_tok.log
  & $py extract_tokens.py --split tok_train | Tee-Object -Append results\extract_tok.log
}
if ($step -eq "base4") { & $py extract_v4.py --split base4 | Tee-Object -Append results\extract_v4.log }
if ($step -eq "base4b") { & $py extract_v4.py --split base4b | Tee-Object -Append results\extract_v4.log }
if ($step -eq "s4auto") {
  # Sprint 4: base4b -> F1b-kontroll -> (bara om F1b klaras) test4 och train4. Avbrutet arbete återupptas.
  & $py extract_v4.py --split base4b | Tee-Object -Append results\s4auto.log
  & $py f1b_check.py results\v4_data | Tee-Object -Append results\s4auto.log
  if ($LASTEXITCODE -eq 0) {
    "F1b klarad - fortsätter med test4 och train4" | Tee-Object -Append results\s4auto.log
    & $py extract_v4.py --split test4  | Tee-Object -Append results\s4auto.log
    & $py extract_v4.py --split train4 | Tee-Object -Append results\s4auto.log
    "S4AUTO KLART" | Tee-Object -Append results\s4auto.log
  } else {
    "F1b EJ klarad - stannar enligt förregistreringen" | Tee-Object -Append results\s4auto.log
  }
}
if ($step -eq "s4auto2") {
  # Sprint 4: base4c -> F1c-kontroll -> (bara om F1c klaras) test4 och train4. Avbrutet arbete återupptas.
  & $py extract_v4.py --split base4c | Tee-Object -Append results\s4auto2.log
  & $py f1c_check.py results\v4_data | Tee-Object -Append results\s4auto2.log
  if ($LASTEXITCODE -eq 0) {
    "F1c klarad - fortsätter med test4 och train4" | Tee-Object -Append results\s4auto2.log
    & $py extract_v4.py --split test4  | Tee-Object -Append results\s4auto2.log
    & $py extract_v4.py --split train4 | Tee-Object -Append results\s4auto2.log
    "S4AUTO2 KLART" | Tee-Object -Append results\s4auto2.log
  } else {
    "F1c EJ klarad - stannar enligt förregistreringen" | Tee-Object -Append results\s4auto2.log
  }
}
if ($step -eq "extract4") {
  & $py extract_v4.py --split test4  | Tee-Object -Append results\extract_v4.log
  & $py extract_v4.py --split train4 | Tee-Object -Append results\extract_v4.log
}
if ($step -eq "chat") { & $py chat.py }
