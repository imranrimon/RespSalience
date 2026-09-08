# Backfill checkpoints for SSL-variant seeds 3-5, which were first run with --no-save-checkpoints.
# Needed so the rejected-variant arm of the null ablation is complete in the release bundle.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = Join-Path $env:USERPROFILE "anaconda3\envs\resp_env\python.exe"
if (-not (Test-Path $py)) { $py = Join-Path $env:USERPROFILE "anaconda3\python.exe" }
foreach ($s in 3,4,5) {
  & $py "scripts/train_resp_salience_ssl.py" --seed $s --kfold pedrr_kfold_main332 --tag _ssl332det
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
& $py "scripts/build_release_bundle.py" --zip
exit $LASTEXITCODE
