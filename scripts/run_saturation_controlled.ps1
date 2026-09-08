# Controlled arm of the saturation-severity analysis: belt signal, progressively low-passed.
# Serial by construction (one process). Launch behind await_gpu_then_train_salience.ps1 so it never
# starts on top of another training job.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = Join-Path $env:USERPROFILE "anaconda3\envs\resp_env\python.exe"
if (-not (Test-Path $py)) { $py = Join-Path $env:USERPROFILE "anaconda3\python.exe" }
& $py "scripts/saturation_severity.py" --arm controlled --seeds 3 --epochs 30
exit $LASTEXITCODE
