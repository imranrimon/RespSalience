<#
Re-train the headline RespSalience model with checkpoint persistence enabled.

Why this exists: train_resp_salience_ssl.py originally wrote only per-fold prediction CSVs and
discarded the weights, so the model the paper recommends had no releasable artifact while the
baseline it argues against had twenty.

Two safety properties, both deliberate:

  1. STRICTLY SERIAL. One python process at a time, three in total (each trains all five folds
     internally). Host RAM, not GPU memory, is the binding constraint on this machine, and running
     these concurrently has hung it before. Do not parallelise this loop.

  2. WRITES TO A SEPARATE TAG. Output goes to pedrr_kfold_salience_ssl332ckpt_seed{0,1,2}, NOT to
     the canonical pedrr_kfold_salience_ssl332_seed{0,1,2}. Every headline number in the manuscript
     is re-derived from those canonical prediction files; overwriting them with a fresh run would
     silently move the published numbers. Compare first, promote deliberately or not at all.

Usage:  powershell -ExecutionPolicy Bypass -File scripts/train_salience_checkpoints.ps1
#>

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$logDir = Join-Path $root "outputs\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$master = Join-Path $logDir "salience_ckpt_$stamp.log"

function Log($msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $msg
    Write-Output $line
    Add-Content -Path $master -Value $line -Encoding utf8
}

Log "start: 3 seeds x 5 folds, serial, tag=_ssl332ckpt (canonical _ssl332 left untouched)"

# A detached process does NOT inherit the interactive shell's PATH: launching this via
# Win32_Process.Create resolved bare "python" to the WindowsApps shim and to a bare 3.14 install,
# neither of which has this project's torch. Resolve an interpreter that actually imports torch
# rather than trusting PATH.
$candidates = @(
    (Join-Path $env:USERPROFILE "anaconda3\envs\resp_env\python.exe"),
    (Join-Path $env:USERPROFILE "anaconda3\python.exe")
)
$python = $null
foreach ($c in $candidates) {
    if (Test-Path $c) {
        & $c -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>$null
        if ($LASTEXITCODE -eq 0) { $python = $c; break }
    }
}
if (-not $python) { Log "FATAL: no interpreter with a working CUDA torch found"; exit 1 }
Log "interpreter: $python"

# Host RAM, not GPU memory, is what has hung this machine before. Refuse to start on top of another
# heavy job rather than discovering the problem by locking the desktop.
$os = Get-CimInstance Win32_OperatingSystem
$freeGB = [math]::Round($os.FreePhysicalMemory / 1MB, 1)
Log "free host RAM: $freeGB GB"
if ($freeGB -lt 4.0) {
    Log "ABORT: only $freeGB GB free (need >= 4.0). Another job is likely running; finish it first."
    exit 1
}

$ok = 0
foreach ($seed in 0, 1, 2) {
    $seedLog = Join-Path $logDir "salience_ckpt_seed$seed`_$stamp.log"
    Log "seed $seed -> $seedLog"
    $sw = [Diagnostics.Stopwatch]::StartNew()

    & $python "scripts/train_resp_salience_ssl.py" `
        --seed $seed --kfold pedrr_kfold_main332 --tag _ssl332ckpt `
        *>&1 | Tee-Object -FilePath $seedLog

    $code = $LASTEXITCODE
    $sw.Stop()
    if ($code -ne 0) {
        Log "seed $seed FAILED (exit $code) after $([int]$sw.Elapsed.TotalMinutes) min - stopping"
        exit $code
    }
    $ok++
    Log "seed $seed done in $([int]$sw.Elapsed.TotalMinutes) min"
}

$ckpts = @(Get-ChildItem -Path (Join-Path $root "outputs\paper_runs") -Recurse -Filter "resp_salience_best.pt" -ErrorAction SilentlyContinue)
Log "complete: $ok/3 seeds, $($ckpts.Count) checkpoints written (expect 15)"
Log "NEXT: python scripts/compare_salience_runs.py   # before promoting anything"
