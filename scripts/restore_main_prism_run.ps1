# Restore the main PRISM-Breath run (the one artifact the reproducibility audit found missing).
#
# WHY THIS EXISTS
#   The audit (see manuscript/appendix_reproducibility.tex) found that every reported number
#   reproduces from the archive EXCEPT those depending on the main PRISM run. That run's
#   predictions are gone, and it cannot simply be retrained: no stored signal directory holds the
#   learned-flow representation together with depth for all 332 subjects.
#
#       data/pedrr_signals_v2                 332 subjects   depth live, flow = HAND-CRAFTED
#       data/pedrr_signals_learned            297 subjects   depth live, flow = learned
#       data/pedrr_signals_learned_flowonly   332 subjects   depth ZEROED, flow = learned
#
#   All k-fold runs share data/organized/kfold_learned (folds of 67/67/66/66/66); the 297-subject
#   runs are those manifests minus subjects whose signal file is absent.
#
# TWO OPTIONS. Pick one deliberately -- they answer different questions.
#
#   OPTION A (reproducible today, no extraction):
#     Re-baseline the main model on the 297-subject learned-flow cohort. This matches the
#     architecture the methods section describes and the cohort the framing analysis already uses,
#     and it makes the framing table internally matched. Cost: the eight ablation rows currently
#     reproduce on 332 and would need re-running on 297 to stay comparable.
#
#   OPTION B (restores the published 332-subject claim):
#     Re-extract the learned flow for the 35 subjects missing from data/pedrr_signals_learned,
#     giving a 332-subject learned-flow-with-depth set, then train. Restores the published cohort
#     size but costs an extraction pass before any training.
#
# GPU NOTES (learned the hard way on this machine)
#   - Run SINGLE-PROCESS. Concurrent 4-way training triggered a TDR reset (exit 0x40010004);
#     cap at 2-way if you parallelise anything.
#   - Launch from an unrestricted terminal or via Task Scheduler, not from an agent session.
#   - Both stages are resume-safe: -SkipExisting skips folds whose predictions already exist.
#
# USAGE
#   powershell -ExecutionPolicy Bypass -File scripts/restore_main_prism_run.ps1 -Option A
#   powershell -ExecutionPolicy Bypass -File scripts/restore_main_prism_run.ps1 -Option B

param(
    [ValidateSet('A','B')] [string]$Option = 'A',
    [string]$Tag = '',
    [int]$Epochs = 100,
    [int]$PretrainEpochs = 100,
    [int]$FreezeEncoderEpochs = 2,
    [switch]$SkipExisting,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$env:FOR_DISABLE_CONSOLE_CTRL_HANDLER = '1'

if ($Option -eq 'A') {
    $SignalDir = 'data/pedrr_signals_learned'
    if (-not $Tag) { $Tag = 'main_297' }
    Write-Host "Option A: 297-subject learned-flow cohort (no extraction needed)."
} else {
    $SignalDir = 'data/pedrr_signals_learned_332'
    if (-not $Tag) { $Tag = 'main_332' }
    Write-Host "Option B: re-extract learned flow to 332 subjects, then train."
    Write-Host "  Extraction step must produce $SignalDir with depth AND learned flow for all 332."
    Write-Host "  Start from scripts/build_efficientphys_flow_cleanfold.py (which built the learned set)"
    Write-Host "  and widen its subject list to the full manifest; it currently covers 297."
    if (-not (Test-Path $SignalDir)) {
        Write-Warning "$SignalDir does not exist yet. Build it before running the training stage."
        Write-Warning "Stopping here so this script does not silently train on the wrong signals."
        exit 2
    }
}

# Sanity: refuse to train on a signal set whose depth channel is dead, which is what makes
# pedrr_signals_learned_flowonly unusable for the full model.
$check = @"
import glob, sys, numpy as np
fs = sorted(glob.glob(r'$SignalDir/*.npz'))
if not fs:
    print('NO SIGNALS'); sys.exit(3)
live = {'depth': 0, 'flow': 0}
for f in fs[:40]:
    z = np.load(f, allow_pickle=True)
    for k in live:
        if k in z and not np.allclose(np.asarray(z[k], dtype=float), 0): live[k] += 1
print(f'{len(fs)} signals; live in first 40: {live}')
sys.exit(0 if live['depth'] > 0 and live['flow'] > 0 else 4)
"@
$check | python -
if ($LASTEXITCODE -ne 0) {
    Write-Error "Signal check failed for $SignalDir (exit $LASTEXITCODE). A zeroed depth channel means this set cannot train the full model."
    exit $LASTEXITCODE
}

$args = @(
    'scripts/run_prism_kfold.py',
    '--signal-dir', $SignalDir,
    '--tag', $Tag,
    '--epochs', $Epochs,
    '--pretrain-epochs', $PretrainEpochs,
    '--freeze-encoder-epochs', $FreezeEncoderEpochs
)
if ($SkipExisting) { $args += '--skip-existing' }

Write-Host "`n+ python $($args -join ' ')`n"
if ($DryRun) { Write-Host 'Dry run; nothing launched.'; exit 0 }
python @args
if ($LASTEXITCODE -ne 0) { Write-Error "Training failed (exit $LASTEXITCODE)."; exit $LASTEXITCODE }

Write-Host "`nDone. Now regenerate the tables from the new artifact:"
Write-Host "  python scripts/regenerate_supplementary_tables.py --prism-run pedrr_kfold_$Tag"
Write-Host "Then re-audit and update manuscript/appendix_reproducibility.tex."
