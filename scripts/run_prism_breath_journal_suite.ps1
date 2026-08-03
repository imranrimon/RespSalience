[CmdletBinding()]
param(
    [string]$Python = "C:\Users\rimon\anaconda3\envs\resp_env\python.exe",
    [string]$RunId = "",
    [string]$OutputBase = "outputs\paper_runs",
    [string]$SignalDir = "data\paper_signals",
    [int[]]$Seeds = @(42, 52, 62),
    [int]$BestSeed = 42,
    [int]$SeqLen = 300,
    [int]$DModel = 64,
    [int]$BatchSize = 16,
    [int]$PretrainEpochs = 100,
    [int]$TrainEpochs = 100,
    [double]$PretrainLr = 1e-3,
    [double]$TrainLr = 3e-4,
    [double]$WeightDecay = 1e-4,
    [double]$ContrastiveWeight = 0.15,
    [string]$DisagreementBins = "0.8,0.9",
    [switch]$ForceMainEvalOnly,
    [switch]$SkipManifestBuild,
    [switch]$SkipSignalExtraction,
    [switch]$SkipMainRuns,
    [switch]$SkipAblations,
    [switch]$SkipOfficialAirBaselines,
    [switch]$SkipLegacyBaselines,
    [switch]$SkipExpandedPaperBaselines,
    [switch]$SkipPaperTables,
    [string]$OfficialAirBaselines = "AIRFlowNet-Official,DeepPhys-Official,MTTS-CAN-Official,EfficientPhys-Official",
    [int]$OfficialAirTrainEpochs = 15,
    [int]$OfficialAirBatchSize = 4,
    [double]$OfficialAirLr = 1e-3,
    [string]$ExpandedBaselinePreset = "paper_full",
    [int]$ExpandedBaselineLimit = 0
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Test-AllPaths {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Paths
    )
    foreach ($path in $Paths) {
        if (-not (Test-Path -LiteralPath $path)) {
            return $false
        }
    }
    return $true
}

function Invoke-PythonStep {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )
    $cmdText = ($Arguments | ForEach-Object {
        if ($_ -match "\s") { '"{0}"' -f $_ } else { $_ }
    }) -join " "
    Write-Host "[RUN] $Python $cmdText"
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE"
    }
}

function Invoke-OrSkip {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Label,
        [Parameter(Mandatory = $true)]
        [string[]]$Outputs,
        [Parameter(Mandatory = $true)]
        [scriptblock]$Action
    )
    if (Test-AllPaths -Paths $Outputs) {
        Write-Host "[SKIP] $Label"
    }
    else {
        & $Action
    }
}

function Assert-PathExists {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [Parameter(Mandatory = $true)]
        [string]$Label
    )
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "$Label not found: $Path"
    }
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

if ([string]::IsNullOrWhiteSpace($RunId)) {
    $RunId = Get-Date -Format "yyyyMMdd_HHmmss"
}

$Root = Join-Path $ProjectRoot (Join-Path $OutputBase $RunId)
$Signals = Join-Path $ProjectRoot $SignalDir
$TranscriptPath = Join-Path $Root "run.log"

New-Item -ItemType Directory -Force -Path $Root, $Signals | Out-Null

$env:YOLO_CONFIG_DIR = Join-Path $ProjectRoot ".ultralytics"
$env:WANDB_MODE = "disabled"
$env:WANDB_SILENT = "true"
$env:HF_HUB_DISABLE_TELEMETRY = "1"
New-Item -ItemType Directory -Force -Path $env:YOLO_CONFIG_DIR | Out-Null

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python executable not found: $Python"
}

if ($ForceMainEvalOnly) {
    $SkipManifestBuild = $true
    $SkipSignalExtraction = $true
    $SkipMainRuns = $false
    $SkipAblations = $true
    $SkipOfficialAirBaselines = $true
    $SkipLegacyBaselines = $true
    $SkipExpandedPaperBaselines = $true
    Write-Host "[MODE] ForceMainEvalOnly enabled: re-running only PRISM val/calibration/test from existing checkpoints."
}

$transcriptStarted = $false
Push-Location $ProjectRoot
try {
    try {
        Start-Transcript -Path $TranscriptPath -Append | Out-Null
        $transcriptStarted = $true
    }
    catch {
        Write-Warning "Could not start transcript at $TranscriptPath. Continuing without transcript. $_"
    }

    $TrainManifest = "data\organized\respiratory_benchmark_train.json"
    $ValManifest = "data\organized\respiratory_benchmark_val.json"
    $TestManifest = "data\organized\respiratory_benchmark_test.json"

    if (-not $SkipManifestBuild) {
        Invoke-PythonStep @(
            "scripts/build_respiratory_benchmark_manifest.py",
            "--base-dir", "data",
            "--output-dir", "data\organized"
        )
    }
    else {
        Write-Host "[SKIP] manifest build"
    }

    if (-not $SkipSignalExtraction) {
        Invoke-PythonStep @(
            "scripts/extract_benchmark_signals.py",
            "--manifest", $TrainManifest,
            "--output", $SignalDir,
            "--config", "src\config\test_infer.yaml",
            "--checkpoint", "data\models\VIRENet_Coarse2fine_Body_best.pth"
        )
        Invoke-PythonStep @(
            "scripts/extract_benchmark_signals.py",
            "--manifest", $ValManifest,
            "--output", $SignalDir,
            "--config", "src\config\test_infer.yaml",
            "--checkpoint", "data\models\VIRENet_Coarse2fine_Body_best.pth"
        )
        Invoke-PythonStep @(
            "scripts/extract_benchmark_signals.py",
            "--manifest", $TestManifest,
            "--output", $SignalDir,
            "--config", "src\config\test_infer.yaml",
            "--checkpoint", "data\models\VIRENet_Coarse2fine_Body_best.pth"
        )
    }
    else {
        Write-Host "[SKIP] signal extraction"
    }

    if (-not $SkipMainRuns) {
        foreach ($seed in $Seeds) {
            $seedRoot = Join-Path $Root "main_seed$seed"
            $pretrainDir = Join-Path $seedRoot "pretrain"
            $trainDir = Join-Path $seedRoot "train"
            $valDir = Join-Path $seedRoot "val_eval"
            $conformalDir = Join-Path $seedRoot "conformal"
            $testUncalDir = Join-Path $seedRoot "test_eval_uncalibrated"
            $testCalDir = Join-Path $seedRoot "test_eval"
            $pretrainCkpt = Join-Path $pretrainDir "prism_breath_pretrained.pt"
            $trainCkpt = Join-Path $trainDir "prism_breath_best.pt"

            if ($ForceMainEvalOnly) {
                Assert-PathExists -Path $trainCkpt -Label "PRISM checkpoint for main_seed$seed"
                Write-Host "[SKIP] main_seed$seed pretrain (ForceMainEvalOnly)"
                Write-Host "[SKIP] main_seed$seed train (ForceMainEvalOnly)"
            }
            else {
                Invoke-OrSkip -Label "main_seed$seed pretrain" -Outputs @(
                    $pretrainCkpt,
                    (Join-Path $pretrainDir "pretraining_history.json")
                ) -Action {
                    Invoke-PythonStep @(
                        "scripts/pretrain_prism_breath.py",
                        "--signal-dir", $SignalDir,
                        "--train-manifest", $TrainManifest,
                        "--val-manifest", $ValManifest,
                        "--output-dir", $pretrainDir,
                        "--model-variant", "prism_breath",
                        "--seq-len", "$SeqLen",
                        "--d-model", "$DModel",
                        "--batch-size", "$BatchSize",
                        "--epochs", "$PretrainEpochs",
                        "--lr", "$PretrainLr",
                        "--contrastive-weight", "$ContrastiveWeight",
                        "--seed", "$seed"
                    )
                }

                Invoke-OrSkip -Label "main_seed$seed train" -Outputs @(
                    $trainCkpt,
                    (Join-Path $trainDir "training_history.json")
                ) -Action {
                    Invoke-PythonStep @(
                        "scripts/train_prism_breath.py",
                        "--signal-dir", $SignalDir,
                        "--train-manifest", $TrainManifest,
                        "--val-manifest", $ValManifest,
                        "--output-dir", $trainDir,
                        "--model-variant", "prism_breath",
                        "--seq-len", "$SeqLen",
                        "--d-model", "$DModel",
                        "--num-experts", "3",
                        "--num-prototypes", "6",
                        "--codebook-size", "12",
                        "--epochs", "$TrainEpochs",
                        "--batch-size", "$BatchSize",
                        "--lr", "$TrainLr",
                        "--weight-decay", "$WeightDecay",
                        "--encoder-pretrained", $pretrainCkpt,
                        "--freeze-encoder-epochs", "2",
                        "--modality-dropout-prob", "0.1",
                        "--seed", "$seed"
                    )
                }
            }

            $runValEval = {
                Invoke-PythonStep @(
                    "scripts/evaluate_prism_breath.py",
                    "--signal-dir", $SignalDir,
                    "--manifest", $ValManifest,
                    "--checkpoint", $trainCkpt,
                    "--output-dir", $valDir,
                    "--batch-size", "$BatchSize",
                    "--disagreement-bins", $DisagreementBins,
                    "--counterfactual-limit", "50"
                )
            }
            if ($ForceMainEvalOnly) {
                & $runValEval
            }
            else {
                Invoke-OrSkip -Label "main_seed$seed val_eval" -Outputs @(
                    (Join-Path $valDir "prism_breath_summary.json"),
                    (Join-Path $valDir "prism_breath_predictions.csv")
                ) -Action $runValEval
            }

            $runConformal = {
                Invoke-PythonStep @(
                    "scripts/calibrate_conformal.py",
                    "--samples", (Join-Path $valDir "prism_breath_predictions.csv"),
                    "--output", (Join-Path $conformalDir "prism_qhat.csv"),
                    "--alpha", "0.1",
                    "--use-all-samples",
                    "--group-by-source",
                    "--condition-on-disagreement",
                    "--split-unit", "subject",
                    "--disagreement-bins", $DisagreementBins
                )
            }
            if ($ForceMainEvalOnly) {
                & $runConformal
            }
            else {
                Invoke-OrSkip -Label "main_seed$seed conformal" -Outputs @(
                    (Join-Path $conformalDir "prism_qhat.csv")
                ) -Action $runConformal
            }

            $runTestUncal = {
                Invoke-PythonStep @(
                    "scripts/evaluate_prism_breath.py",
                    "--signal-dir", $SignalDir,
                    "--manifest", $TestManifest,
                    "--checkpoint", $trainCkpt,
                    "--output-dir", $testUncalDir,
                    "--batch-size", "$BatchSize",
                    "--disagreement-bins", $DisagreementBins,
                    "--counterfactual-limit", "100"
                )
            }
            if ($ForceMainEvalOnly) {
                & $runTestUncal
            }
            else {
                Invoke-OrSkip -Label "main_seed$seed test_eval_uncalibrated" -Outputs @(
                    (Join-Path $testUncalDir "prism_breath_summary.json"),
                    (Join-Path $testUncalDir "prism_breath_predictions.csv")
                ) -Action $runTestUncal
            }

            $runTestCal = {
                Invoke-PythonStep @(
                    "scripts/evaluate_prism_breath.py",
                    "--signal-dir", $SignalDir,
                    "--manifest", $TestManifest,
                    "--checkpoint", $trainCkpt,
                    "--output-dir", $testCalDir,
                    "--batch-size", "$BatchSize",
                    "--calibration-csv", (Join-Path $conformalDir "prism_qhat.csv"),
                    "--condition-on-disagreement",
                    "--disagreement-bins", $DisagreementBins,
                    "--counterfactual-limit", "100"
                )
            }
            if ($ForceMainEvalOnly) {
                & $runTestCal
            }
            else {
                Invoke-OrSkip -Label "main_seed$seed test_eval" -Outputs @(
                    (Join-Path $testCalDir "prism_breath_summary.json"),
                    (Join-Path $testCalDir "prism_breath_predictions.csv")
                ) -Action $runTestCal
            }
        }
    }
    else {
        Write-Host "[SKIP] main PRISM-Breath runs"
    }

    if (-not $SkipAblations) {
        foreach ($seed in $Seeds) {
            Invoke-PythonStep @(
                "scripts/run_prism_breath_study.py",
                "--signal-dir", $SignalDir,
                "--train-manifest", $TrainManifest,
                "--val-manifest", $ValManifest,
                "--test-manifest", $TestManifest,
                "--output-root", (Join-Path $Root "study_seed$seed"),
                "--variants", "prism_breath", "prism_shared_trunk", "prism_no_scan", "prism_no_spectral", "prism_no_codebook", "prism_no_prototypes", "airflow_compact",
                "--seq-len", "$SeqLen",
                "--d-model", "$DModel",
                "--batch-size", "$BatchSize",
                "--pretrain-epochs", "$PretrainEpochs",
                "--train-epochs", "$TrainEpochs",
                "--disagreement-bins", $DisagreementBins,
                "--seed", "$seed",
                "--skip-existing"
            )
        }
    }
    else {
        Write-Host "[SKIP] PRISM ablation study"
    }

    if (-not $SkipOfficialAirBaselines) {
        foreach ($seed in $Seeds) {
            Invoke-PythonStep @(
                "scripts/run_official_air_baselines.py",
                "--train-manifest", $TrainManifest,
                "--val-manifest", $ValManifest,
                "--test-manifest", $TestManifest,
                "--output-root", (Join-Path $Root "official_air_seed$seed"),
                "--baselines", $OfficialAirBaselines,
                "--train-epochs", "$OfficialAirTrainEpochs",
                "--batch-size", "$OfficialAirBatchSize",
                "--lr", "$OfficialAirLr",
                "--seed", "$seed",
                "--skip-existing"
            )
        }
    }
    else {
        Write-Host "[SKIP] official AIR baselines"
    }

    if (-not $SkipLegacyBaselines) {
        foreach ($seed in $Seeds) {
            Invoke-PythonStep @(
                "scripts/run_legacy_canonical_benchmark.py",
                "--signal-dir", $SignalDir,
                "--train-manifest", $TrainManifest,
                "--val-manifest", $ValManifest,
                "--test-manifest", $TestManifest,
                "--output-root", (Join-Path $Root "legacy_seed$seed"),
                "--baselines", "Flow-Only,Depth-Only,Fused-VMD,Contrastive-Finetune,Transformer-Fusion,DeepPhys",
                "--ssl-epochs", "$PretrainEpochs",
                "--finetune-epochs", "$TrainEpochs",
                "--batch-size", "$BatchSize",
                "--ssl-lr", "$PretrainLr",
                "--finetune-lr", "$TrainLr",
                "--weight-decay", "$WeightDecay",
                "--seq-len", "$SeqLen",
                "--d-model", "$DModel",
                "--seed", "$seed",
                "--disable-vlm",
                "--skip-existing"
            )
        }
    }
    else {
        Write-Host "[SKIP] legacy baselines"
    }

    if (-not $SkipExpandedPaperBaselines) {
        foreach ($seed in $Seeds) {
            Invoke-PythonStep @(
                "scripts/run_expanded_paper_baselines.py",
                "--train-manifest", $TrainManifest,
                "--val-manifest", $ValManifest,
                "--test-manifest", $TestManifest,
                "--output-root", (Join-Path $Root "expanded_seed$seed"),
                "--preset", $ExpandedBaselinePreset,
                "--seed", "$seed",
                "--limit", "$ExpandedBaselineLimit",
                "--skip-existing"
            )
        }
    }
    else {
        Write-Host "[SKIP] expanded paper baselines"
    }

    if (-not $SkipPaperTables) {
        Invoke-PythonStep @(
            "scripts/generate_prism_breath_paper_tables.py",
            "--run-root", $Root,
            "--output-dir", (Join-Path $Root "paper_tables"),
            "--best-seed", "$BestSeed"
        )
    }
    else {
        Write-Host "[SKIP] paper table generation"
    }

    Write-Host ""
    Write-Host "Journal suite complete."
    Write-Host "Run root: $Root"
}
finally {
    if ($transcriptStarted) {
        try {
            Stop-Transcript | Out-Null
        }
        catch {
        }
    }
    Pop-Location
}
