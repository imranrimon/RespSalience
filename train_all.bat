@echo off
REM ================================================================
REM  FULL EXPERIMENTAL SUITE - Respiratory Rate Estimation
REM  Run from the repository root (all paths below are relative to it)
REM  Activate env first:    conda activate resp_env
REM ================================================================
SETLOCAL ENABLEDELAYEDEXPANSION

SET LOG=data\training_logs
SET MODELS=data\models
SET RESULTS=data\ablation_results

MD "%LOG%"          2>nul
MD "%MODELS%\denoisers" 2>nul
MD "%MODELS%\edge"  2>nul

ECHO.
ECHO ================================================================
ECHO  Respiratory Rate Estimation - Full Experimental Suite
ECHO  Started: %DATE% %TIME%
ECHO ================================================================
ECHO.

python -c "import torch,cv2,scipy,numpy; print('GPU:', torch.cuda.is_available())"
IF ERRORLEVEL 1 ( ECHO ERROR: Run: conda activate respiratory_env & EXIT /B 1 )
ECHO.

REM ================================================================
REM PHASE 1 - Signal Extraction  (Fused-VMD, classical, no GPU)
REM Runs run_ablation_study.py --save-signals to extract all .npz
REM Output: data/ablation_results/signals/*.npz
REM SKIP if signals already found.
REM ================================================================
ECHO [PHASE 1] Checking for extracted signals...
DIR /b "%RESULTS%\signals\*.npz" >nul 2>&1
IF ERRORLEVEL 1 (
    ECHO   Extracting signals from all videos...
    python scripts/run_ablation_study.py --save-signals > "%LOG%\phase1_signals.log" 2>&1
    IF ERRORLEVEL 1 ( ECHO   WARNING - check %LOG%\phase1_signals.log ) ELSE ( ECHO   Done. )
) ELSE (
    ECHO   SKIP - signals already exist.
)
ECHO.

REM ================================================================
REM PHASE 2 - Train 7 DL Signal Denoisers
REM 80/20 train/val split + early stopping (patience=7)
REM Output: data/models/denoisers/{model}_best.pth
REM Each model is INDEPENDENT - open 7 terminals to run in parallel.
REM ================================================================
ECHO [PHASE 2] Training DL Signal Denoisers - 7 models, 50 epochs each...
ECHO   Checking existing checkpoints in data\models\denoisers\
DIR /b "data\models\denoisers\*.pth" 2>nul || ECHO   (none yet)
ECHO.
CALL :DENOISER unet
CALL :DENOISER wavenet
CALL :DENOISER transformer
CALL :DENOISER diffusion
CALL :DENOISER cnn-bilstm
CALL :DENOISER tfmae
CALL :DENOISER mamba
ECHO   All denoisers done.
ECHO.

REM ================================================================
REM PHASE 3 - Hyperparameter Sweep
REM 8 configs: d_model x lr x recon_weight  (50 epochs each)
REM Output: data/ablation_results/hp_sweep_results.json + .csv
REM ================================================================
ECHO [PHASE 3] Hyperparameter sweep...
IF EXIST "%RESULTS%\hp_sweep_results.json" (
    ECHO   SKIP - hp_sweep_results.json already exists.
) ELSE (
    python scripts/hp_sweep.py --quick --epochs 50 > "%LOG%\phase3_hpsweep.log" 2>&1
    IF ERRORLEVEL 1 ( ECHO   WARNING - check %LOG%\phase3_hpsweep.log ) ELSE ( ECHO   Done. )
)
ECHO.

REM ================================================================
REM PHASE 4 - Train Quad-Modal Fusion Transformer
REM Stage 1: SPMat-BT contrastive pretraining (50 epochs)
REM   Output: data/models/spmat_bt.pth
REM Stage 2: VLM-Gated fine-tuning (30 epochs, 90/10 val split,
REM          early stopping patience=10, saves best VAL MAE)
REM   Output: data/models/vlm_gated_fused.pth
REM ================================================================
ECHO [PHASE 4] Training Quad-Modal Fusion Transformer...
IF EXIST "%MODELS%\vlm_gated_fused.pth" (
    ECHO   SKIP - vlm_gated_fused.pth already exists.
) ELSE (
    IF NOT EXIST "%MODELS%\spmat_bt.pth" (
        ECHO   Stage 1/2: SPMat-BT pretraining - 50 epochs...
        python scripts/train_vlm_fusion.py --stage pretrain --epochs 50 > "%LOG%\phase4_pretrain.log" 2>&1
        IF ERRORLEVEL 1 ( ECHO   WARNING - check %LOG%\phase4_pretrain.log ) ELSE ( ECHO   Stage 1 done. )
    ) ELSE (
        ECHO   Stage 1/2: SKIP - spmat_bt.pth exists.
    )
    ECHO   Stage 2/2: VLM-Gated fine-tuning - 30 epochs...
    python scripts/train_vlm_fusion.py --stage finetune --ft_epochs 30 > "%LOG%\phase4_finetune.log" 2>&1
    IF ERRORLEVEL 1 ( ECHO   WARNING - check %LOG%\phase4_finetune.log ) ELSE ( ECHO   Stage 2 done. )
)
ECHO.

REM ================================================================
REM PHASE 5 - Full Ablation Evaluation (ALL ~30 baselines)
REM Classical:    Flow-Only, Depth-Only, Fused-SNR, Fused-VMD,
REM               Fused-Keypoint, Fused-VMD-Keypoint, Legacy-XGB
REM DL Denoiser:  UNet, WaveNet, Transformer, Diffusion,
REM               CNN-BiLSTM, TFMAE, Mamba
REM Semi-Sup:     MSM, SAE, Contrastive, SPMat-BT,
REM               VLM-Gated, Quad-Modal Full (Ours)
REM Robustness:   Drop-Audio, Drop-Flow, Drop-Depth
REM Output: data/ablation_results/ablation_report_*.csv + plots
REM Runtime: ~4-8 hours on GPU
REM ================================================================
ECHO [PHASE 5] Full ablation evaluation - all baselines x all datasets...
ECHO   Runtime: 4-8 hours. Logging to %LOG%\phase5_ablation.log
python scripts/run_ablation_study.py > "%LOG%\phase5_ablation.log" 2>&1
IF ERRORLEVEL 1 ( ECHO   WARNING - check %LOG%\phase5_ablation.log ) ELSE ( ECHO   Done. )
ECHO.

REM ================================================================
REM PHASE 6 - ICBHI Audio-Only Evaluation
REM 920 WAV lung-sound recordings via UniversalAdapter.
REM GT derived from .txt annotation files (cycle timestamps).
REM Output: data/ablation_results/icbhi_audio_results.json
REM ================================================================
ECHO [PHASE 6] ICBHI audio-only evaluation...
IF EXIST "%RESULTS%\icbhi_audio_results.json" (
    ECHO   SKIP - icbhi_audio_results.json already exists.
) ELSE (
    python scripts/eval_icbhi.py > "%LOG%\phase6_icbhi.log" 2>&1
    IF ERRORLEVEL 1 ( ECHO   WARNING - check %LOG%\phase6_icbhi.log ) ELSE ( ECHO   Done. )
)
ECHO.

REM ================================================================
REM PHASE 7 - Edge Model Export (INT8 quantized + ONNX)
REM Output: data/models/edge/fusion_transformer_quantized.pth
REM ================================================================
ECHO [PHASE 7] Edge model export...
IF EXIST "%MODELS%\edge\fusion_transformer_quantized.pth" (
    ECHO   SKIP - quantized model already exists.
) ELSE (
    IF EXIST "%MODELS%\vlm_gated_fused.pth" (
        python scripts/export_edge.py --checkpoint data/models/vlm_gated_fused.pth > "%LOG%\phase7_export.log" 2>&1
        IF ERRORLEVEL 1 ( ECHO   WARNING - check %LOG%\phase7_export.log ) ELSE ( ECHO   Done. )
    ) ELSE (
        ECHO   SKIP - vlm_gated_fused.pth not found. Complete Phase 4 first.
    )
)
ECHO.

REM ================================================================
REM PHASE 8 - Advanced Foundation Models (RespMAE, RespDiff, CLISP)
REM Generates synthetic data if absent, then explicitly trains the
REM large unsupervised foundation models.
REM Outputs: data/models/foundation/*.pth
REM ================================================================
ECHO [PHASE 8] Advanced Foundation Models (Phase 5 of implementation)...
IF EXIST "%MODELS%\foundation\clisp_best.pth" (
    ECHO   SKIP - Foundation models already trained.
) ELSE (
    ECHO   Checking for foundation dataset...
    IF NOT EXIST "data\foundation_data\synthetic\clinical_notes_metadata.csv" (
        ECHO   1/2 Generating synthetic clinical waveforms and notes ^(few minutes^)...
        python scripts/setup_foundation_data.py > "%LOG%\phase8_data.log" 2>&1
        IF ERRORLEVEL 1 ( ECHO   WARNING - check %LOG%\phase8_data.log ) ELSE ( ECHO   Data generated. )
    )
    
    ECHO   2/2 Pre-training Foundational Models ^(RespMAE, RespDiff, CLISP^)...
    python scripts/train_foundation.py --model all > "%LOG%\phase8_train.log" 2>&1
    IF ERRORLEVEL 1 ( ECHO   WARNING - check %LOG%\phase8_train.log ) ELSE ( ECHO   Training Complete. )
)
ECHO.

REM ================================================================
ECHO ================================================================
ECHO  ALL PHASES COMPLETE - %DATE% %TIME%
ECHO ================================================================
ECHO.
ECHO  Key outputs:
ECHO    Signals:        %RESULTS%\signals\*.npz
ECHO    Denoisers:      %MODELS%\denoisers\*_best.pth
ECHO    Fusion model:   %MODELS%\vlm_gated_fused.pth
ECHO    Foundation:     %MODELS%\foundation\*.pth
ECHO    Ablation CSV:   %RESULTS%\ablation_report_*.csv
ECHO    Plots:          %RESULTS%\plots\*.png
ECHO    ICBHI results:  %RESULTS%\icbhi_audio_results.json
ECHO    Edge model:     %MODELS%\edge\
ECHO    Logs:           %LOG%\*.log
ECHO.
ECHO  Visualise: streamlit run src/ui/dashboard.py
ECHO.
PAUSE
GOTO :EOF

REM ================================================================
REM  SUBROUTINE :DENOISER  -- train one model if checkpoint missing
REM  Usage inside batch:  CALL :DENOISER modelname
REM ================================================================
:DENOISER
IF EXIST "data\models\denoisers\%~1_best.pth" (
    ECHO   SKIP %~1 - checkpoint exists.
) ELSE (
    ECHO.
    ECHO ============================================================
    ECHO   Training %~1  ... output shown live below
    ECHO ============================================================
    python scripts/train_denoiser.py --model %~1 --epochs 50 --patience 7
    IF ERRORLEVEL 1 ( ECHO   WARNING: %~1 training failed. ) ELSE ( ECHO   Done %~1 - saved to data\models\denoisers\%~1_best.pth )
)
EXIT /B 0
