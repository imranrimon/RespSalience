<#
Wait for the machine to be free, then run the RespSalience checkpoint training queue.

Host RAM is the binding constraint here, not GPU memory: this machine has 15 GB and has been hung
before by two training jobs running at once. So rather than starting the queue and hoping, this
watcher blocks until the machine can actually take it.

Release conditions, all of which must hold together:
  * no other heavy python training process is running (matched by script name, not by PID, so a
    restarted job still counts);
  * at least $MinFreeGB of host RAM is free;
  * both of the above hold on consecutive checks $StableChecks apart, so a momentary dip in another
    job's working set does not look like it exiting.

Launch detached so it survives the terminal that started it:
  Invoke-CimMethod Win32_Process Create -Arguments @{ CommandLine = 'powershell.exe -ExecutionPolicy Bypass -NoProfile -File "<this>"' }
#>

param(
    [double] $MinFreeGB   = 4.5,
    [int]    $PollSeconds = 120,
    [int]    $StableChecks = 3,
    [int]    $MaxWaitHours = 24
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$logDir = Join-Path $root "outputs\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log    = Join-Path $logDir "await_salience_$(Get-Date -Format 'yyyyMMdd-HHmmss').log"
$marker = Join-Path $logDir "salience_ckpt_STATUS.txt"

function Log($msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
    Add-Content -Path $log -Value $line -Encoding utf8
}

# Other people's training jobs we must not compete with. Matched on the command line.
$BusyPatterns = @("train_ldm_prior", "train_diffusion", "--dataset CelebA")

function Get-BlockingJobs {
    $procs = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue)
    $hits = @()
    foreach ($p in $procs) {
        if (-not $p.CommandLine) { continue }
        # Never count our own training queue as a blocker.
        if ($p.CommandLine -like "*train_resp_salience_ssl*") { continue }
        foreach ($pat in $BusyPatterns) {
            if ($p.CommandLine -like "*$pat*") { $hits += $p; break }
        }
    }
    return $hits
}

function Get-FreeGB {
    $os = Get-CimInstance Win32_OperatingSystem
    return [math]::Round($os.FreePhysicalMemory / 1MB, 2)
}

Set-Content -Path $marker -Value "WAITING $(Get-Date -Format s)" -Encoding utf8
Log "watcher started; need no blocking job and >= $MinFreeGB GB free, stable over $StableChecks checks"

$deadline = (Get-Date).AddHours($MaxWaitHours)
$clear = 0

while ($true) {
    if ((Get-Date) -gt $deadline) {
        Log "TIMEOUT after $MaxWaitHours h - never became free; not starting"
        Set-Content -Path $marker -Value "TIMEOUT $(Get-Date -Format s)" -Encoding utf8
        exit 2
    }

    $blocking = Get-BlockingJobs
    $free = Get-FreeGB

    if ($blocking.Count -eq 0 -and $free -ge $MinFreeGB) {
        $clear++
        Log "clear $clear/$StableChecks (free $free GB)"
        if ($clear -ge $StableChecks) { break }
    } else {
        if ($clear -gt 0) { Log "reset: blocking=$($blocking.Count) free=$free GB" }
        $clear = 0
    }
    Start-Sleep -Seconds $PollSeconds
}

Log "machine is free (free $(Get-FreeGB) GB) - starting the training queue"
Set-Content -Path $marker -Value "RUNNING $(Get-Date -Format s)" -Encoding utf8

& powershell.exe -ExecutionPolicy Bypass -NoProfile -File (Join-Path $root "scripts\train_salience_checkpoints.ps1") *>&1 |
    ForEach-Object { Add-Content -Path $log -Value $_ -Encoding utf8 }
$code = $LASTEXITCODE

$n = @(Get-ChildItem -Path (Join-Path $root "outputs\paper_runs") -Recurse -Filter "resp_salience_best.pt" -ErrorAction SilentlyContinue).Count
if ($code -eq 0) {
    Log "training queue finished cleanly; $n checkpoints on disk"
    Set-Content -Path $marker -Value "DONE $(Get-Date -Format s) checkpoints=$n" -Encoding utf8
} else {
    Log "training queue FAILED (exit $code); $n checkpoints on disk"
    Set-Content -Path $marker -Value "FAILED $(Get-Date -Format s) exit=$code checkpoints=$n" -Encoding utf8
}
exit $code
