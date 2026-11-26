param(
    [string]$Model = "Qwen/Qwen2.5-3B-Instruct",
    [string]$ModelTag = "qwen",
    [string]$Seeds = "42,123",
    [string]$Benchmarks = "arc_challenge"
)

# Scores already-trained adapters on extra benchmarks. No retraining happens here:
# the rank allocation and adapter weights are both reused as they are on disk.

$root = $PSScriptRoot
$py = Join-Path $root ".venv\Scripts\python.exe"
Set-Location $root

# Seeds arrives as text so both "-Seeds 42,123" and "-Seeds 42 -Seeds 7" work
$seedList = @($Seeds -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" } | ForEach-Object { [int]$_ })

$modes = @("uniform_lora", "uniform_pissa", "entropy_pissa", "entropy_pissa_inverted")
$failed = @()

# The un-finetuned base model is the zero-shot floor that makes absolute scores interpretable.
# Each model gets its own directory so Qwen and Phi references never overwrite each other.
$baseDir = Join-Path "outputs" "base_reference_$ModelTag"
$baseEval = Join-Path $baseDir "eval_base_model_$($Benchmarks.Split(',')[0]).json"
if (-not (Test-Path $baseEval)) {
    $logDir = Join-Path "outputs\logs" "${ModelTag}_base"
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
    Write-Output "=== BASE REFERENCE $Model on $Benchmarks ==="
    $baseArgs = @(
        "train.py",
        "--mode", "uniform_lora",
        "--model_name", $Model,
        "--output_dir", "outputs",
        "--run_name", "base_reference",
        "--eval_benchmarks", $Benchmarks,
        "--eval_limit", "500",
        "--eval_batch_size", "8",
        "--max_new_tokens", "512",
        "--bbh_max_per_subtask", "100",
        "--seed", "42",
        "--skip_train",
        "--skip_plots",
        "--base_model_reference"
    )
    $p = Start-Process -FilePath $py -ArgumentList $baseArgs -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput (Join-Path $logDir "base.log") `
        -RedirectStandardError (Join-Path $logDir "base.log.err")
    if ($p.ExitCode -ne 0) {
        Write-Output "=== BASE REFERENCE FAILED ==="
        Get-Content (Join-Path $logDir "base.log.err") -Tail 15 -ErrorAction SilentlyContinue
        $failed += "base_reference_$ModelTag"
    } else {
        Write-Output "=== BASE REFERENCE DONE ==="
    }
}

foreach ($seed in $seedList) {
    $runGroup = "${ModelTag}_seed${seed}"
    $logDir = Join-Path "outputs\logs" $runGroup
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null

    foreach ($m in $modes) {
        $runName = "$runGroup/$m"
        $adapterDir = Join-Path "outputs" "$runName\adapter"
        if (-not (Test-Path $adapterDir)) {
            Write-Output "=== SKIP $runName (no adapter) ==="
            $failed += "$runName (missing adapter)"
            continue
        }

        # Resume support: skip modes whose extra-benchmark eval already exists
        $firstBench = $Benchmarks.Split(',')[0].Trim()
        $existing = Join-Path "outputs" "$runName\eval_${m}_$firstBench.json"
        if (Test-Path $existing) {
            Write-Output "=== SKIP $runName on $Benchmarks (already complete) ==="
            continue
        }

        $log = Join-Path $logDir "eval_extra_$m.log"
        Write-Output "=== EVAL $runName on $Benchmarks ==="
        $argList = @(
            "train.py",
            "--mode", $m,
            "--model_name", $Model,
            "--run_name", $runName,
            "--output_dir", "outputs",
            "--eval_benchmarks", $Benchmarks,
            "--eval_limit", "500",
            "--eval_batch_size", "8",
            "--max_new_tokens", "512",
            "--bbh_max_per_subtask", "100",
            "--seed", "$seed",
            "--skip_train",
            "--skip_plots"
        )
        $proc = Start-Process -FilePath $py -ArgumentList $argList -NoNewWindow -Wait -PassThru `
            -RedirectStandardOutput $log -RedirectStandardError "$log.err"
        if ($proc.ExitCode -ne 0) {
            Write-Output "=== EVAL FAILED $runName (exit $($proc.ExitCode)) ==="
            Get-Content "$log.err" -Tail 20 -ErrorAction SilentlyContinue
            $failed += $runName
        } else {
            Write-Output "=== EVAL DONE $runName ==="
        }
    }
}

if ($failed.Count -gt 0) {
    Write-Output "########## FAILED: $($failed -join ', ') ##########"
    exit 1
}
Write-Output "########## EXTRA BENCHMARK EVALS COMPLETE ##########"
