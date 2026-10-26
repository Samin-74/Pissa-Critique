param(
    [Parameter(Mandatory=$true)][string]$Model,
    [Parameter(Mandatory=$true)][string]$ModelTag,
    [Parameter(Mandatory=$true)][int]$Seed
)

$root = $PSScriptRoot
$py = Join-Path $root ".venv\Scripts\python.exe"
Set-Location $root

$runGroup = "${ModelTag}_seed${Seed}"
$logDir = Join-Path "outputs\logs" $runGroup
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$trust = @()

$common = @(
    "--model_name", $Model,
    "--dataset_name", "openai/gsm8k",
    "--output_dir", "outputs",
    "--uniform_rank", "16",
    "--lora_alpha", "32",
    "--r_min", "4",
    "--max_seq_length", "512",
    "--per_device_batch_size", "4",
    "--gradient_accumulation_steps", "2",
    "--learning_rate", "2e-4",
    "--num_train_epochs", "2",
    "--train_limit", "3000",
    "--eval_limit", "500",
    "--eval_batch_size", "8",
    "--max_new_tokens", "512",
    "--logging_steps", "10",
    "--save_steps", "100000",
    "--seed", "$Seed",
    "--skip_plots"
) + $trust

# Each direction needs its own allocation, so both are computed once per model
$allocDir = "outputs\analysis\$ModelTag"
$allocJson = Join-Path $allocDir "allocation.json"
$allocInvDir = "outputs\analysis\${ModelTag}_inverted"
$allocInvJson = Join-Path $allocInvDir "allocation.json"

if (-not (Test-Path $allocJson)) {
    Write-Output "=== ALLOCATION $ModelTag ==="
    $allocArgs = @("analyze.py", "--model_name", $Model, "--output_dir", $allocDir,
                   "--uniform_rank", "16", "--r_min", "4", "--skip_plots") + $trust
    $p = Start-Process -FilePath $py -ArgumentList $allocArgs -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput (Join-Path $logDir "allocation.log") `
        -RedirectStandardError (Join-Path $logDir "allocation.log.err")
    if ($p.ExitCode -ne 0) {
        Write-Output "=== ALLOCATION FAILED $ModelTag ==="
        Get-Content (Join-Path $logDir "allocation.log.err") -Tail 20
        exit 1
    }
    Write-Output "=== ALLOCATION DONE $ModelTag ==="
}

if (-not (Test-Path $allocInvJson)) {
    Write-Output "=== ALLOCATION INVERTED $ModelTag ==="
    $allocInvArgs = @("analyze.py", "--model_name", $Model, "--output_dir", $allocInvDir,
                      "--uniform_rank", "16", "--r_min", "4", "--skip_plots",
                      "--no-high_entropy_more_rank") + $trust
    $p = Start-Process -FilePath $py -ArgumentList $allocInvArgs -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput (Join-Path $logDir "allocation_inverted.log") `
        -RedirectStandardError (Join-Path $logDir "allocation_inverted.log.err")
    if ($p.ExitCode -ne 0) {
        Write-Output "=== ALLOCATION INVERTED FAILED $ModelTag ==="
        Get-Content (Join-Path $logDir "allocation_inverted.log.err") -Tail 20
        exit 1
    }
    Write-Output "=== ALLOCATION INVERTED DONE $ModelTag ==="
}

$modes = @("uniform_lora", "uniform_pissa", "entropy_pissa", "entropy_pissa_inverted")

foreach ($m in $modes) {
    $runName = "$runGroup/$m"
    $log = Join-Path $logDir "$m.log"
    $allocForMode = $allocJson
    if ($m -eq "entropy_pissa_inverted") { $allocForMode = $allocInvJson }

    # Resume support: skip runs that already produced an eval result
    $evalFile = Join-Path "outputs" "$runName\eval_$m.json"
    if (Test-Path $evalFile) {
        Write-Output "=== SKIP $runName (already complete) ==="
        continue
    }

    Write-Output "=== START $runName ==="
    $argList = @("train.py", "--mode", $m, "--run_name", $runName,
                 "--allocation_json", $allocForMode) + $common
    $attempt = 0
    $maxAttempts = 2
    $ok = $false
    while ($attempt -lt $maxAttempts -and -not $ok) {
        $attempt++
        $proc = Start-Process -FilePath $py -ArgumentList $argList -NoNewWindow -Wait -PassThru `
            -RedirectStandardOutput $log -RedirectStandardError "$log.err"
        if ($proc.ExitCode -eq 0) {
            $ok = $true
        } else {
            Write-Output "=== ATTEMPT $attempt FAILED $runName (exit $($proc.ExitCode)) ==="
            Get-Content "$log.err" -Tail 15 -ErrorAction SilentlyContinue
            if ($attempt -lt $maxAttempts) {
                Write-Output "=== RETRYING $runName ==="
                Start-Sleep -Seconds 20
            }
        }
    }
    if (-not $ok) {
        Write-Output "=== FAILED $runName after $maxAttempts attempts ==="
        exit 1
    }
    Write-Output "=== DONE $runName ==="
}
Write-Output "=== ALL DONE $runGroup ==="
