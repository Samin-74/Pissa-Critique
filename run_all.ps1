param(
    [string]$Only = ""
)

$root = $PSScriptRoot
$script = Join-Path $root "run_suite.ps1"

$jobs = @(
    @{ Model = "Qwen/Qwen2.5-3B-Instruct"; Tag = "qwen"; Seed = 42 },
    @{ Model = "Qwen/Qwen2.5-3B-Instruct"; Tag = "qwen"; Seed = 123 },
    @{ Model = "microsoft/Phi-3.5-mini-instruct"; Tag = "phi"; Seed = 42 },
    @{ Model = "microsoft/Phi-3.5-mini-instruct"; Tag = "phi"; Seed = 123 }
)

$failed = @()

foreach ($j in $jobs) {
    if ($Only -and $Only -ne "$($j.Tag)_seed$($j.Seed)") { continue }
    Write-Output "########## GROUP $($j.Tag)_seed$($j.Seed) ##########"
    & powershell -NoProfile -ExecutionPolicy Bypass -File $script `
        -Model $j.Model -ModelTag $j.Tag -Seed $j.Seed
    if ($LASTEXITCODE -ne 0) {
        # Keep going so one bad group cannot strand the remaining runs
        Write-Output "########## GROUP FAILED $($j.Tag)_seed$($j.Seed) ##########"
        $failed += "$($j.Tag)_seed$($j.Seed)"
    }
}

if ($failed.Count -gt 0) {
    Write-Output "########## FAILED GROUPS: $($failed -join ', ') ##########"
} else {
    Write-Output "########## ALL GROUPS COMPLETE ##########"
}
