param([string]$OutputDirectory = "target/gen9-v41-gate")
$ErrorActionPreference = "Stop"
$taskRoot = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
Set-Location -LiteralPath $taskRoot
$taskChanges = @()
if (Test-Path -LiteralPath "HANDOFF-MANIFEST.json") {
    $taskManifest = Get-Content -LiteralPath "HANDOFF-MANIFEST.json" -Raw | ConvertFrom-Json
    foreach ($taskEntry in $taskManifest.PSObject.Properties) {
        $taskPayload = Get-Item -LiteralPath (Join-Path $taskRoot $taskEntry.Name)
        $taskHash = (Get-FileHash -LiteralPath $taskPayload.FullName -Algorithm SHA256).Hash.ToLower()
        if ($taskPayload.Length -ne $taskEntry.Value.bytes -or $taskHash -ne $taskEntry.Value.sha256) {
            if ($taskEntry.Name.StartsWith("hierarchos-vulkan/validation/fixtures/")) {
                throw "Frozen CPU fixture/checkpoint changed: $($taskEntry.Name)"
            }
            $taskChanges += [ordered]@{path=$taskEntry.Name; sha256=$taskHash}
        }
    }
    Write-Host "PASS frozen fixture hashes; $($taskChanges.Count) changed handoff files recorded"
}
if (Test-Path -LiteralPath $OutputDirectory) { throw "Choose a fresh output directory: $OutputDirectory" }
$taskOutput = New-Item -ItemType Directory -Path $OutputDirectory
$taskResults = @()
function Run-Gate([string]$Label, [string[]]$Arguments) {
    $taskLog = Join-Path $taskOutput.FullName ($Label + ".log")
    # Windows PowerShell treats native stderr as an error record. Cargo emits
    # routine build progress there; the exit code is the success criterion.
    $taskPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & cargo @Arguments *> $taskLog
    $taskExit = $LASTEXITCODE
    $ErrorActionPreference = $taskPreference
    if ($taskExit -ne 0) { throw "$Label failed; see $taskLog" }
    $taskContent = Get-Content -LiteralPath $taskLog -Raw
    $taskMatch = [regex]::Match($taskContent, "test result: ok\. (\d+) passed; 0 failed;")
    if (-not $taskMatch.Success -or [int]$taskMatch.Groups[1].Value -lt 1) {
        throw "$Label did not execute a passing test: $taskLog"
    }
    $script:taskResults += [ordered]@{ name=$Label; passed=[int]$taskMatch.Groups[1].Value;
        log=($Label + ".log"); sha256=(Get-FileHash -LiteralPath $taskLog -Algorithm SHA256).Hash.ToLower() }
    Write-Host "PASS $Label"
}
$taskCommon = @("test", "--locked", "--manifest-path", "hierarchos-vulkan/Cargo.toml", "--lib", "-j", "1")
foreach ($taskCase in @(
    @("v41-default", "deepseek_v41::"),
    @("vulkan", "vulkan::tests"),
    @("generic-streaming", "parameter_streaming::"),
    @("offload", "weight_offload::tests")
)) { Run-Gate $taskCase[0] ($taskCommon + @($taskCase[1], "--", "--nocapture", "--test-threads=1")) }
foreach ($taskCase in @(
    @("package-default", "deepseek_v41_default_package_scalable_release_strict_all_logits"),
    @("joint-resident", "deepseek_v41_joint_depth_and_384_expert_strict_all_logits"),
    @("joint-all-streamed", "deepseek_v41_streamed_vocabulary_joint_strict_all_logits"),
    @("peft-trajectory", "deepseek_v41_production_uninterrupted_training_logit_trajectory")
)) { Run-Gate $taskCase[0] ($taskCommon + @($taskCase[1], "--", "--ignored", "--nocapture", "--test-threads=1")) }
$taskReceipt = [ordered]@{
    status="recorded_target_gates_passed"; utc=[DateTime]::UtcNow.ToString("o");
    absolute_logit_limit=2e-7; comparison="strict_less_than"; rtol=0;
    tests=$taskResults; passed_tests=($taskResults | Measure-Object -Property passed -Sum).Sum;
    changes_from_handoff=$taskChanges;
    manifest_sha256=if (Test-Path HANDOFF-MANIFEST.json) { (Get-FileHash HANDOFF-MANIFEST.json).Hash.ToLower() } else { $null };
    limits=@("Scaled checkpoints, no full production weights", "Inspect device names in logs; gates use the default Vulkan device", "Inference and native PEFT gates; not a production throughput claim")
}
$taskReceipt | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $taskOutput.FullName "qualification.json") -Encoding UTF8
Write-Host "Recorded gates: $($taskOutput.FullName)"
