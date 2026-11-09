[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$OutDir,

    [switch]$SkipBuild,

    [switch]$KeepBuildArtifacts
)

$ErrorActionPreference = "Stop"

function Invoke-CargoChecked {
    param([string[]]$CargoArgs)

    & cargo @CargoArgs
    if ($LASTEXITCODE -ne 0) {
        throw "cargo $($CargoArgs -join ' ') failed with exit code $LASTEXITCODE"
    }
}

function Copy-NativeCrate {
    param(
        [string]$Name,
        [string[]]$ExtraFiles = @(),
        [string[]]$ExtraDirectories = @()
    )

    $source = Join-Path $repoRoot $Name
    if (-not (Test-Path -LiteralPath $source -PathType Container)) {
        throw "Missing native crate: $source"
    }

    $destination = Join-Path $crateRoot $Name
    New-Item -ItemType Directory -Path $destination | Out-Null
    Copy-Item -LiteralPath (Join-Path $source "Cargo.toml") -Destination $destination
    if (Test-Path -LiteralPath (Join-Path $source "README.md") -PathType Leaf) {
        Copy-Item -LiteralPath (Join-Path $source "README.md") -Destination $destination
    }
    Copy-Item -LiteralPath (Join-Path $source "src") -Destination $destination -Recurse

    foreach ($file in $ExtraFiles) {
        $sourceFile = Join-Path $source $file
        if (-not (Test-Path -LiteralPath $sourceFile -PathType Leaf)) {
            throw "Missing standalone source file: $sourceFile"
        }
        Copy-Item -LiteralPath $sourceFile -Destination $destination
    }
    foreach ($directory in $ExtraDirectories) {
        $sourceDirectory = Join-Path $source $directory
        if (-not (Test-Path -LiteralPath $sourceDirectory -PathType Container)) {
            throw "Missing standalone source directory: $sourceDirectory"
        }
        Copy-Item -LiteralPath $sourceDirectory -Destination $destination -Recurse
    }
}

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptRoot
$templateRoot = Join-Path $scriptRoot "standalone"
$outputRoot = [System.IO.Path]::GetFullPath($ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutDir))

if (Test-Path -LiteralPath $outputRoot) {
    $existing = @(Get-ChildItem -LiteralPath $outputRoot -Force)
    if ($existing.Count -ne 0) {
        throw "Standalone output directory must be new or empty: $outputRoot"
    }
} else {
    New-Item -ItemType Directory -Path $outputRoot | Out-Null
}

$crateRoot = Join-Path $outputRoot "crates"
New-Item -ItemType Directory -Path $crateRoot | Out-Null

Copy-NativeCrate -Name "hierarchos-inference"
Copy-NativeCrate -Name "hierarchos-vulkan" `
    -ExtraFiles @("build.rs", "rdma_bindings.rs", "rdma_shim.c", "COMPATIBILITY.md", "README_ARCHITECTURES.md") `
    -ExtraDirectories @("shaders", "cases")
Copy-NativeCrate -Name "hierarchos-native-cli"
Copy-NativeCrate -Name "hierarchos-gui"

# The crate manifest retains its standalone shader harness target. Keep that
# source plus the small decision records compiled into the release binary, but
# do not ship the 361 MiB release-evidence tree or development fixture corpus.
$validationSource = Join-Path $scriptRoot "validation"
$validationDestination = Join-Path (Join-Path $crateRoot "hierarchos-vulkan") "validation"
if (-not (Test-Path -LiteralPath $validationSource -PathType Container)) {
    throw "Missing validation source directory: $validationSource"
}
New-Item -ItemType Directory -Path $validationDestination | Out-Null
$decisionRecords = @(Get-ChildItem -LiteralPath $validationSource -Filter "*.json" -File)
if ($decisionRecords.Count -eq 0) {
    throw "No runtime validation decision records found in $validationSource"
}
$decisionRecords | Copy-Item -Destination $validationDestination
Copy-Item -LiteralPath (Join-Path $validationSource "attention_kernel_cache_check.rs") -Destination $validationDestination

Copy-Item -LiteralPath (Join-Path $templateRoot "Cargo.toml") -Destination $outputRoot
Copy-Item -LiteralPath (Join-Path $templateRoot "README.md") -Destination $outputRoot
Copy-Item -LiteralPath (Join-Path $templateRoot "LICENSE") -Destination $outputRoot
Copy-Item -LiteralPath (Join-Path $templateRoot "RELEASE_NOTES.md") -Destination $outputRoot
Copy-Item -LiteralPath (Join-Path $templateRoot "CLEAN_MACHINE_TEST.md") -Destination $outputRoot
Copy-Item -LiteralPath (Join-Path $scriptRoot "README_ARCHITECTURES.md") -Destination $outputRoot
Copy-Item -LiteralPath (Join-Path $repoRoot "THIRD_PARTY_NOTICES.md") -Destination $outputRoot
$licensesSource = Join-Path $repoRoot "licenses"
if (-not (Test-Path -LiteralPath $licensesSource -PathType Container)) {
    throw "Missing bundled third-party licenses: $licensesSource"
}
Copy-Item -LiteralPath $licensesSource -Destination $outputRoot -Recurse

if (-not $SkipBuild) {
    Push-Location $outputRoot
    try {
        Invoke-CargoChecked -CargoArgs @("build", "--release", "-p", "hierarchos-native-cli", "--bin", "hierarchos-native-cli")
        Invoke-CargoChecked -CargoArgs @(
            "build", "--release", "-p", "hierarchos-vulkan",
            "--bin", "hierarchos-vulkan-train",
            "--bin", "hierarchos-vulkan-devices",
            "--bin", "hierarchos-vulkan-transformer-train",
            "--bin", "hierarchos-vulkan-transformer-logits"
        )
        Invoke-CargoChecked -CargoArgs @("build", "--release", "-p", "hierarchos-inference", "--bin", "hierarchos-infer")
        Invoke-CargoChecked -CargoArgs @("build", "--release", "-p", "hierarchos-gui", "--bin", "hierarchos-native")
    } finally {
        Pop-Location
    }

    $binDir = Join-Path $outputRoot "bin"
    New-Item -ItemType Directory -Path $binDir | Out-Null
    $suffix = if ($IsWindows -or $env:OS -eq "Windows_NT") { ".exe" } else { "" }
    foreach ($binary in @(
        "hierarchos-native-cli",
        "hierarchos-vulkan-train",
        "hierarchos-vulkan-devices",
        "hierarchos-vulkan-transformer-train",
        "hierarchos-vulkan-transformer-logits",
        "hierarchos-infer",
        "hierarchos-native"
    )) {
        $built = Join-Path $outputRoot "target\release\$binary$suffix"
        if (-not (Test-Path -LiteralPath $built -PathType Leaf)) {
            throw "Expected release binary was not produced: $built"
        }
        Copy-Item -LiteralPath $built -Destination $binDir
    }

    if (-not $KeepBuildArtifacts) {
        $targetDir = Join-Path $outputRoot "target"
        if (Test-Path -LiteralPath $targetDir -PathType Container) {
            $resolvedTarget = (Resolve-Path -LiteralPath $targetDir).Path
            $resolvedOutput = (Resolve-Path -LiteralPath $outputRoot).Path
            $expectedTarget = [System.IO.Path]::GetFullPath((Join-Path $resolvedOutput "target"))
            if ($resolvedTarget -ne $expectedTarget) {
                throw "Refusing to clean unexpected standalone target path: $resolvedTarget"
            }
            Remove-Item -LiteralPath $resolvedTarget -Recurse -Force
        }
    }
}

$sourceCommit = (& git -C $repoRoot rev-parse HEAD 2>$null)
if ($LASTEXITCODE -ne 0 -or -not $sourceCommit) {
    $sourceCommit = "unknown"
}
$sourceStatus = @(& git -C $repoRoot status --porcelain 2>$null)
$sourceState = if ($sourceStatus.Count -eq 0) { "clean" } else { "dirty" }
$buildTimestamp = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
$cargoVersion = (& cargo --version 2>&1 | Out-String).Trim()
$rustcVersion = (& rustc --version 2>&1 | Out-String).Trim()
$buildLines = @(
    "Hierarchos Native Windows x64 release candidate",
    "Source commit: $sourceCommit",
    "Source state at packaging: $sourceState",
    "Build timestamp (UTC): $buildTimestamp",
    "Target: x86_64-pc-windows-msvc",
    "Profile: release (opt-level=3, thin LTO, codegen-units=1)",
    "$cargoVersion",
    "$rustcVersion"
)
$binDir = Join-Path $outputRoot "bin"
if (Test-Path -LiteralPath $binDir -PathType Container) {
    $buildLines += "Binaries:"
    foreach ($binary in @(Get-ChildItem -LiteralPath $binDir -File | Sort-Object Name)) {
        $hash = (Get-FileHash -LiteralPath $binary.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        $buildLines += "  $($binary.Name)  sha256=$hash"
    }
}
[System.IO.File]::WriteAllLines(
    (Join-Path $outputRoot "BUILD_INFO.txt"),
    $buildLines,
    (New-Object System.Text.UTF8Encoding($false))
)

Write-Host "Standalone Hierarchos backend staged at $outputRoot"
if ($SkipBuild) {
    Write-Host "Source-only package created (-SkipBuild)."
} else {
    Write-Host "Release binaries are in $(Join-Path $outputRoot 'bin')"
    if (-not $KeepBuildArtifacts) {
        Write-Host "Standalone Cargo target artifacts were removed after packaging."
    }
}
Write-Host "Build provenance: $(Join-Path $outputRoot 'BUILD_INFO.txt')"
