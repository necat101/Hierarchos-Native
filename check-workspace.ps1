[CmdletBinding()]
param(
    # Keep the generated workspace under this directory instead of the default
    # reusable one under target/workspace-check.
    [string]$WorkDir,

    # Check tests, benches and examples too. The default checks every binary.
    [switch]$AllTargets,

    # Leave the generated workspace and its junctions in place for inspection.
    [switch]$KeepWorkDir
)

$ErrorActionPreference = "Stop"

$crates = @(
    "hierarchos-inference",
    "hierarchos-vulkan",
    "hierarchos-native-cli",
    "hierarchos-gui"
)

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
foreach ($crate in $crates) {
    $manifest = Join-Path (Join-Path $repoRoot $crate) "Cargo.toml"
    if (-not (Test-Path -LiteralPath $manifest -PathType Leaf)) {
        throw "Missing crate manifest: $manifest"
    }
}

$userWorkDir = [bool]$WorkDir
if (-not $WorkDir) {
    # A stable path is what makes repeated checks warm: Cargo fingerprints the
    # member paths, so a fresh temp directory would rebuild the workspace every
    # run even when nothing changed.
    $WorkDir = Join-Path $repoRoot "target\workspace-check"
}
$workRoot = [System.IO.Path]::GetFullPath($WorkDir)
if (Test-Path -LiteralPath $workRoot) {
    $existing = @(Get-ChildItem -LiteralPath $workRoot -Force)
    if ($userWorkDir -and $existing.Count -ne 0) {
        throw "Workspace check directory must be new or empty: $workRoot"
    }
    # Recover from an interrupted run: remove only the junctions and manifests
    # this script created, never anything they point at.
    foreach ($crate in $crates) {
        $link = Join-Path $workRoot $crate
        if (Test-Path -LiteralPath $link) {
            $attributes = (Get-Item -LiteralPath $link -Force).Attributes
            if ($attributes -band [System.IO.FileAttributes]::ReparsePoint) {
                [System.IO.Directory]::Delete($link, $false)
            } else {
                throw "Refusing to replace a real directory at $link"
            }
        }
    }
    foreach ($name in @("Cargo.toml", "Cargo.lock")) {
        $path = Join-Path $workRoot $name
        if (Test-Path -LiteralPath $path -PathType Leaf) {
            Remove-Item -LiteralPath $path -Force
        }
    }
} else {
    New-Item -ItemType Directory -Path $workRoot | Out-Null
}

$links = @()
$previousTargetDir = $env:CARGO_TARGET_DIR
try {
    # Directory junctions let Cargo see all four crates as members of one
    # workspace without copying sources or needing administrator rights.
    foreach ($crate in $crates) {
        $link = Join-Path $workRoot $crate
        New-Item -ItemType Junction -Path $link -Target (Join-Path $repoRoot $crate) | Out-Null
        $links += $link
    }

    $workspaceManifest = Join-Path $workRoot "Cargo.toml"
    $members = ($crates | ForEach-Object { "    `"$_`"" }) -join ",`n"
    @"
[workspace]
resolver = "2"
members = [
$members
]
"@ | Set-Content -LiteralPath $workspaceManifest -Encoding ASCII

    # A fresh virtual workspace has no lockfile, so cargo would resolve every
    # dependency to the newest release instead of the versions the shipped
    # crates pin.  Newer releases can require a toolchain the repository does
    # not use (for example sysinfo 0.39.6 needs rustc 1.95 while the pinned
    # 0.38.4 does not), which fails this check for reasons unrelated to the
    # workspace's source.  Seed the lock from the main crate so the check reads
    # the same dependency graph as the qualified binaries; cargo still adds
    # entries that are only needed by the other members.
    $seedLock = Join-Path $repoRoot "hierarchos-vulkan\Cargo.lock"
    $workspaceLock = Join-Path $workRoot "Cargo.lock"
    if ((Test-Path -LiteralPath $seedLock) -and -not (Test-Path -LiteralPath $workspaceLock)) {
        Copy-Item -LiteralPath $seedLock -Destination $workspaceLock
    }

    # Share the repository target directory so dependency builds stay warm.
    # The stable workspace path keeps the crate fingerprints warm too.
    if (-not $env:CARGO_TARGET_DIR) {
        $env:CARGO_TARGET_DIR = Join-Path $repoRoot "target"
    }

    Push-Location $workRoot
    try {
        $metadataText = & cargo metadata --no-deps --format-version 1 --manifest-path $workspaceManifest
        if ($LASTEXITCODE -ne 0) {
            throw "cargo metadata failed with exit code $LASTEXITCODE"
        }
        $metadata = ($metadataText -join "`n") | ConvertFrom-Json
        $packages = @($metadata.packages)
        $packageNames = @($packages | ForEach-Object { $_.name })
        $missing = @($crates | Where-Object { $packageNames -notcontains $_ })
        if ($missing.Count -ne 0) {
            throw "Workspace is missing crates: $($missing -join ', ')"
        }

        $binaries = @()
        foreach ($package in $packages) {
            foreach ($target in $package.targets) {
                if (@($target.kind) -contains "bin") {
                    $binaries += "$($package.name)/$($target.name)"
                }
            }
        }
        foreach ($crate in $crates) {
            if (-not ($binaries | Where-Object { $_ -like "$crate/*" })) {
                throw "$crate declares no binary target"
            }
        }

        $cargoArgs = @("check", "--workspace")
        if ($AllTargets) {
            $cargoArgs += "--all-targets"
        } else {
            $cargoArgs += "--bins"
        }
        & cargo @cargoArgs
        if ($LASTEXITCODE -ne 0) {
            throw "cargo $($cargoArgs -join ' ') failed with exit code $LASTEXITCODE"
        }

        Write-Host "workspace check passed: $($crates.Count) crates, $($binaries.Count) binaries"
        $binaries | ForEach-Object { Write-Host "  $_" }
    } finally {
        Pop-Location
    }
} finally {
    if ($previousTargetDir) {
        $env:CARGO_TARGET_DIR = $previousTargetDir
    } elseif (Test-Path Env:\CARGO_TARGET_DIR) {
        Remove-Item Env:\CARGO_TARGET_DIR
    }

    if (-not $KeepWorkDir) {
        foreach ($link in $links) {
            if (Test-Path -LiteralPath $link) {
                # Delete the junction itself, never the crate it points at.
                [System.IO.Directory]::Delete($link, $false)
            }
        }
        $workspaceManifest = Join-Path $workRoot "Cargo.toml"
        if (Test-Path -LiteralPath $workspaceManifest) {
            Remove-Item -LiteralPath $workspaceManifest -Force
        }
    } elseif ($WorkDir) {
        Write-Host "kept workspace: $workRoot"
    }
}
