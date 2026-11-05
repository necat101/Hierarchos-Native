[CmdletBinding()]
param([string]$OutDir = (Join-Path $PSScriptRoot '..\target\image-engine'))

$ErrorActionPreference = 'Stop'
$releaseTag = 'master-1052-f5df5f8'
$assetName = 'sd-master-f5df5f8-bin-win-vulkan-x64.zip'
$expectedSha256 = '2247d5e17fc9d426dfbe0bdd164ccd557a11049f4e61071caade3a3f069f6f6d'
$destination = [System.IO.Path]::GetFullPath($OutDir)
if (Test-Path -LiteralPath $destination) {
    if (-not (Test-Path -LiteralPath $destination -PathType Container)) {
        throw "Engine destination is not a directory: $destination"
    }
    if (@(Get-ChildItem -LiteralPath $destination -Force).Count -ne 0) {
        throw "Choose a new or empty engine directory: $destination"
    }
} else {
    New-Item -ItemType Directory -Path $destination | Out-Null
}
$archive = Join-Path $destination $assetName
$url = "https://github.com/leejet/stable-diffusion.cpp/releases/download/$releaseTag/$assetName"
Invoke-WebRequest -Uri $url -OutFile $archive
if ((Get-FileHash -Algorithm SHA256 -LiteralPath $archive).Hash -ne $expectedSha256) {
    throw 'Official image-engine archive SHA-256 mismatch; nothing was extracted.'
}
Expand-Archive -LiteralPath $archive -DestinationPath $destination
$engine = Join-Path $destination 'sd-cli.exe'
if (-not (Test-Path -LiteralPath $engine -PathType Leaf)) {
    throw "Pinned archive did not contain sd-cli.exe at $destination"
}
[ordered]@{
    schema = 1
    project = 'https://github.com/leejet/stable-diffusion.cpp'
    release = $releaseTag
    source_commit = 'f5df5f8'
    archive_sha256 = $expectedSha256
    executable_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $engine).Hash.ToLowerInvariant()
    backend = 'Vulkan'
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $destination 'hierarchos-image-engine.json') -Encoding UTF8
Write-Host "Image engine installed: $engine"
Write-Host 'Keep sd-cli.exe and all its DLLs together. Select it in the GUI or set HIERARCHOS_SD_CLI.'
