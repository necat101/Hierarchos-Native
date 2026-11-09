[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$BinDirectory,

    [string]$ReportPath,

    [int]$ExpectedExecutableCount = 7,

    [string[]]$AllowExtraDll = @()
)

$ErrorActionPreference = "Stop"

function Get-UInt16 {
    param([byte[]]$Bytes, [int]$Offset)
    return [System.BitConverter]::ToUInt16($Bytes, $Offset)
}

function Get-UInt32 {
    param([byte[]]$Bytes, [int]$Offset)
    return [System.BitConverter]::ToUInt32($Bytes, $Offset)
}

function Read-AsciiZ {
    param([byte[]]$Bytes, [int]$Offset)
    $text = New-Object System.Text.StringBuilder
    while ($Offset -lt $Bytes.Length -and $Bytes[$Offset] -ne 0) {
        [void]$text.Append([char]$Bytes[$Offset])
        $Offset++
    }
    return $text.ToString()
}

function Resolve-Rva {
    param([uint32]$Rva, [object[]]$Sections)
    foreach ($section in $Sections) {
        $end = [uint64]$section.VirtualAddress + [uint64]$section.Span
        if ([uint64]$Rva -ge [uint64]$section.VirtualAddress -and [uint64]$Rva -lt $end) {
            $delta = [uint64]$Rva - [uint64]$section.VirtualAddress
            return [int]([uint64]$section.PointerToRawData + $delta)
        }
    }
    throw ("PE RVA 0x{0:X8} is not backed by a section" -f $Rva)
}

function Get-PeImports {
    param([byte[]]$Bytes)

    if ($Bytes.Length -lt 64) {
        throw "file is too small to be a PE image"
    }
    $peOffset = [int](Get-UInt32 $Bytes 0x3c)
    if ($peOffset -lt 0 -or $peOffset + 24 -gt $Bytes.Length) {
        throw "invalid PE header offset"
    }
    if ((Get-UInt32 $Bytes $peOffset) -ne 0x00004550) {
        throw "missing PE signature"
    }

    $coff = $peOffset + 4
    $sectionCount = [int](Get-UInt16 $Bytes ($coff + 2))
    $optionalSize = [int](Get-UInt16 $Bytes ($coff + 16))
    $optional = $coff + 20
    if ($optional + $optionalSize -gt $Bytes.Length) {
        throw "truncated optional header"
    }
    $magic = Get-UInt16 $Bytes $optional
    if ($magic -eq 0x20b) {
        $dataDirectory = $optional + 112
    } elseif ($magic -eq 0x10b) {
        $dataDirectory = $optional + 96
    } else {
        throw ("unsupported optional-header magic 0x{0:X4}" -f $magic)
    }

    $importRva = Get-UInt32 $Bytes ($dataDirectory + 8)
    $importSize = Get-UInt32 $Bytes ($dataDirectory + 12)
    if ($importRva -eq 0) {
        return @()
    }

    $sectionTable = $optional + $optionalSize
    $sections = @()
    for ($index = 0; $index -lt $sectionCount; $index++) {
        $sectionOffset = $sectionTable + (40 * $index)
        if ($sectionOffset + 40 -gt $Bytes.Length) {
            throw "truncated section table"
        }
        $virtualSize = Get-UInt32 $Bytes ($sectionOffset + 8)
        $virtualAddress = Get-UInt32 $Bytes ($sectionOffset + 12)
        $rawSize = Get-UInt32 $Bytes ($sectionOffset + 16)
        $rawPointer = Get-UInt32 $Bytes ($sectionOffset + 20)
        $span = [Math]::Max([uint64]$virtualSize, [uint64]$rawSize)
        if ($span -eq 0) {
            continue
        }
        $sections += [pscustomobject]@{
            Name = (Read-AsciiZ $Bytes $sectionOffset).Trim()
            VirtualAddress = $virtualAddress
            VirtualSize = $virtualSize
            SizeOfRawData = $rawSize
            PointerToRawData = $rawPointer
            Span = $span
        }
    }
    if ($sections.Count -eq 0) {
        throw "PE has no loaded sections"
    }

    $descriptor = Resolve-Rva $importRva $sections
    $imports = @()
    $seen = @{}
    $maximumDescriptors = [Math]::Max(1, [int]($importSize / 20) + 1)
    for ($index = 0; $index -lt $maximumDescriptors; $index++) {
        $offset = $descriptor + (20 * $index)
        if ($offset + 20 -gt $Bytes.Length) {
            throw "truncated import descriptor table"
        }
        $originalFirstThunk = Get-UInt32 $Bytes $offset
        $nameRva = Get-UInt32 $Bytes ($offset + 12)
        $firstThunk = Get-UInt32 $Bytes ($offset + 16)
        if ($originalFirstThunk -eq 0 -and $nameRva -eq 0 -and $firstThunk -eq 0) {
            break
        }
        if ($nameRva -eq 0) {
            continue
        }
        $nameOffset = Resolve-Rva $nameRva $sections
        $dll = (Read-AsciiZ $Bytes $nameOffset).Trim()
        if ($dll) {
            $key = $dll.ToLowerInvariant()
            if (-not $seen.ContainsKey($key)) {
                $seen[$key] = $true
                $imports += $dll
            }
        }
    }
    return $imports
}

$systemDlls = @(
    "advapi32.dll", "bcrypt.dll", "bcryptprimitives.dll", "cabinet.dll",
    "cfgmgr32.dll", "combase.dll", "comctl32.dll", "comdlg32.dll",
    "crypt32.dll", "cryptbase.dll", "d2d1.dll", "d3d9.dll", "d3d11.dll",
    "d3d12.dll", "d3dcompiler_47.dll", "dbghelp.dll", "dnsapi.dll",
    "dwmapi.dll", "dxgi.dll", "gdi32.dll", "gdiplus.dll", "imm32.dll",
    "iphlpapi.dll", "kernel32.dll", "kernelbase.dll", "mpr.dll",
    "msasn1.dll", "msimg32.dll", "mswsock.dll", "ncrypt.dll", "netapi32.dll",
    "normaliz.dll", "ntdll.dll", "ole32.dll", "oleacc.dll", "oleaut32.dll",
    "opengl32.dll", "pdh.dll", "powrprof.dll", "psapi.dll", "rpcrt4.dll",
    "sechost.dll", "secur32.dll", "setupapi.dll", "shcore.dll", "shell32.dll",
    "shlwapi.dll", "user32.dll", "userenv.dll", "uxtheme.dll", "version.dll",
    "win32u.dll", "windowscodecs.dll", "winhttp.dll", "wininet.dll",
    "winmm.dll", "wintrust.dll", "ws2_32.dll", "wtsapi32.dll",
    "ucrtbase.dll", "msvcrt.dll"
)
$externalDlls = @("vulkan-1.dll")
$forbiddenPatterns = @(
    "^vcruntime\d",
    "^msvcp\d",
    "^concrt\d",
    "^vcomp\d",
    "^mfc\d"
)
$allowedExtra = @($AllowExtraDll | ForEach-Object { $_.ToLowerInvariant() })

if (-not (Test-Path -LiteralPath $BinDirectory -PathType Container)) {
    throw "missing executable directory: $BinDirectory"
}
$executables = @(Get-ChildItem -LiteralPath $BinDirectory -File -Filter "*.exe" | Sort-Object Name)
$problems = New-Object System.Collections.Generic.List[string]
$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Hierarchos Native portable dependency closure")
$lines.Add("")
$lines.Add("Directory: bin")
$lines.Add("Executables: $($executables.Count)")
$lines.Add("")

if ($ExpectedExecutableCount -gt 0 -and $executables.Count -ne $ExpectedExecutableCount) {
    $problems.Add("expected $ExpectedExecutableCount executables, found $($executables.Count)")
}

foreach ($executable in $executables) {
    try {
        $imports = @(Get-PeImports ([System.IO.File]::ReadAllBytes($executable.FullName)))
    } catch {
        $problems.Add("$($executable.Name): could not parse PE imports: $($_.Exception.Message)")
        $lines.Add("$($executable.Name): PARSE_ERROR")
        $lines.Add("")
        continue
    }

    $lines.Add("$($executable.Name)")
    if ($imports.Count -eq 0) {
        $lines.Add("  (no static import table)")
    }
    foreach ($dll in $imports) {
        $lower = $dll.ToLowerInvariant()
        $kind = "SYSTEM"
        $isForbidden = $false
        foreach ($pattern in $forbiddenPatterns) {
            if ($lower -match $pattern) {
                $isForbidden = $true
                break
            }
        }
        if ($isForbidden) {
            $kind = "VC_RUNTIME"
            $problems.Add("$($executable.Name): forbidden Visual C++ runtime import $dll")
        } elseif ($externalDlls -contains $lower) {
            $kind = "EXTERNAL_GPU"
        } elseif (($systemDlls -contains $lower) -or ($lower -match "^(api|ext)-ms-win-") -or ($allowedExtra -contains $lower)) {
            $kind = "SYSTEM"
        } else {
            $kind = "UNEXPECTED"
            $problems.Add("$($executable.Name): unexpected non-system import $dll")
        }
        $lines.Add("  [$kind] $dll")
    }
    $lines.Add("")
}

if ($problems.Count -eq 0) {
    $lines.Add("RESULT: PASS")
} else {
    $lines.Add("RESULT: FAIL")
    foreach ($problem in $problems) {
        $lines.Add("  ERROR: $problem")
    }
}

$report = $lines -join [Environment]::NewLine
if ($ReportPath) {
    $parent = Split-Path -Parent $ReportPath
    if ($parent -and -not (Test-Path -LiteralPath $parent -PathType Container)) {
        New-Item -ItemType Directory -Path $parent | Out-Null
    }
    Set-Content -LiteralPath $ReportPath -Value $report -Encoding UTF8
}
Write-Output $report
if ($problems.Count -gt 0) {
    exit 1
}
