param(
    [string]$Version = "0.1.0",
    [string]$Python = "auto",
    [switch]$SkipInstaller
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$AppName = "VehicleScratchMotionDetector"
$VenvDir = Join-Path $Root ".venv-build"
$DistDir = Join-Path $Root "dist"
$BuildDir = Join-Path $Root "build"
$ReleaseDir = Join-Path $Root "release"
$AppDistDir = Join-Path $DistDir $AppName
$InstallerScript = Join-Path $Root "installer\VehicleScratchMotionDetector.iss"

function Remove-IfExists {
    param([string]$Path)
    if (Test-Path -LiteralPath $Path) {
        Remove-Item -LiteralPath $Path -Recurse -Force
    }
}

function Find-InnoCompiler {
    $command = Get-Command "ISCC.exe" -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }

    $candidates = @(
        "${env:LOCALAPPDATA}\Programs\Inno Setup 6\ISCC.exe",
        "${env:LOCALAPPDATA}\Programs\Inno Setup 7\ISCC.exe",
        "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
        "${env:ProgramFiles}\Inno Setup 6\ISCC.exe",
        "${env:ProgramFiles(x86)}\Inno Setup 7\ISCC.exe",
        "${env:ProgramFiles}\Inno Setup 7\ISCC.exe"
    )

    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path -LiteralPath $candidate)) {
            return $candidate
        }
    }

    return $null
}

function Test-PythonCommand {
    param([string[]]$Command)
    $exe = $Command[0]
    $args = @()
    if ($Command.Count -gt 1) {
        $args = $Command[1..($Command.Count - 1)]
    }

    try {
        & $exe @args --version *> $null
        return $LASTEXITCODE -eq 0
    } catch {
        return $false
    }
}

function Resolve-PythonCommand {
    if ($Python -ne "auto") {
        return @($Python)
    }

    if (Test-PythonCommand @("python")) {
        return @("python")
    }

    $pyLauncher = Get-Command "py" -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        foreach ($versionArg in @("-3.12", "-3.11", "-3")) {
            if (Test-PythonCommand @("py", $versionArg)) {
                return @("py", $versionArg)
            }
        }
    }

    throw "Could not find Python. Install Python 3.12, or pass -Python C:\Path\To\python.exe."
}

Write-Host "Building Vehicle Scratch Motion Detector $Version"
Write-Host "Project root: $Root"

Remove-IfExists $DistDir
Remove-IfExists $BuildDir
Remove-IfExists $ReleaseDir
New-Item -ItemType Directory -Force -Path $ReleaseDir | Out-Null

if (-not (Test-Path -LiteralPath $VenvDir)) {
    Write-Host "Creating build virtual environment..."
    $PythonCommand = Resolve-PythonCommand
    $PythonExe = $PythonCommand[0]
    $PythonArgs = @()
    if ($PythonCommand.Count -gt 1) {
        $PythonArgs = $PythonCommand[1..($PythonCommand.Count - 1)]
    }
    Write-Host "Using Python command: $($PythonCommand -join ' ')"
    & $PythonExe @PythonArgs -m venv $VenvDir
}

$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
if (-not (Test-Path -LiteralPath $VenvPython)) {
    throw "Could not find venv Python: $VenvPython"
}

$BuildPythonVersion = & $VenvPython -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
Write-Host "Build Python version: $BuildPythonVersion"
if ($BuildPythonVersion -notin @("3.10", "3.11", "3.12", "3.13")) {
    Write-Warning "PyInstaller may not support this Python version yet. For release builds, Python 3.12 is recommended."
}

Write-Host "Installing build dependencies..."
& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -r (Join-Path $Root "requirements.txt") pyinstaller

Write-Host "Running PyInstaller..."
& $VenvPython -m PyInstaller `
    --clean `
    --noconfirm `
    --onedir `
    --console `
    --name $AppName `
    --collect-all cv2 `
    (Join-Path $Root "main.py")

if (-not (Test-Path -LiteralPath (Join-Path $AppDistDir "$AppName.exe"))) {
    throw "PyInstaller did not create $AppName.exe"
}

Copy-Item -LiteralPath (Join-Path $Root "README.md") -Destination $AppDistDir -Force
Copy-Item -LiteralPath (Join-Path $Root "PRIVACY.md") -Destination $AppDistDir -Force
Copy-Item -LiteralPath (Join-Path $Root "LICENSE") -Destination $AppDistDir -Force

$UsageFile = Join-Path $AppDistDir "RUN_FROM_COMMAND_PROMPT.txt"
@"
Vehicle Scratch Motion Detector

Open PowerShell or Command Prompt in this folder, then run:

  .\$AppName.exe --help
  .\$AppName.exe --input "C:\path\to\video.mp4" --output "C:\path\to\output"

This tool produces candidate review clips only. It does not identify people,
infer intent, assign responsibility, or conclude that a crime happened.
"@ | Set-Content -LiteralPath $UsageFile -Encoding UTF8

$PortableZip = Join-Path $ReleaseDir "$AppName-$Version-windows-x64-portable.zip"
Write-Host "Creating portable zip: $PortableZip"
Compress-Archive -Path (Join-Path $AppDistDir "*") -DestinationPath $PortableZip -Force

if (-not $SkipInstaller) {
    $InnoCompiler = Find-InnoCompiler
    if ($InnoCompiler) {
        Write-Host "Building installer with Inno Setup: $InnoCompiler"
        & $InnoCompiler `
            "/DMyAppVersion=$Version" `
            "/DSourceDir=$AppDistDir" `
            "/DOutputDir=$ReleaseDir" `
            $InstallerScript
    } else {
        Write-Warning "Inno Setup compiler ISCC.exe was not found. Skipping installer build."
        Write-Warning "Install it from https://jrsoftware.org/isdl.php or run: winget install --id JRSoftware.InnoSetup -e -s winget"
    }
}

Write-Host "Build virtual environment kept at .venv-build for faster future builds."

Write-Host "Build complete. Release files:"
Get-ChildItem -LiteralPath $ReleaseDir | ForEach-Object {
    Write-Host "  $($_.FullName)"
}
