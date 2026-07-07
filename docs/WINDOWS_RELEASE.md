# Windows Release Build

This project can be distributed to Windows users without requiring them to install Python.

The recommended release flow is:

1. Build a portable executable folder with PyInstaller.
2. Zip the portable folder.
3. Build a Windows installer with Inno Setup.
4. Upload both files to GitHub Releases.

## Local Build

From the project root:

```powershell
.\build_windows.ps1 -Version 0.1.0
```

The script creates:

```text
release/
  VehicleScratchMotionDetector-0.1.0-windows-x64-portable.zip
  VehicleScratchMotionDetector-Setup-0.1.0-windows-x64.exe
```

If Inno Setup is not installed, the script still creates the portable zip and skips the installer.

Install Inno Setup:

```powershell
winget install --id JRSoftware.InnoSetup -e -s winget
```

Python 3.12 is recommended for release builds. If your local Python is too new for PyInstaller, use the GitHub Actions workflow instead.

## Manual GitHub Actions Build

Open the repository on GitHub, then run:

```text
Actions -> Build Windows Release -> Run workflow
```

This creates downloadable workflow artifacts.

## Publish A GitHub Release

Create and push a version tag:

```powershell
git tag v0.1.0
git push origin v0.1.0
```

The workflow builds the Windows artifacts and creates a GitHub Release for that tag.

## Files In The Installer

The installer includes:

- `VehicleScratchMotionDetector.exe`
- OpenCV and Python runtime dependencies collected by PyInstaller
- `README.md`
- `PRIVACY.md`
- `LICENSE`
- a command-prompt shortcut that opens the tool's `--help`

The installer does not include sample surveillance videos or generated reports.
