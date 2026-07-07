#define MyAppName "Vehicle Scratch Motion Detector"
#define MyAppExeName "VehicleScratchMotionDetector.exe"

#ifndef MyAppVersion
#define MyAppVersion "0.1.0"
#endif

#ifndef SourceDir
#define SourceDir "..\dist\VehicleScratchMotionDetector"
#endif

#ifndef OutputDir
#define OutputDir "..\release"
#endif

[Setup]
AppId={{7E65D56A-77E5-4F0B-891B-8F532628A4A1}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher=Vehicle Scratch Motion Detector contributors
AppPublisherURL=https://github.com/oceanofpacific/Vehicle-Scratch-Motion-Detector
AppSupportURL=https://github.com/oceanofpacific/Vehicle-Scratch-Motion-Detector/issues
AppUpdatesURL=https://github.com/oceanofpacific/Vehicle-Scratch-Motion-Detector/releases
DefaultDirName={localappdata}\Programs\Vehicle Scratch Motion Detector
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
LicenseFile=..\LICENSE
OutputDir={#OutputDir}
OutputBaseFilename=VehicleScratchMotionDetector-Setup-{#MyAppVersion}-windows-x64
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64
UninstallDisplayIcon={app}\{#MyAppExeName}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName} Help"; Filename: "{sys}\cmd.exe"; Parameters: "/K cd /d ""{app}"" && ""{app}\{#MyAppExeName}"" --help"; WorkingDir: "{app}"
Name: "{autodesktop}\{#MyAppName} Help"; Filename: "{sys}\cmd.exe"; Parameters: "/K cd /d ""{app}"" && ""{app}\{#MyAppExeName}"" --help"; WorkingDir: "{app}"; Tasks: desktopicon
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"

[Run]
Filename: "{sys}\cmd.exe"; Parameters: "/K cd /d ""{app}"" && ""{app}\{#MyAppExeName}"" --help"; Description: "Show command line help"; Flags: postinstall skipifsilent unchecked
