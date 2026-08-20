; VoiceTTSr Studio - Inno Setup Script
; Generates a professional Windows installation wizard

#define MyAppName "VoiceTTSr Studio"
#define MyAppVersion "1.7.0"
#define MyAppPublisher "mosesrb"
#define MyAppURL "https://github.com/mosesrb/VoiceTTSr"
#define MyAppExeName "VoiceTTSr.exe"

[Setup]
AppId={{D9B38F7A-6274-4C3D-8824-85F08C2361B1}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={localappdata}\Programs\VoiceTTSr
DisableProgramGroupPage=yes
LicenseFile=..\LICENSE
OutputDir=..\dist
OutputBaseFilename=VoiceTTSr_Setup_v{#MyAppVersion}
SetupIconFile=..\icon.ico
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "..\VoiceTTSr.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\voice_cloner_gui.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\xtts_worker.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\qwen_worker.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\chatterbox_worker.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\rvc_worker.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\skyrim_utils.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\download_resources.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\*.bat"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\requirements.txt"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\icon.ico"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\icon.png"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
Source: "..\README.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\LICENSE"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\core\*"; DestDir: "{app}\core"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\dsp\*"; DestDir: "{app}\dsp"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\ui\*"; DestDir: "{app}\ui"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\docs\*"; DestDir: "{app}\docs"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\tools\*"; DestDir: "{app}\tools"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\icon.ico"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\icon.ico"; Tasks: desktopicon

[Run]
Filename: "{app}\install_all.bat"; Description: "Initialize AI engine environments now (PyTorch & CUDA setup)"; Flags: postinstall skipifsilent unchecked
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent
