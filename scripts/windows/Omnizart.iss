; Build with Inno Setup 7: ISCC.exe /DAppSource="C:\path\to\portable" Omnizart.iss
#ifndef AppSource
  #define AppSource "..\..\dist\Omnizart-Windows-PianoVoice"
#endif

[Setup]
AppId={{7A581ADD-C170-4D4C-BF6E-7D5B68D0D0D7}
AppName=Omnizart
AppVersion=2.17.0
AppPublisher=Omnizart contributors
DefaultDirName={localappdata}\Programs\Omnizart
DefaultGroupName=Omnizart
UsePreviousAppDir=yes
UsePreviousGroup=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
SetupArchitecture=x64
OutputDir=..\..\dist
OutputBaseFilename=Omnizart-Windows-Setup-2.17
Compression=zip/1
SolidCompression=no
WizardStyle=modern
UninstallDisplayName=Omnizart
SetupLogging=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"; Flags: unchecked

; Version 2.5 and older bundled the CPU-only distribution. DirectML uses
; the same Python module, so remove only its obsolete package metadata.
[InstallDelete]
Type: filesandordirs; Name: "{app}\runtime\Lib\site-packages\onnxruntime-1.20.1.dist-info"
; An earlier optional-model installer could upgrade scientific dependencies in
; place. Remove only these managed packages and their metadata before copying
; the bundled versions, so old extension modules cannot shadow the new ones.
Type: filesandordirs; Name: "{app}\runtime\Lib\site-packages\numpy"
Type: filesandordirs; Name: "{app}\runtime\Lib\site-packages\numpy.libs"
Type: filesandordirs; Name: "{app}\runtime\Lib\site-packages\numpy-*.dist-info"
Type: filesandordirs; Name: "{app}\runtime\Lib\site-packages\numba"
Type: filesandordirs; Name: "{app}\runtime\Lib\site-packages\numba-*.dist-info"
Type: filesandordirs; Name: "{app}\runtime\Lib\site-packages\llvmlite"
Type: filesandordirs; Name: "{app}\runtime\Lib\site-packages\llvmlite.libs"
Type: filesandordirs; Name: "{app}\runtime\Lib\site-packages\llvmlite-*.dist-info"
Type: filesandordirs; Name: "{app}\runtime\Lib\site-packages\scipy"
Type: filesandordirs; Name: "{app}\runtime\Lib\site-packages\scipy.libs"
Type: filesandordirs; Name: "{app}\runtime\Lib\site-packages\scipy-*.dist-info"

[Files]
Source: "{#AppSource}\*"; DestDir: "{app}"; Excludes: "*.pyc"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\Omnizart"; Filename: "{sys}\wscript.exe"; Parameters: """{app}\Omnizart.vbs"""; WorkingDir: "{app}"
Name: "{autodesktop}\Omnizart"; Filename: "{sys}\wscript.exe"; Parameters: """{app}\Omnizart.vbs"""; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{sys}\wscript.exe"; Parameters: """{app}\Omnizart.vbs"""; Description: "Launch Omnizart"; Flags: nowait postinstall skipifsilent
