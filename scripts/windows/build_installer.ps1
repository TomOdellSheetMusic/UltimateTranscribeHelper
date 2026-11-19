param(
    [string]$PortableFolder = "dist\Omnizart-Windows-PianoVoice",
    [string]$Compiler = "",
    [string]$OutputBaseName = "Omnizart-Windows-Setup-2.33"
)
$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..")).Path
if (-not [System.IO.Path]::IsPathRooted($PortableFolder)) {
    $PortableFolder = Join-Path $projectRoot $PortableFolder
}
$PortableFolder = (Resolve-Path -LiteralPath $PortableFolder).Path
if (-not (Test-Path -LiteralPath (Join-Path $PortableFolder "Omnizart.vbs"))) {
    throw "The portable app is incomplete: $PortableFolder"
}
if (-not $Compiler) {
    $candidates = @(
        (Join-Path $projectRoot "windows-wheel-check\inno\ISCC.exe"),
        (Join-Path ${env:ProgramFiles} "Inno Setup 7\ISCC.exe"),
        (Join-Path ${env:ProgramFiles(x86)} "Inno Setup 6\ISCC.exe")
    )
    $Compiler = $candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
}
if (-not $Compiler -or -not (Test-Path -LiteralPath $Compiler)) {
    throw "Inno Setup compiler not found. Install Inno Setup 7 or pass -Compiler with the path to ISCC.exe."
}
$script = Join-Path $PSScriptRoot "Omnizart.iss"
if ([System.IO.Path]::GetFileName($OutputBaseName) -ne $OutputBaseName -or -not $OutputBaseName) {
    throw "OutputBaseName must be a file name without a path."
}
$output = Join-Path $projectRoot "dist\$OutputBaseName.exe"
if (Test-Path -LiteralPath $output) {
    throw "Installer already exists: $output. Choose -OutputBaseName or move the existing file."
}
& $Compiler "/Qp" "/F$OutputBaseName" "/DAppSource=$PortableFolder" $script
if ($LASTEXITCODE -ne 0) { throw "Installer compilation failed with exit code $LASTEXITCODE." }
Write-Host "Installer: $output"
