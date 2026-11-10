# Builds KinectSdkBridge.exe with the C# compiler that ships with Windows
# (.NET Framework 4) against the Kinect for Windows SDK 1.8.
#
# Run from the xenia-canary source root:
#   powershell -ExecutionPolicy Bypass -File tools\kinect_sdk_bridge\build.ps1
# Output: build\kinect_sdk_bridge\KinectSdkBridge.exe

$ErrorActionPreference = "Stop"
$Source = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Out = Join-Path $Source "build\kinect_sdk_bridge"

$Csc = Join-Path $env:WINDIR "Microsoft.NET\Framework64\v4.0.30319\csc.exe"
if (-not (Test-Path $Csc)) {
    $Csc = Join-Path $env:WINDIR "Microsoft.NET\Framework\v4.0.30319\csc.exe"
}
if (-not (Test-Path $Csc)) { throw "The .NET Framework 4 C# compiler was not found." }

$Sdk = $env:KINECTSDK10_DIR
if (-not $Sdk) { $Sdk = "C:\Program Files\Microsoft SDKs\Kinect\v1.8\" }
$KinectDll = Join-Path $Sdk "Assemblies\Microsoft.Kinect.dll"
if (-not (Test-Path $KinectDll)) {
    throw "Microsoft.Kinect.dll not found in $Sdk - install the Kinect for Windows SDK 1.8."
}

New-Item -ItemType Directory -Force $Out | Out-Null
& $Csc /nologo /target:winexe /optimize+ /platform:anycpu `
    "/out:$(Join-Path $Out 'KinectSdkBridge.exe')" `
    "/reference:$KinectDll" /reference:System.Windows.Forms.dll /reference:System.Drawing.dll `
    (Join-Path $PSScriptRoot "KinectSdkBridge.cs")
if ($LASTEXITCODE -ne 0) { throw "Compiling the bridge failed." }
Copy-Item $KinectDll $Out -Force
Write-Host "Built $(Join-Path $Out 'KinectSdkBridge.exe')"
