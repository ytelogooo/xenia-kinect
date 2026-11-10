@echo off
rem Xenia Kinect: play with a Kinect for Xbox 360 tracked by the Kinect for
rem Windows SDK 1.8 on the PC (tools\kinect_sdk_bridge).
rem Drag a game (.iso, .xex or default.xex) onto this file.
setlocal
set "HERE=%~dp0"
set "BRIDGE=%HERE%build\kinect_sdk_bridge\KinectSdkBridge.exe"
if "%~1"=="" (
  echo Drag a game ^(.iso, .xex or default.xex^) onto this file to play it.
  pause
  exit /b 1
)
if not exist "%BRIDGE%" (
  echo Building the Kinect SDK bridge ^(first run only^)...
  powershell -NoProfile -ExecutionPolicy Bypass -File "%HERE%tools\kinect_sdk_bridge\build.ps1"
  if errorlevel 1 (
    echo.
    echo The bridge could not be built. Install the Kinect for Windows SDK 1.8
    echo and try again.
    pause
    exit /b 1
  )
)
start "" "%BRIDGE%"
start "" "%HERE%xenia_canary.exe" ^
  --allow_kinect_camera=false ^
  --kinect_connected=true ^
  --kinect_skeleton_udp_port=27050 ^
  --kinect_emulated_skeleton=false ^
  --kinect_player_index=true ^
  --allow_avatar_initialization=true ^
  --readback_memexport=false ^
  "%~1"
