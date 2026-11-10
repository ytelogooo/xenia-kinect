@echo off
rem Xenia Kinect: play with a real Kinect for Xbox 360 (the game's own tracking).
rem Drag a game (.iso, .xex or default.xex) onto this file.
rem Needs UsbDk (https://github.com/daynix/UsbDk/releases) and the Kinect's
rem power adapter. Close the Kinect SDK bridge and other Kinect programs first.
setlocal
if "%~1"=="" (
  echo Drag a game ^(.iso, .xex or default.xex^) onto this file to play it.
  pause
  exit /b 1
)
start "" "%~dp0xenia_canary.exe" ^
  --allow_kinect_camera=true ^
  --kinect_connected=true ^
  --kinect_system_tracking=false ^
  --allow_avatar_initialization=true ^
  --readback_memexport=true ^
  --readback_resolve=fast ^
  "%~1"
