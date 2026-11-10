@echo off
rem Xenia Kinect: play with a webcam, tracked by Google MediaPipe
rem (tools\kinect_bridge). Needs Python 3 on the PATH.
rem Drag a game (.iso, .xex or default.xex) onto this file.
rem Set CAMERA below if the wrong camera opens (see: python kinect_bridge.py --list).
setlocal
set "HERE=%~dp0"
set "CAMERA=0"
if "%~1"=="" (
  echo Drag a game ^(.iso, .xex or default.xex^) onto this file to play it.
  pause
  exit /b 1
)
where python >nul 2>nul
if errorlevel 1 (
  echo Python 3 was not found. Install it from https://www.python.org/downloads/
  echo ^(tick "Add python.exe to PATH"^) and try again.
  pause
  exit /b 1
)
if not exist "%HERE%tools\kinect_bridge\.installed" (
  echo Installing the webcam bridge's Python packages ^(first run only^)...
  python -m pip install -r "%HERE%tools\kinect_bridge\requirements.txt"
  if errorlevel 1 (
    echo.
    echo Installing the packages failed.
    pause
    exit /b 1
  )
  echo.> "%HERE%tools\kinect_bridge\.installed"
)
start "Kinect webcam bridge" /D "%HERE%tools\kinect_bridge" python kinect_bridge.py --camera %CAMERA%
start "" "%HERE%xenia_canary.exe" ^
  --allow_kinect_camera=false ^
  --kinect_connected=true ^
  --kinect_skeleton_udp_port=27050 ^
  --allow_avatar_initialization=true ^
  --readback_memexport=false ^
  "%~1"
