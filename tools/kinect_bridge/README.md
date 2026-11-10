# Kinect bridge (camera → Kinect skeleton)

Tracks your body with a webcam (or a phone used as a webcam) using Google
MediaPipe Pose and sends a Kinect v1 skeleton to Xenia over UDP.

## Setup

1. Install Python 3 (python.org or `winget install Python.Python.3.12`).
2. In this folder: `python -m pip install -r requirements.txt`
3. In `xenia-canary.config.toml`, under `[Kernel]`:
   ```
   kinect_connected = true
   kinect_skeleton_udp_port = 27050
   ```
   (`kinect_emulated_skeleton` can stay on; the live skeleton takes priority.)

## Run

```
python kinect_bridge.py --list        # find the camera index
python kinect_bridge.py --camera 1    # start; Q in the preview quits
```

Start the bridge before or after Xenia; Xenia uses the newest skeleton and
treats the player as gone if none arrives for 0.25 s.

Useful options: `--hfov` (camera field of view, fixes distance estimates),
`--camera-height` (metres above the floor), `--mirror`, `--model lite|full|heavy`,
`--smoothing` (0–1).

Stand so your whole body, head to feet, is in the picture, about 2–3 m from
the camera, in a well-lit room.
