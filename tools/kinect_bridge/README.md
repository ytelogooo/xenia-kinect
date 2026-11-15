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

## Camera picture

The bridge also sends the camera picture, as the Kinect SDK bridge does, for
the games' player pictures, face squares and recordings (Kinect Sports'
Show Off & Share, Just Dance 4's Autodance; that one also needs
`readback_memexport = true`). It is lined up with the skeleton the way a
Kinect's colour camera is, so `--hfov` should match your camera.

In the preview: W/S move the face square down/up, X/Z make it bigger/smaller,
F flips the picture left/right if the games show you on the wrong side.
`--no-colour` turns the picture off, `--colour-every 2` sends every second
one, `--no-face-crop` always sends the whole picture.
