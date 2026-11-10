# Kinect SDK bridge (real Kinect v1 → Xenia)

Tracks you with a real Kinect for Xbox 360 using the Kinect for Windows SDK
1.8 on the PC and sends the skeleton to Xenia over UDP, like the camera
bridge in `tools/kinect_bridge`. Tracking runs natively on the PC, so Xenia
needs no GPU readbacks and keeps its normal speed.

## Setup

1. Kinect for Windows SDK 1.8 installed, Kinect plugged in with its power
   adapter (the SDK's Skeleton Basics sample should work).
2. Build (from the source root):
   `powershell -ExecutionPolicy Bypass -File tools\kinect_sdk_bridge\build.ps1`
3. In `xenia-canary.config.toml` (edit the existing lines):
   ```
   [HID]
   allow_kinect_camera = false     # the bridge uses the Kinect, not Xenia
   [Kernel]
   kinect_connected = true
   kinect_skeleton_udp_port = 27050
   kinect_emulated_skeleton = false
   kinect_player_index = true
   [GPU]
   readback_memexport = false
   ```

## Run

Start `build\kinect_sdk_bridge\KinectSdkBridge.exe`, then Xenia. The window
shows the depth image with your skeleton (green = tracked, yellow =
guessed; the yellow R marks your right hand). Up/Down tilt the Kinect, O
toggles the outline clean-up, Q quits.

Options: `--smoothing 0.5` (0 = off, lowest latency; up to 0.95),
`--raw` (send the SDK's skeleton without levelling it with the floor),
`--invert-x` (swap left and right), `--seated`, `--port 27050`,
`--no-depth` (send only the skeleton), `--raw-outline` (see below),
`--players 6` and `--sdk-chooses` (see "More than one player").

Besides the skeleton, the bridge sends the depth image with your real
outline (XKDP packets), which Xenia uses for the games' player silhouettes
(`[Kernel] kinect_live_depth`; `kinect_live_depth_mirror` flips it).

The Kinect SDK often counts things touching you at about the same distance
(a bed, a chair, a door frame) as part of you. The bridge keeps only the
part of the outline within reach of your own skeleton, so the games don't
draw the furniture as your body. In the window, blue is what is sent as
you; purple is what the SDK marked as a player but is not sent (removed
furniture, or a "ghost" player the SDK sees on a door). O switches between
the cleaned outline and the SDK's own; `--raw-outline` starts with the
SDK's.

It also sends the camera image (XKCL packets, 640x480 at 15 frames per
second). Just Dance shows your face in the player slots before a song; Xenia
writes the picture into the game's colour stream (`[Kernel]
kinect_live_colour`; `kinect_live_colour_mirror` flips it). `--no-colour`
turns it off and `--colour-every 3` sends fewer frames.

The camera image is cropped to a square around your head (found with the SDK's
skeleton-to-colour mapping), because on a console the NUI runtime delivers the
face the games show in the player slot already cropped. `--face-size 0.30`
sets the width of the square in metres (smaller = closer on the face),
`--face-offset 0.08` shifts it that many metres below the SDK's head joint
(which sits near the top of the head), and `--no-face-crop` sends the whole
camera image instead. In the bridge window, Z/X zoom in/out and W/S move the
face up/down in the square while you watch the game; the window shows the
current values to use as options.

Just Dance 2015 and 2017 crop each player's face out of the camera image
themselves, so with two or more players start the bridge with
`--no-face-crop`: Xenia then corrects where the games look for each head
(`[Kernel] kinect_jd_colour_mapping = "auto"`), and W/S in the bridge window
still move the faces up/down in the squares (Z/X do nothing then; the games
choose the size).

## More than one player

The bridge sends everyone the Kinect sees, up to six people. Like on a
console, a Kinect v1 gives full skeletons (all 20 joints) to two people at a
time and only a position to the others. Each person keeps a number (P1-P6,
shown above their head) while they stay in view; it is their skeleton slot
in the game and their value in the outline, and each number has its own
outline colour in the window. The "main player" (green skeleton) is the one
the face crop follows.

Games that pick who gets the two full skeletons (Just Dance 2017 does, and
rotates them between the players) have their choice passed to the bridge,
which asks the Kinect SDK to track those two (when the game names only one,
the second full skeleton goes to someone else in view); the window shows the
game's pick. `--sdk-chooses` lets the SDK pick instead, and `--players 1` sends
only the main player in the old one-player format. The SDK sometimes sees a
"person" that isn't there (a coat, a door); `--players 2` limits what is
sent if a game counts such a ghost as a player.
