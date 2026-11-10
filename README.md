<p align="center">
    <img height="256px" src="https://raw.githubusercontent.com/xenia-canary/xenia/master/assets/icon/256.png" />
</p>

<h1 align="center">Xenia Kinect — Kinect for Xbox 360 on Xenia Canary</h1>

> [!IMPORTANT]
> This is an **unofficial fork** of [Xenia Canary](https://github.com/xenia-canary/xenia-canary).
> It is not made, reviewed or supported by the Xenia team. Report problems with
> it **here** ([Issues](https://github.com/ytelogooo/xenia-kinect/issues)), not
> on the Xenia Canary repository or Discord.

Xenia Kinect adds Kinect support to Xenia Canary: Kinect games see a Kinect
sensor, and you play them with your body — with a **real Kinect for Xbox 360**
plugged into your PC, or with **just a webcam**.

## AI disclosure

The changes in this fork were written with Claude (Anthropic's AI assistant)
under the fork owner's direction, and tested by the owner on their own PC,
Kinect and games. No Xenia developer has reviewed them. Commits written this
way carry a `Co-Authored-By: Claude` line. Expect rough edges, and please
report what you find.

## Download

**[⬇️ Latest test build (Windows)](https://github.com/ytelogooo/xenia-kinect/releases/tag/kinect-latest)**
— built automatically from the `kinect` branch every time it changes.

Unzip it anywhere. Inside you'll find `xenia_canary.exe`, three
**Play with ….bat** launchers (one per way of playing, see below) and the
bridge tools. Drag a game (its `.iso`, `.xex` or folder's `default.xex`) onto
the launcher you want.

## Three ways to play

| | What you need | How it tracks you | Best for |
|---|---|---|---|
| **Real Kinect** | Kinect for Xbox 360 + PC adapter, [UsbDk](https://github.com/daynix/UsbDk/releases) | The game's own Kinect tracking, exactly as on a console | Kinect Adventures; the most console-like |
| **Kinect SDK bridge** | Kinect for Xbox 360 + PC adapter, [Kinect for Windows SDK 1.8](https://www.microsoft.com/en-us/download/details.aspx?id=40278) (`KinectSDK-v1.8-Setup.exe`) | Microsoft's PC Kinect tracking, sent to the game | Just Dance, Dance Central; lighter on the PC |
| **Webcam bridge** | Any webcam (or a phone as a webcam), Python 3 | Google MediaPipe, from a normal camera | Trying it without a Kinect |

### Real Kinect — `Play with real Kinect.bat`

1. Plug in the Kinect with its **power adapter** (the "Kinect for Windows /
   Xbox 360 sensor adapter"). Model 1414 (the original Kinect) is the one
   tested.
2. Install **UsbDk** (`UsbDk_*_x64.msi` from its
   [releases page](https://github.com/daynix/UsbDk/releases)) and restart.
   UsbDk lets Xenia borrow the Kinect from Windows while the game runs.
3. Close anything else using the Kinect (the SDK bridge, Kinect Studio, …).
4. Drag the game onto `Play with real Kinect.bat`.

The game runs its own Kinect software on the real camera, motor and
microphone: tracking, the tilt motor, the floor, Kinect ID photos and your
avatar work as on a console. Xenia measures the floor itself the first time
(the game may tilt the Kinect down to look for it) and remembers it.

If the Kinect isn't found (`Kinect: cannot list USB devices` in
`xenia.log`), unplug its power and USB, wait a few seconds and plug it back in.

### Kinect SDK bridge — `Play with Kinect SDK bridge.bat`

1. Install the **Kinect for Windows SDK 1.8** and check the Kinect works in
   its *Skeleton Basics* sample.
2. Drag the game onto `Play with Kinect SDK bridge.bat`. The first time, it
   builds the bridge (`KinectSdkBridge.exe`) with the C# compiler that comes
   with Windows; then it starts the bridge and the game.

The bridge window shows your skeleton and outline; **Up/Down** tilt the
Kinect, **Q** quits. Details and options:
[tools/kinect_sdk_bridge](tools/kinect_sdk_bridge/README.md).

### Webcam bridge — `Play with webcam bridge.bat`

1. Install **Python 3** ([python.org](https://www.python.org/downloads/), tick
   *Add python.exe to PATH*).
2. Drag the game onto `Play with webcam bridge.bat`. The first time, it
   installs MediaPipe and OpenCV and downloads the pose model.

Stand 2–3 m from the camera, whole body in view, in a well-lit room. Options:
[tools/kinect_bridge](tools/kinect_bridge/README.md).

## Games

Tested by the fork owner. "Bridge" means either bridge.

| Game | Real Kinect | Bridge | Notes |
|---|---|---|---|
| Kinect Adventures! | ✅ Playable | — | 30 FPS with the GPU changes below; Kinect ID, avatar, tilt motor and floor work |
| Just Dance 3 | ✅ Playable | ✅ | Player square, hand cursor, scoring |
| Just Dance 4 | ✅ Playable | ✅ | DLC: only up to 14 packs show (known issue) |
| Just Dance 2014 | ✅ Playable | ✅ Playable | Player faces in the player slots |
| Just Dance 2015 | ✅ Playable | ✅ | Faces for up to two players; DLC |
| Just Dance 2016 | ✅ Playable | ❓ | Face in the player square matches a console |
| Just Dance 2017 | ❓ | ✅ | Two full skeletons, faces |
| Dance Central 3 | ❓ | ✅ | Player outline; songs decrypt without console keys |

❓ = not tested yet in that mode. Tried a game? Open an issue with the game,
the mode, what worked and your `xenia.log`.

## What's changed from Xenia Canary

### Kinect
- **Real Kinect over USB** (libusb + UsbDk, based on knuckleslee's camera
  support): depth, colour (8-bit Bayer and 16-bit YUV), calibration, the tilt
  motor, LED and accelerometer, and the floor — remembered like a console
  does, and measured from the depth image when none is known.
- **Bridges:** live skeletons over UDP from the Kinect SDK bridge (up to six
  players, two full skeletons, real outlines and camera picture) or the
  webcam bridge.
- **NUI system services:** Kinect ID (identity data, photos), the tilt and
  floor exports, the Kinect microphone array (from a PC recording device),
  Kinect Tuner dialog, avatars from the system avatar asset pack.
- Per-game support for player images, faces and hand tracking in Just Dance
  2014–2017, Just Dance 3/4 and Dance Central 3.

### Performance (Direct3D 12)
- Resolve rectangle lists done as one batch instead of one GPU stall per
  rectangle (the Kinect tracker resolves thousands of tiny ones a frame):
  Kinect Adventures went from about 14 to 30 FPS.
- Fewer GPU waits for read-backs, read-back buffers kept mapped, GPU work
  sent to the GPU while the frame is still being recorded
  (`d3d12_early_submit`; Just Dance 4 went from 20 to 30 FPS), and optional
  one-frame-late memexport results (`readback_memexport_delayed`).

### Other
- Handles closed through the game's own object types, memory range queries
  (backported), LZX decompression (`LDI*`), `KeInitializeMutant`,
  `XeKeysAesCbc` with built-in Harmonix song keys, DLC packages kept open
  while their files are in use, and diagnostics (`show_fps`, GPU read-back
  statistics, `trace_game_io`).

## Settings

The launchers set what each mode needs on the command line, so your
`xenia-canary.config.toml` stays as it is. To set them yourself instead:

| Setting | Real Kinect | SDK / webcam bridge |
|---|---|---|
| `[HID] allow_kinect_camera` | `true` | `false` |
| `[Kernel] kinect_connected` | `true` | `true` |
| `[Kernel] kinect_skeleton_udp_port` | — | `27050` |
| `[GPU] readback_memexport` | `true` | `false` |
| `[GPU] readback_resolve` | `"fast"` | (your choice) |
| `[Kernel] allow_avatar_initialization` | `true` | `true` |

Also useful: `[HID] kinect_usb_motor` (let games tilt the Kinect),
`[HID] kinect_floor_height` (force the floor, in mm below the sensor; 0 =
measure), `[HID] kinect_colour_gains` (apply games' colour gains),
`[HID] kinect_colour_gain_max` (limit them, 1–8),
`[GPU] readback_memexport_delayed` (faster, tracking a frame
later), `[Display] show_fps` (frame rate in the window title).

## Known issues

- Only the Kinect for Xbox 360 (v1) is supported; Kinect for Xbox One / v2
  isn't.
- Real Kinect and the SDK bridge can't run at the same time (one program owns
  the Kinect).
- Real Kinect mode: a player already standing still when the game starts may
  not be picked up (seen in Kinect Adventures and Just Dance 4) — step out of
  view and back in.
- Real Kinect mode: in Just Dance 4, 2014 and 2015 faces can look too bright
  or too saturated in a dim room; try `[HID] kinect_colour_gain_max = 2`
  (or `kinect_colour_gains = false`). More light helps most.
- Just Dance 4: installing more than 14 DLC packs makes all of them disappear.
- Real Kinect mode is Windows only.

## Building

Like Xenia Canary — see [building.md](docs/building.md):

```
git clone -b kinect https://github.com/ytelogooo/xenia-kinect.git
cd xenia-kinect
xb setup
xb build --config=release
```

## Credits

- The [Xenia](https://github.com/xenia-project/xenia) and
  [Xenia Canary](https://github.com/xenia-canary/xenia-canary) teams.
- knuckleslee's Kinect camera support (xenia-canary PR #1274, via
  JacoMur05/kinect_DanceCentral).
- [dcwestra/xenia-edge](https://github.com/dcwestra/xenia-edge) (avatars,
  Harmonix song keys).
- Backports by Tim Ryan (ryantimjohn) and Adrian.
- [libfreenect](https://github.com/OpenKinect/libfreenect) (the Kinect USB
  protocol and depth conversion), [libusb](https://libusb.info/),
  [UsbDk](https://github.com/daynix/UsbDk),
  [MediaPipe](https://github.com/google-ai-edge/mediapipe).

## License

BSD, the same as Xenia — see [LICENSE](LICENSE). This project is for
research and preservation; it doesn't include games or console files. Use
games you own.
