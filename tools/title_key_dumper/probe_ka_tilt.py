#!/usr/bin/env python3
"""Shows how a real Xbox 360 answers Kinect Adventures' tilt and floor questions.

In Xenia (real Kinect mode, motor on) Kinect Adventures' NUI runtime never
finds the floor: it keeps re-tilting the sensor and never calls
XamNuiCameraRememberFloor. This script records the same things on your
console, read-only except for a scratch buffer from ExAllocatePool:

1. XamNuiCameraTiltGetStatus, XamNuiCameraElevationGetAngle and
   XamNuiGetDepthCalibration results (what XAM reports: gravity, flags,
   remembered floor, calibration).
2. Kinect Adventures' auto-tilt state block (0x83898E10, 0x200 bytes) sampled
   about 10 times a second while the game starts and looks for you, written
   whenever it changes, with the interesting fields decoded.

--watch-xam instead calls XamNuiCameraTiltGetStatus about twice a second
and writes it whenever the floor or flags change - run it at the dashboard
(no game) and move the Kinect by hand to see whether the console measures the
floor again by itself:

    python probe_ka_tilt.py --ip 192.168.1.91 --watch-xam --seconds 60 --out ka-console-floor.txt

Needs the same setup as dump_title_keys.py (XBDM + JRPC2 on the console).
Start Kinect Adventures on the console (Kinect plugged in), then right away:

    python probe_ka_tilt.py --ip 192.168.1.91 --seconds 90 --out ka-console-tilt.txt

and stand where you normally play until you are past the player square.

To compare "calibrate" with Xenia (where the tracker loses you as calibration
starts and the auto-tilt then hunts up and down), sample faster and also log
the title's own skeleton tracker as it changes, then pick calibrate and stand
still until the Kinect settles:

    python probe_ka_tilt.py --ip 192.168.1.91 --seconds 60 --interval 0.05 --skeleton --out ka-console-calibrate.txt
"""

import argparse
import struct
import sys
import time

from dump_title_keys import Console, ORD_EX_FREE_POOL, allocate

ORD_XAM_NUI_GET_DEPTH_CALIBRATION = 0x860  # xam.xex
ORD_XAM_NUI_CAMERA_TILT_GET_STATUS = 0x870
ORD_XAM_NUI_CAMERA_ELEVATION_GET_ANGLE = 0x881

# Kinect Adventures (title 4D5308ED, the disc version Xenia runs): the NUI
# runtime's auto-tilt state, and code bytes at 0x82D2CD48 to check it's the
# same build (addresses differ with a title update).
KA_STATE = 0x83898E10
KA_STATE_SIZE = 0x200
KA_CHECK_ADDRESS = 0x82D2CD48
KA_CHECK_BYTES = bytes.fromhex("3d40838a3d6058743bea8e10616b5373")
# Its skeleton tracker's frame (NUI_SKELETON_FRAME as the title gets it):
# +0x08 frame number, +0x10 floor clip plane, then 6 skeletons of 0x1C0
# bytes from +0x30: +0x00 tracking state (2 tracked, 1 position only),
# +0x04 tracking ID, +0x10 position (x, y, z, w) in metres.
KA_SKELETON_FRAME = 0x83AD1C90
KA_SKELETON_FRAME_SIZE = 0x30 + 6 * 0x1C0


def be32(data, offset):
    return struct.unpack_from(">I", data, offset)[0]


def bes32(data, offset):
    return struct.unpack_from(">i", data, offset)[0]


def bef(data, offset):
    return struct.unpack_from(">f", data, offset)[0]


def decode_tilt_status(data, base=0):
    """XamNuiCameraTiltGetStatus's 0x50 bytes (base = where they start)."""
    fast = [bes32(data, base + 0x08 + 4 * i) for i in range(6)]
    legs = [bes32(data, base + 0x20 + 4 * i) for i in range(3)]
    flags = be32(data, base + 0x2C)
    plane = [bef(data, base + 0x30 + 4 * i) for i in range(4)]
    tail_flags = be32(data, base + 0x40)
    tail_legs = [bes32(data, base + 0x44 + 4 * i) for i in range(3)]
    return (f"latest {fast[:3]} short {fast[3:]} "
            f"gravity {legs} flags 0x{flags:08X} floor plane "
            f"[{', '.join(f'{v:.4g}' for v in plane)}] floor flags "
            f"0x{tail_flags:08X} floor gravity {tail_legs}")


def decode_state(data):
    """Kinect Adventures' auto-tilt block, fields its code is seen to use."""
    fields = [
        f"state {bes32(data, 0x50)}",
        f"tries {bes32(data, 0x58)}",
        f"wait {bes32(data, 0x5C)}",
        f"target {bes32(data, 0x64)}",
        f"+0x3C {be32(data, 0x3C):08X}",
        f"+0x1A0 {be32(data, 0x1A0):08X}",
        f"+0x1AC {be32(data, 0x1AC):08X}",
        f"+0x1B0 {be32(data, 0x1B0):08X}",
        f"+0x1BC {bes32(data, 0x1BC)}",
        f"+0x1D0 {be32(data, 0x1D0):08X}",
    ]
    return "; ".join(fields) + "; status: " + decode_tilt_status(data, 0x150)


def decode_skeletons(data):
    """The tracker's frame: who is tracked or seen, and where."""
    people = []
    for i in range(6):
        base = 0x30 + i * 0x1C0
        state = be32(data, base)
        if state not in (1, 2):
            continue
        x, y, z = (bef(data, base + 0x10 + 4 * k) for k in range(3))
        people.append(f"slot {i} {'tracked' if state == 2 else 'seen'} "
                      f"id {be32(data, base + 4)} at ({x:.2f}, {y:.2f}, "
                      f"{z:.2f}) m")
    plane = [bef(data, 0x10 + 4 * k) for k in range(4)]
    return (f"frame {be32(data, 8)}: "
            f"{'; '.join(people) if people else 'nobody'}; floor clip plane "
            f"[{', '.join(f'{v:.3f}' for v in plane)}]")


def skeleton_key(data):
    """What counts as a change: who is in which state, positions to 10 cm."""
    key = []
    for i in range(6):
        base = 0x30 + i * 0x1C0
        state = be32(data, base)
        key.append(state)
        if state in (1, 2):
            key.extend(round(bef(data, base + 0x10 + 4 * k), 1)
                       for k in range(3))
    return tuple(key)


def xam_calls(console, out):
    pool = allocate(console, 0x100)
    try:
        status_buffer = pool
        console.set_memory(status_buffer,
                           struct.pack(">II", 0x50, 0x58745373) + bytes(0x48))
        result = console.call("xam.xex", ORD_XAM_NUI_CAMERA_TILT_GET_STATUS,
                              status_buffer)
        data = console.get_memory(status_buffer, 0x50)
        out(f"XamNuiCameraTiltGetStatus -> {result:08X}")
        out(f"  raw {data.hex().upper()}")
        out(f"  {decode_tilt_status(data)}")

        angle_buffer = pool + 0x60
        console.set_memory(angle_buffer, bytes(8))
        result = console.call("xam.xex",
                              ORD_XAM_NUI_CAMERA_ELEVATION_GET_ANGLE,
                              angle_buffer, angle_buffer + 4)
        data = console.get_memory(angle_buffer, 8)
        out(f"XamNuiCameraElevationGetAngle -> {result:08X}: angle "
            f"{bes32(data, 0)}, second output 0x{be32(data, 4):08X}")

        calibration_buffer = pool + 0x80
        console.set_memory(calibration_buffer, bytes(0x30))
        result = console.call("xam.xex", ORD_XAM_NUI_GET_DEPTH_CALIBRATION,
                              calibration_buffer)
        data = console.get_memory(calibration_buffer, 0x30)
        out(f"XamNuiGetDepthCalibration -> {result:08X}")
        out(f"  raw {data.hex().upper()}")
        out("  as floats " + " ".join(f"{bef(data, i):.6g}"
                                      for i in range(0, 0x30, 4)))
    finally:
        console.call("xboxkrnl.exe", ORD_EX_FREE_POOL, pool)


def watch_xam(console, seconds, out):
    """Calls XamNuiCameraTiltGetStatus repeatedly, writes floor/flag changes."""
    pool = allocate(console, 0x100)
    try:
        previous = None
        start = time.monotonic()
        while time.monotonic() - start < seconds:
            t = int((time.monotonic() - start) * 1000)
            console.set_memory(pool, struct.pack(">II", 0x50, 0x58745373) +
                               bytes(0x48))
            result = console.call("xam.xex",
                                  ORD_XAM_NUI_CAMERA_TILT_GET_STATUS, pool)
            data = console.get_memory(pool, 0x50)
            # Flags and the floor tail; gravity changes all the time.
            key = (result, data[0x2C:0x50])
            if key != previous:
                out(f"t={t} -> {result:08X} {decode_tilt_status(data)}")
                previous = key
            time.sleep(0.3)
    finally:
        console.call("xboxkrnl.exe", ORD_EX_FREE_POOL, pool)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--ip", required=True, help="the console's IP address")
    parser.add_argument("--seconds", type=float, default=90)
    parser.add_argument("--interval", type=float, default=0.1)
    parser.add_argument("--out", default="ka-console-tilt.txt")
    parser.add_argument("--no-calls", action="store_true",
                        help="only read memory, don't call XAM")
    parser.add_argument("--watch-xam", action="store_true",
                        help="only watch XamNuiCameraTiltGetStatus (works at "
                             "the dashboard)")
    parser.add_argument("--skeleton", action="store_true",
                        help="also log the title's skeleton tracker frame "
                             "whenever who is seen or where changes")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    console = Console(args.ip, args.debug)
    print("Connected to", args.ip)
    with open(args.out, "w", encoding="ascii", errors="replace") as f:
        def out(line):
            print(line)
            f.write(line + "\n")
            f.flush()

        if args.watch_xam:
            out("== XamNuiCameraTiltGetStatus, written when the flags or the "
                "floor change")
            watch_xam(console, args.seconds, out)
            print("Wrote", args.out)
            return 0

        check = console.get_memory(KA_CHECK_ADDRESS, len(KA_CHECK_BYTES))
        sample = check == KA_CHECK_BYTES
        if not sample:
            out(f"Kinect Adventures code check at 0x{KA_CHECK_ADDRESS:08X}: "
                f"{check.hex()} (expected {KA_CHECK_BYTES.hex()}) - not "
                f"running, or a different build (title update?); only the XAM "
                f"calls are done")

        if not args.no_calls:
            out("== XAM at start")
            xam_calls(console, out)

        if sample:
            out(f"== Kinect Adventures auto-tilt block at 0x{KA_STATE:08X}, "
                f"written on change")
            previous = None
            previous_skeletons = None
            start = time.monotonic()
            while time.monotonic() - start < args.seconds:
                t = int((time.monotonic() - start) * 1000)
                data = console.get_memory(KA_STATE, KA_STATE_SIZE)
                if data != previous:
                    out(f"t={t} {decode_state(data)}")
                    f.write(f"t={t} raw {data.hex().upper()}\n")
                    previous = data
                if args.skeleton:
                    frame = console.get_memory(KA_SKELETON_FRAME,
                                               KA_SKELETON_FRAME_SIZE)
                    key = skeleton_key(frame)
                    if key != previous_skeletons:
                        out(f"t={t} tracker {decode_skeletons(frame)}")
                        previous_skeletons = key
                time.sleep(args.interval)

        if not args.no_calls:
            out("== XAM at the end")
            xam_calls(console, out)
    print("Wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
