#!/usr/bin/env python3
"""Samples a block of a running title's memory on your own Xbox 360.

Reads a pointer, adds an offset and hex-dumps that many bytes about 20 times
a second over XBDM, writing lines in the same format as Xenia's diagnostic
logs ("JD4 cursor: t=<ms> <hex>"), so the same game state can be compared
between the console and Xenia.

Default: Just Dance 4's menu cursor object, *(0x82DE9B44) + 0x5DC, 0x120
bytes. Start Just Dance 4 on the console (Kinect plugged in), go to the menu
with the hand cursor, then run:

    python watch_memory.py --ip 192.168.1.91 --seconds 25 --out console-cursor.txt

and move your right hand the same way as in the Xenia test.

--jd4-body also writes "JD4 body:" lines like Xenia's kinect_debug_jd4_cursor
log: the cursor owner's per-player body state (+0x10E0, 0x120 bytes), the
skeleton frame header (0x30 bytes) and the cursor player's skeleton slot.
"""

import argparse
import sys
import time

from dump_title_keys import Console


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--ip", required=True, help="the console's IP address")
    parser.add_argument("--pointer", type=lambda v: int(v, 0), default=0x82DE9B44,
                        help="address holding the base pointer (0 = none)")
    parser.add_argument("--offset", type=lambda v: int(v, 0), default=0x5DC)
    parser.add_argument("--length", type=lambda v: int(v, 0), default=0x120)
    parser.add_argument("--interval", type=float, default=0.05,
                        help="seconds between samples")
    parser.add_argument("--seconds", type=float, default=25)
    parser.add_argument("--label", default="JD4 cursor")
    parser.add_argument("--out", default="console-cursor.txt")
    parser.add_argument("--jd4-body", action="store_true",
                        help="also sample Just Dance 4's body state")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    console = Console(args.ip, args.debug)
    print("Connected to", args.ip)
    base = 0
    if args.pointer:
        base = int.from_bytes(console.get_memory(args.pointer, 4), "big")
        print(f"*0x{args.pointer:08X} = 0x{base:08X}")
        if not base:
            print("The pointer is 0: is the game running and in its menus?")
            return 1
    address = base + args.offset
    print(f"Sampling 0x{args.length:X} bytes at 0x{address:08X} for "
          f"{args.seconds:g} s...")
    lines = 0
    start = time.monotonic()
    with open(args.out, "w", encoding="ascii") as out:
        while time.monotonic() - start < args.seconds:
            t = int(time.monotonic() * 1000)
            data = console.get_memory(address, args.length)
            out.write(f"{args.label}: t={t} {data.hex().upper()}\n")
            lines += 1
            if args.jd4_body and base:
                slot = int.from_bytes(data[:4], "big")
                if slot <= 5:
                    body = console.get_memory(base + 0x10E0, 0x120)
                    frame = int.from_bytes(
                        console.get_memory(base + 0x6F4, 4), "big")
                    if frame:
                        header = console.get_memory(frame, 0x30)
                        skeleton = console.get_memory(
                            frame + 0x30 + slot * 0x1C0, 0x1C0)
                        out.write(f"JD4 body: t={t} slot={slot} "
                                  f"{body.hex().upper()} {header.hex().upper()} "
                                  f"{skeleton.hex().upper()}\n")
            wait = args.interval - (time.monotonic() * 1000 - t) / 1000
            if wait > 0:
                time.sleep(wait)
    print(f"Wrote {lines} samples to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
