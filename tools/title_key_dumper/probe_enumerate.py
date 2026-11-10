#!/usr/bin/env python3
"""Asks your own Xbox 360 how it enumerates DLC, call by call.

Just Dance 2015 creates its DLC enumerator with 8 items per call, gives
XEnumerate a buffer for exactly 8 entries and loops until the call fails.
Xenia crashes when 9 or more DLC packages are installed. This script makes the
same XamContentCreateEnumerator / XamEnumerate calls on the console and prints,
for every call, the result code, how many items came back and which ones, so
we can see what a real console hands the game when more than 8 are installed.

The enumeration is for the title that is running on the console, so start
Just Dance 2015 on the console first and wait at its menu (in the dashboard it
lists the dashboard's content, which is none).

Needs the same setup as dump_title_keys.py (XBDM + JRPC2 on the console).

    python probe_enumerate.py --ip 192.168.1.91
    python probe_enumerate.py --ip 192.168.1.91 --items 8,32 --out probe.txt
"""

import argparse
import sys

from dump_title_keys import Console, ORD_EX_FREE_POOL, allocate

ORD_NT_CLOSE = 0x0CF  # xboxkrnl.exe
ORD_XAM_ENUMERATE = 0x250  # xam.xex
ORD_XAM_CONTENT_CREATE_ENUMERATOR = 0x25C  # xam.xex

XUSER_INDEX_ANY = 0xFE
CONTENT_TYPE_MARKETPLACE = 2  # DLC
ENTRY_SIZE = 0x134  # sizeof(XCONTENT_DATA)
MAX_CALLS = 20

# Scratch memory layout (all inside one ExAllocatePool block).
OFF_BUFFER_SIZE = 0x00
OFF_HANDLE = 0x04
OFF_ITEMS = 0x08
OFF_DATA = 0x100


def be32(data):
    return int.from_bytes(data[:4], "big")


def decode_entry(raw):
    """Decodes one XCONTENT_DATA: device, type, display name, file name."""
    device = be32(raw[0x0:0x4])
    content_type = be32(raw[0x4:0x8])
    name = raw[0x8:0x108].decode("utf-16-be", "replace").split("\0")[0]
    file_name = raw[0x108:0x132].split(b"\0")[0].decode("ascii", "replace")
    return device, content_type, name, file_name


def enumerate_once(console, pool, items_per_call, content_type, log):
    """Runs one whole enumeration; returns the entries in the order seen."""
    seen = []
    buffer_size_ptr = pool + OFF_BUFFER_SIZE
    handle_ptr = pool + OFF_HANDLE
    items_ptr = pool + OFF_ITEMS
    data_ptr = pool + OFF_DATA

    console.set_memory(pool, bytes(0x10))
    status = console.call("xam.xex", ORD_XAM_CONTENT_CREATE_ENUMERATOR,
                          XUSER_INDEX_ANY, 0, content_type, 0,
                          items_per_call, buffer_size_ptr, handle_ptr)
    buffer_size = be32(console.get_memory(buffer_size_ptr, 4))
    handle = be32(console.get_memory(handle_ptr, 4))
    log(f"XamContentCreateEnumerator(items={items_per_call}) -> "
        f"{status:08X}, buffer size 0x{buffer_size:X}, handle 0x{handle:08X}")
    if status != 0 or not handle:
        log("  could not create the enumerator; is a title running?")
        return seen
    if buffer_size < items_per_call * ENTRY_SIZE:
        buffer_size = items_per_call * ENTRY_SIZE

    try:
        for call_number in range(1, MAX_CALLS + 1):
            console.set_memory(items_ptr, bytes(4))
            status = console.call("xam.xex", ORD_XAM_ENUMERATE, handle, 0,
                                  data_ptr, buffer_size, items_ptr, 0)
            count = be32(console.get_memory(items_ptr, 4))
            log(f"  XamEnumerate call {call_number} -> {status:08X}, "
                f"{count} item(s)")
            if status != 0:
                break
            for index in range(min(count, items_per_call)):
                raw = console.get_memory(data_ptr + index * ENTRY_SIZE,
                                         ENTRY_SIZE)
                device, ctype, name, file_name = decode_entry(raw)
                seen.append((name, file_name))
                log(f"    [{len(seen):2d}] {name}  ({file_name}, "
                    f"device {device}, type 0x{ctype:X})")
            if count == 0:
                break
    finally:
        console.call("xboxkrnl.exe", ORD_NT_CLOSE, handle)
    log(f"  total {len(seen)} item(s) over the whole enumeration")
    return seen


def run_probe(console, items_list, content_type, log):
    pool = allocate(console, OFF_DATA + max(items_list) * ENTRY_SIZE)
    results = {}
    try:
        for items_per_call in items_list:
            log("")
            results[items_per_call] = enumerate_once(
                console, pool, items_per_call, content_type, log)
    finally:
        console.call("xboxkrnl.exe", ORD_EX_FREE_POOL, pool)
    return results


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--ip", required=True, help="the console's IP address")
    parser.add_argument("--items", default="8,32",
                        help="items-per-call values to try, e.g. 8,32 "
                        "(Just Dance 2015 uses 8)")
    parser.add_argument("--content-type", type=lambda v: int(v, 0),
                        default=CONTENT_TYPE_MARKETPLACE,
                        help="XCONTENT_TYPE (2 = DLC)")
    parser.add_argument("--out", default="enumerate_probe.txt",
                        help="also write the output to this file")
    parser.add_argument("--system", action="store_true",
                        help="run calls on a system thread")
    parser.add_argument("--debug", action="store_true",
                        help="print everything sent and received")
    args = parser.parse_args()

    items_list = [int(v) for v in args.items.split(",") if v.strip()]
    if not items_list or any(v < 1 or v > 64 for v in items_list):
        parser.error("--items needs values between 1 and 64")

    lines = []

    def log(text):
        print(text)
        lines.append(text)

    console = Console(args.ip, args.debug, args.system)
    log(f"Connected to {args.ip}")
    try:
        run_probe(console, items_list, args.content_type, log)
    finally:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
