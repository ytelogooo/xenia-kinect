#!/usr/bin/env python3
"""Finds the song key values in a Harmonix game's files, so that all songs
can be answered by your console in one go (see dump_title_keys.py).

Xenia logs the 16-byte values a title decrypts with title keys
(title_keys_needed.txt / title_keys.txt). This script looks for those values
in the game folder, works out where they sit in the song files (.mogg), reads
the value of every other song from the same place and adds them to
title_keys_needed.txt.

    python find_song_keys.py --game "D:\\Games\\Dance Central 3" ^
        --keys  "D:\\...\\Release\\title_keys.txt" ^
        --needed "D:\\...\\Release\\title_keys_needed.txt"

It only reads the game files. Without --write it only reports.
"""

import argparse
import collections
import os
import re
import struct
import sys

CHUNK = 16 * 1024 * 1024


def read_known(paths):
    """(slot, set key, mode, input) of calls Xenia logged."""
    known = []
    for path in paths:
        if not path or not os.path.exists(path):
            continue
        with open(path, encoding="ascii", errors="replace") as f:
            for line in f:
                w = line.split()
                if len(w) >= 5 and w[0] in ("NEED", "CBC"):
                    known.append((w[1].upper(), w[2].upper(), w[3].lower(),
                                  w[4].upper()))
    return list(dict.fromkeys(known))


def answered(keys_path):
    done = set()
    if keys_path and os.path.exists(keys_path):
        with open(keys_path, encoding="ascii", errors="replace") as f:
            for line in f:
                w = line.split()
                if len(w) >= 6 and w[0] == "CBC":
                    done.add((w[1].upper(), w[2].upper(), w[3].lower(),
                              w[4].upper()))
    return done


def all_files(root):
    for folder, _, names in os.walk(root):
        for name in names:
            yield os.path.join(folder, name)


def search_file(path, needles):
    """Yields (needle, offset) for every occurrence."""
    longest = max(len(n) for n in needles)
    with open(path, "rb") as f:
        base = 0
        tail = b""
        while True:
            chunk = f.read(CHUNK)
            if not chunk:
                break
            data = tail + chunk
            start = base - len(tail)
            for needle in needles:
                at = data.find(needle)
                while at != -1:
                    if at + len(needle) > len(tail):
                        yield needle, start + at
                    at = data.find(needle, at + 1)
            tail = data[-(longest - 1):]
            base += len(chunk)


MAX_ENTRIES = 100000
HEADER_SEARCH = 1024 * 1024


def find_header_before(path, offset):
    """The mogg header that the value at offset belongs to, or None."""
    begin = max(0, offset - HEADER_SEARCH)
    with open(path, "rb") as f:
        f.seek(begin)
        data = f.read(offset - begin)
    starts = [m.start() for m in
              re.finditer(rb"(?=[\x0a-\x20]\x00\x00\x00)", data)]
    for h in reversed(starts):
        if len(data) - h >= 20:
            version, header_size, map_version, seek_interval, entries = \
                struct.unpack_from("<5I", data, h)
            map_end = h + 20 + 8 * entries
            if (10 <= version <= 0x20 and entries <= MAX_ENTRIES and
                    map_version in (0x10, 0x0F, 0x11) and
                    0 < seek_interval <= 200000 and map_end <= len(data) and
                    len(data) - map_end <= 4096):
                return {"at": begin + h, "version": version,
                        "map_version": map_version,
                        "seek_interval": seek_interval, "entries": entries,
                        "map_end": begin + map_end}
    return None


def header_layout(header, offset):
    return (header["version"], header["map_version"], header["seek_interval"],
            offset - header["map_end"])


def find_pattern(path, pattern):
    """Offsets of a regex pattern (shorter than 64 bytes) in a file."""
    with open(path, "rb") as f:
        base = 0
        tail = b""
        while True:
            chunk = f.read(CHUNK)
            if not chunk:
                break
            data = tail + chunk
            start = base - len(tail)
            for m in pattern.finditer(data):
                if m.end() > len(tail):
                    yield start + m.start()
            tail = data[-63:]
            base += len(chunk)


def mogg_header(path):
    """(version, header end, OggMap end) of a .mogg, or None."""
    with open(path, "rb") as f:
        head = f.read(20)
        if len(head) < 20:
            return None
        version, header_size, _, _, entries = struct.unpack("<5I", head)
        if not 10 <= version <= 0x20 or entries > 100000:
            return None
        return version, header_size, 20 + 8 * entries


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--game", required=True, help="the game's folder")
    parser.add_argument("--keys", default="title_keys.txt")
    parser.add_argument("--needed", default="title_keys_needed.txt")
    parser.add_argument("--write", action="store_true",
                        help="add the songs' values to the needed file")
    args = parser.parse_args()

    known = read_known([args.keys, args.needed])
    if not known:
        print("No known values: load one song in Xenia first (it writes "
              "title_keys_needed.txt).")
        return 1
    print(f"{len(known)} known value(s) from Xenia's logs")

    files = list(all_files(args.game))
    by_ext = collections.Counter()
    size_by_ext = collections.Counter()
    for path in files:
        ext = os.path.splitext(path)[1].lower() or "(none)"
        by_ext[ext] += 1
        size_by_ext[ext] += os.path.getsize(path)
    print(f"\n{len(files)} files in {args.game}:")
    for ext, count in by_ext.most_common(15):
        print(f"  {ext:10s} {count:6d} files {size_by_ext[ext] / 2**20:10.1f} MB")

    needles = {bytes.fromhex(k[3]): k for k in known}
    print("\nSearching the game files for the known values "
          "(may take a few minutes)...")
    hits = []
    for path in files:
        try:
            for needle, offset in search_file(path, list(needles)):
                hits.append((needles[needle], path, offset))
        except OSError as e:
            print("  cannot read", path, e)
    if not hits:
        print("Not found: the game stores them in another form. Send this "
              "output to Claude.")
        return 1
    # Songs may be loose .mogg files or packed in archives (.ark): look for
    # the mogg header in front of each value.
    layouts = collections.Counter()
    hit_files = set()
    for k, path, offset in hits:
        rel = os.path.relpath(path, args.game)
        hit_files.add(path)
        header = find_header_before(path, offset)
        info = " (no mogg header found in front of it)"
        if header:
            layout = header_layout(header, offset)
            layouts[layout] += 1
            info = (f" (mogg header at {header['at']}: version {header['version']}, "
                    f"map version {header['map_version']}, seek interval "
                    f"{header['seek_interval']}, {header['entries']} map entries; "
                    f"value at OggMap end + {layout[3]})")
        with open(path, "rb") as f:
            f.seek(max(0, offset - 32))
            context = f.read(80)
        print(f"  {k[3][:16]}... in {rel} at offset {offset}{info}")
        print(f"     bytes around it: {context.hex()}")

    if not layouts:
        print("\nNo mogg headers found next to the values. Send this output "
              "to Claude.")
        return 1
    if len(layouts) > 1:
        print(f"\nThe layout differs between songs: {dict(layouts)}. Send this "
              "output to Claude.")
        return 1
    layout, count = layouts.most_common(1)[0]
    version, map_version, seek_interval, delta = layout
    print(f"\nSongs (mogg version {version}) keep the value at OggMap end + "
          f"{delta} (seen in {count} place(s)).")

    slot, set_key, mode = known[0][0], known[0][1], known[0][2]
    if any(k[1] != set_key for k in known):
        print("Note: more than one set key seen; using", set_key)
    done = answered(args.keys)
    pattern = re.compile(re.escape(struct.pack("<I", version)) + b".{4}" +
                         re.escape(struct.pack("<II", map_version,
                                               seek_interval)), re.S)
    scan = [p for p in files
            if p in hit_files or p.lower().endswith((".mogg", ".ark"))]
    print(f"Scanning {len(scan)} file(s) for all song headers...")
    new = []
    songs = 0
    for path in scan:
        for at in find_pattern(path, pattern):
            with open(path, "rb") as f:
                f.seek(at)
                head = f.read(20)
                if len(head) < 20:
                    continue
                entries = struct.unpack_from("<I", head, 16)[0]
                if entries > MAX_ENTRIES:
                    continue
                f.seek(at + 20 + 8 * entries + delta)
                value = f.read(16)
            if len(value) != 16:
                continue
            songs += 1
            request = (slot, set_key, mode, value.hex().upper())
            if request not in done and request not in new:
                new.append(request)
    print(f"{songs} song header(s) found; {len(new)} value(s) still need an "
          "answer from the console.")
    if args.write and new:
        with open(args.needed, "a", encoding="ascii") as f:
            for r in new:
                f.write("NEED " + " ".join(r) + "\n")
        print("Added them to", args.needed)
    elif new:
        print("Run again with --write to add them to", args.needed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
