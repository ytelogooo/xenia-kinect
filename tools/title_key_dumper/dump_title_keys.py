#!/usr/bin/env python3
"""Answers Xenia's title key requests with your own Xbox 360.

Some titles (Harmonix: Dance Central, Rock Band) decrypt with key slots
0xE0-0xE8, which the console's hypervisor derives from secrets Xenia doesn't
have. Xenia lists the calls it couldn't answer in title_keys_needed.txt; this
script makes the same calls (XeKeysSetKey + XeKeysAesCbc) on your console over
the network and writes the results to title_keys.txt for Xenia.

Needs a console running XBDM with the JRPC2 plugin (for example BadUpdate +
XeUnshackle with jrpc2 loaded by Dashlaunch), on the same network.

    python dump_title_keys.py --ip 192.168.1.50 --test
    python dump_title_keys.py --ip 192.168.1.50 ^
        --needed "D:\\...\\Release\\title_keys_needed.txt" ^
        --out "D:\\...\\Release\\title_keys.txt"
"""

import argparse
import re
import socket
import sys
import time

XBDM_PORT = 730
# xboxkrnl.exe export ordinals.
ORD_EX_ALLOCATE_POOL = 0x009
ORD_EX_FREE_POOL = 0x00F
ORD_XE_KEYS_SET_KEY = 0x242
ORD_XE_KEYS_AES_CBC = 0x24B
JRPC_TYPE_INT = 1
JRPC_TYPE_XNOTIFY = 12


class Console:
    def __init__(self, ip, debug=False, system_thread=False):
        self.debug = debug
        self.system_thread = system_thread
        self.sock = socket.create_connection((ip, XBDM_PORT), timeout=10)
        self.buffer = b""
        banner = self.read_line()
        if not banner.startswith("201"):
            raise RuntimeError(f"not an XBDM console (got {banner!r})")

    def read_line(self):
        while b"\r\n" not in self.buffer:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise RuntimeError("the console closed the connection")
            self.buffer += chunk
        line, self.buffer = self.buffer.split(b"\r\n", 1)
        text = line.decode("cp1252", "replace")
        if self.debug:
            print("  <", text)
        return text

    def command(self, text):
        if self.debug:
            print("  >", text)
        self.sock.sendall(text.encode("cp1252") + b"\r\n")
        return self.read_line()

    def set_memory(self, address, data):
        reply = self.command(f"setmem addr=0x{address:08X} data={data.hex()}")
        if not reply.startswith("2"):
            raise RuntimeError(f"setmem failed: {reply}")

    def get_memory(self, address, length):
        reply = self.command(f"getmem addr=0x{address:08X} length=0x{length:X}")
        if not reply.startswith("2"):
            raise RuntimeError(f"getmem failed: {reply}")
        hex_text = ""
        while True:
            line = self.read_line()
            if line == ".":
                break
            hex_text += line.strip()
        if "?" in hex_text:
            raise RuntimeError("getmem: memory not readable")
        return bytes.fromhex(hex_text)[:length]

    def call(self, module, ordinal, *args):
        """Calls a function exported by ordinal; integer arguments only."""
        params = "A\\0\\A\\" + str(len(args)) + "\\"
        for value in args:
            value &= 0xFFFFFFFF
            if value >= 0x80000000:
                value -= 0x100000000  # JRPC sends 32-bit values signed
            params += f"1\\{value}\\"
        cmd = (f"consolefeatures ver=2 type={JRPC_TYPE_INT}"
               + (" system" if self.system_thread else "")
               + f' module="{module}" ord={ordinal} as=0 params="{params}"')
        reply = self.command(cmd)
        # Calls answer with a buffer address to poll for the result.
        polls = 0
        while "buf_addr=" in reply:
            polls += 1
            if polls > 40:
                raise RuntimeError(f"JRPC call {module}#{ordinal} timed out")
            time.sleep(0.25)
            address = reply.split("buf_addr=")[1].split()[0]
            if address.lower().startswith("0x"):
                address = address[2:]
            reply = self.poll(address)
        match = re.fullmatch(r"20\d-\s+(?:0x)?([0-9A-Fa-f]+)\s*", reply)
        if not match:
            raise RuntimeError(f"JRPC call {module}#{ordinal} failed: {reply}")
        return int(match.group(1), 16)

    poll_format = None
    POLL_FORMATS = [
        "consolefeatures ver=2 buf_addr=0x{address}",
        "consolefeatures ver=2 type=1 buf_addr=0x{address}",
        "consolefeatures ver=2 buf_addr={address}",
        "consolefeatures ver=2 type=1 params=\"A\\0\\A\\0\\\" buf_addr=0x{address}",
    ]

    def poll(self, address):
        """Asks for a pending call's result; finds the command format once."""
        formats = ([self.poll_format] if self.poll_format else
                   self.POLL_FORMATS)
        reply = ""
        for fmt in formats:
            reply = self.command(fmt.format(address=address))
            if "error=" not in reply:
                self.poll_format = fmt
                return reply
        raise RuntimeError(f"cannot read the JRPC result: {reply}")

    def notify(self, message):
        hex_text = message.encode("cp1252").hex()
        # Same format as jrpcPy's xNotify.
        self.command(f'consolefeatures ver=2 type={JRPC_TYPE_XNOTIFY} '
                     f'params="A\\0\\A\\2\\2/{len(message)}\\{hex_text}"')


def allocate(console, size):
    pool = console.call("xboxkrnl.exe", ORD_EX_ALLOCATE_POOL, size)
    # Never pass a bad address back to the console (ExFreePool, AES output).
    # (Pool memory can be anywhere in the virtual address space, e.g.
    # 0x30017EE8; 0 and the first 64 KB are never valid.)
    if not 0x00010000 <= pool < 0xC0000000:
        raise RuntimeError(f"ExAllocatePool returned an unexpected address "
                           f"0x{pool:08X}; stopping")
    return pool


def parse_needed(path):
    requests = []
    with open(path, encoding="ascii", errors="replace") as f:
        for line in f:
            words = line.split()
            # NEED <slot> <set key> <decrypt|encrypt> <input>
            if len(words) != 5 or words[0] != "NEED":
                if words and words[0] == "NEED":
                    print("skipping (non-zero IV, not supported):", line.strip())
                continue
            requests.append((words[1].upper(), words[2].upper(),
                             words[3].lower(), words[4].upper()))
    return list(dict.fromkeys(requests))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--ip", required=True, help="the console's IP address")
    parser.add_argument("--needed", default="title_keys_needed.txt")
    parser.add_argument("--out", default="title_keys.txt")
    parser.add_argument("--test", action="store_true",
                        help="only check the connection (shows a notification)")
    parser.add_argument("--system", action="store_true",
                        help="run calls on a system thread")
    parser.add_argument("--debug", action="store_true",
                        help="print everything sent and received")
    args = parser.parse_args()

    console = Console(args.ip, args.debug, args.system)
    print("Connected to", args.ip)
    if args.test:
        console.notify("Xenia title key dumper connected")
        pool = allocate(console, 0x100)
        print(f"JRPC works (test allocation at 0x{pool:08X}).")
        console.call("xboxkrnl.exe", ORD_EX_FREE_POOL, pool)
        return 0

    requests = parse_needed(args.needed)
    done = set()
    try:
        with open(args.out, encoding="ascii", errors="replace") as f:
            for line in f:
                w = line.split()
                if len(w) >= 6 and w[0] == "CBC":
                    done.add((w[1].upper(), w[2].upper(), w[3].lower(),
                              w[4].upper()))
    except FileNotFoundError:
        pass
    skipped = sum(1 for r in requests if r in done)
    requests = [r for r in requests if r not in done]
    if skipped:
        print(f"{skipped} request(s) already answered in {args.out}")
    if not requests:
        print("Nothing to do: no unanswered NEED lines in", args.needed)
        return 0
    largest = max(len(bytes.fromhex(r[3])) for r in requests)
    size = 0x100 + 2 * ((largest + 0xFF) & ~0xFF)
    pool = allocate(console, size)
    key_buffer, in_buffer = pool, pool + 0x100
    out_buffer = in_buffer + ((largest + 0xFF) & ~0xFF)
    done = 0
    try:
        with open(args.out, "a", encoding="ascii") as out:
            for slot, set_key, mode, inp in requests:
                key = bytes.fromhex(set_key)
                data = bytes.fromhex(inp)
                console.set_memory(key_buffer, key)
                status = console.call("xboxkrnl.exe", ORD_XE_KEYS_SET_KEY,
                                      int(slot, 16), key_buffer, len(key))
                if status:
                    print(f"XeKeysSetKey failed: {status:08X}")
                    continue
                console.set_memory(in_buffer, data)
                status = console.call("xboxkrnl.exe", ORD_XE_KEYS_AES_CBC,
                                      int(slot, 16), in_buffer, len(data),
                                      out_buffer, 0, 1 if mode == "encrypt" else 0)
                if status:
                    print(f"XeKeysAesCbc failed: {status:08X}")
                    continue
                result = console.get_memory(out_buffer, len(data))
                out.write(f"CBC {slot} {set_key} {mode} {inp} {result.hex().upper()}\n")
                done += 1
                print(f"{slot} {inp[:16]}... -> {result.hex().upper()[:16]}...")
    finally:
        console.call("xboxkrnl.exe", ORD_EX_FREE_POOL, pool)
    print(f"Wrote {done} of {len(requests)} results to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
