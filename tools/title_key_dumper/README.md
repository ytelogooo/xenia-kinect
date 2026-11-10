# Title key dumper (your own Xbox 360 → Xenia)

Harmonix titles (Dance Central, Rock Band) decrypt song keys with key slots
0xE0-0xE8, which the console's hypervisor derives from secrets Xenia doesn't
have. This script asks your own console to do those exact calls.

1. In Xenia, try to load the songs once (they hang on the loading screen).
   Xenia writes the calls it couldn't answer to `title_keys_needed.txt` next
   to `xenia_canary.exe`.
2. Start the console with BadUpdate + XeUnshackle, with XBDM and the JRPC2
   plugin loaded (Dashlaunch `launch.ini`), on the same network as the PC.
3. Check the connection (the console shows a notification):
   `python dump_title_keys.py --ip <console IP> --test`
4. Answer the requests:
   `python dump_title_keys.py --ip <console IP> --needed <...>\title_keys_needed.txt --out <...>\title_keys.txt`
5. Restart Xenia; it reads `title_keys.txt` (`[Kernel] title_key_file`).

`--debug` prints everything sent to and received from the console.

## DLC enumeration probe (Just Dance 2015 with more than 8 DLC)

`probe_enumerate.py` makes the same `XamContentCreateEnumerator` /
`XamEnumerate` calls as Just Dance 2015 (8 items per call) on your console and
prints every call's result code and items, so you can see what a real console
returns when more than 8 DLC packages are installed.

1. Install 9 or more DLC packages for Just Dance 2015 on the console.
2. Start Just Dance 2015 on the console and wait at its menu (the DLC list is
   for the running title).
3. `python probe_enumerate.py --ip <console IP> --out enumerate_probe.txt`

Also note whether the game itself lists every song and keeps running.
