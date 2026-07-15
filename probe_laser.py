import serial
for baud in (9600, 19200, 38400, 57600, 115200):
    for eol in (b"\r\n", b"\r", b"\n"):
        try:
            with serial.Serial("COM5", baud, timeout=0.5) as s:
                s.reset_input_buffer()
                s.write(b"SYST:INF:MOD?" + eol)
                r = s.read(64)
                if r.strip():
                    print(f"ANSWER at baud={baud}, eol={eol!r}: {r}")
        except Exception as e:
            print(f"baud={baud}: {e}"); break
print("done")