"""Bench connection diagnostic — run this BEFORE the notebook.

    python check_bench.py                  # check everything
    python check_bench.py laser camera     # check specific devices

Every check is independent: it probes the device at the lowest level
available (OS -> SDK -> device query), prints PASS/FAIL with the reason,
and a concrete fix hint. Nothing here changes device state except the
camera check, which grabs one throwaway frame.

Exit code 0 = all checked devices passed (scriptable in CI/bench logs).
"""
from __future__ import annotations

import sys
import traceback

RESULTS: list[tuple[str, bool, str]] = []


def report(name: str, ok: bool, detail: str, hint: str = ""):
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {name}: {detail}")
    if not ok and hint:
        print(f"       fix -> {hint}")
    RESULTS.append((name, ok, detail))


# ----------------------------------------------------------------- laser
def check_laser(port_cfg: str = "COM3"):
    print("\n--- LASER (Coherent OBIS, USB serial) ---")
    try:
        from serial.tools import list_ports
        import serial
    except ImportError:
        report("laser", False, "pyserial not installed", "pip install pyserial")
        return

    ports = list(list_ports.comports())
    if not ports:
        report("laser", False, "no COM ports visible to Windows",
               "check the laser USB cable; Device Manager > Ports should list it")
        return
    print("       COM ports found: " +
          ", ".join(f"{p.device} ({p.description})" for p in ports))

    # prefer the configured port, else try any port that answers like an OBIS
    candidates = [port_cfg] + [p.device for p in ports if p.device != port_cfg]
    for port in candidates:
        try:
            with serial.Serial(port, 9600, timeout=1.0) as ser:
                ser.write(b"SYST:INF:MOD?\r\n")
                reply = ser.readline().decode(errors="replace").strip()
                if reply and not reply.startswith("ERR"):
                    extra = ""
                    ser.write(b"SYST:INF:SNUM?\r\n")
                    sn = ser.readline().decode(errors="replace").strip()
                    if sn and not sn.startswith("ERR"):
                        extra = f", S/N {sn}"
                    report("laser", True, f"{reply}{extra} answering on {port}"
                           + ("" if port == port_cfg else
                              f"  (NOTE: config says {port_cfg} — update the yaml)"))
                    return
        except serial.SerialException as e:
            if "PermissionError" in str(e) or "Access is denied" in str(e):
                report("laser", False, f"{port} exists but is locked by another program",
                       "close Coherent Connection, then re-run")
                return
            continue
    report("laser", False, "COM ports exist but none answered an OBIS query",
           f"confirm which port is the laser in Device Manager; tried {candidates}")


# ------------------------------------------------------------------- dmd
def check_dmd():
    print("\n--- DMD (ViALUX, USB + ALP API) ---")
    try:
        from ALP4 import ALP4
    except ImportError:
        report("dmd", False, "ALP4lib not installed or ALP DLL not found",
               "pip install ALP4lib; DLL ships with the ViALUX/EasyProj install")
        return
    try:
        dev = ALP4(version="4.3")
        dev.Initialize()
        report("dmd", True, f"initialized, {dev.nSizeX}x{dev.nSizeY} mirrors")
        dev.Free()
    except Exception as e:
        msg = str(e)
        if "ALP_NOT_ONLINE" in msg or "DEVICE" in msg.upper():
            report("dmd", False, f"ALP loaded but no device answered ({msg})",
                   "check DMD USB cable and power; close EasyProj if open")
        else:
            report("dmd", False, f"ALP error: {msg}",
                   "verify the ALP DLL version matches your controller (try version='4.2'/'4.3')")


# ------------------------------------------------------------------- slm
def check_slm(expected_res=(1920, 1152), lut_file: str | None = None,
              wfc_file: str | None = None):
    print("\n--- SLM (Meadowlark Blink HDMI: display + SDK + files) ---")
    # 1) is it attached as a second display?
    try:
        import ctypes
        user32 = ctypes.windll.user32
        n = user32.GetSystemMetrics(80)  # SM_CMONITORS
        if n >= 2:
            report("slm display", True, f"{n} monitors attached — SLM likely one of them; "
                   f"confirm one is {expected_res[0]}x{expected_res[1]} in Display settings")
        else:
            report("slm display", False, "only 1 monitor visible to Windows",
                   "plug the SLM HDMI cable and set Display settings to 'Extend'")
    except AttributeError:
        report("slm display", False, "not on Windows — display check skipped", "")

    # 2) calibration files present?
    import pathlib
    for label, f in (("slm lut", lut_file), ("slm wfc", wfc_file)):
        if f:
            ok = pathlib.Path(f).exists()
            report(label, ok, f if ok else f"missing: {f}",
                   "fix the path in config/onn_nico.yaml")

    # 3) SDK reachable?
    try:
        from slmsuite.hardware.slms.meadowlark import Meadowlark  # noqa: F401
        report("slm sdk", True, "slmsuite + Blink SDK importable")
    except ImportError as e:
        report("slm sdk", False, f"cannot import Meadowlark wrapper ({e})",
               "pip install slmsuite; Blink SDK DLL comes with the Meadowlark install")


# ---------------------------------------------------------------- camera
def check_camera():
    print("\n--- CAMERA (FLIR Grasshopper3, USB3 + Spinnaker) ---")
    try:
        import PySpin
    except ImportError:
        report("camera", False, "PySpin not installed",
               "install the Spinnaker SDK Python wheel from FLIR/Teledyne")
        return
    system = PySpin.System.GetInstance()
    try:
        cams = system.GetCameras()
        if cams.GetSize() == 0:
            cams.Clear()
            report("camera", False, "Spinnaker sees no cameras",
                   "check USB3 cable; close SpinView; try another USB3 (blue) port")
            return
        cam = cams.GetByIndex(0)
        cam.Init()
        model = cam.TLDevice.DeviceModelName.GetValue()
        sn = cam.TLDevice.DeviceSerialNumber.GetValue()
        # USB speed check — a USB2 link will starve the sensor
        speed = ""
        try:
            s = cam.TLDevice.DeviceCurrentSpeed.ToString()
            speed = f", link speed: {s}"
            if "High" in s and "Super" not in s:
                speed += "  (WARNING: USB2 — move to a blue USB3 port)"
        except Exception:
            pass
        # prove data flows: grab one frame
        cam.BeginAcquisition()
        img = cam.GetNextImage(2000)
        shape = (img.GetHeight(), img.GetWidth())
        incomplete = img.IsIncomplete()
        img.Release()
        cam.EndAcquisition()
        cam.DeInit()
        del cam
        cams.Clear()
        if incomplete:
            report("camera", False, f"{model} found but frame incomplete",
                   "bandwidth issue: USB3 port, shorter cable, or lower Device Link Throughput")
        else:
            report("camera", True, f"{model} S/N {sn}, test frame {shape} OK{speed}")
    except PySpin.SpinnakerException as e:
        report("camera", False, f"Spinnaker error: {e}",
               "close SpinView (it holds an exclusive lock), then re-run")
    finally:
        system.ReleaseInstance()


# ------------------------------------------------------------------ main
def main():
    which = {a.lower() for a in sys.argv[1:]} or {"laser", "dmd", "slm", "camera"}

    # pull ports/paths from the profile if it's there
    port, lut, wfc = "COM3", None, None
    try:
        import yaml
        with open("config/onn_nico.yaml") as f:
            cfg = yaml.safe_load(f)
        port = cfg["laser"].get("port", port)
        lut = cfg["slm"].get("lut_file")
        wfc = cfg["slm"].get("wfc_file")
        print(f"using config/onn_nico.yaml (laser port {port})")
    except Exception:
        print("no config found — using defaults (COM3, no file checks)")

    checks = {"laser": lambda: check_laser(port),
              "dmd": check_dmd,
              "slm": lambda: check_slm(lut_file=lut, wfc_file=wfc),
              "camera": check_camera}
    for name in ("laser", "dmd", "slm", "camera"):
        if name in which:
            try:
                checks[name]()
            except Exception:
                print(f"[FAIL] {name}: unexpected crash")
                traceback.print_exc()
                RESULTS.append((name, False, "crash"))

    print("\n================ SUMMARY ================")
    for name, ok, detail in RESULTS:
        print(f"  {'PASS' if ok else 'FAIL':4}  {name:12} {detail[:60]}")
    failed = [r for r in RESULTS if not r[1]]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
