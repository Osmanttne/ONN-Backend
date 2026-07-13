"""Coherent OBIS 660-75FP driver (SCPI over USB virtual COM port).

Command reference: OBIS LG/LX Integrator's manual. Power is commanded in
WATTS on the wire; this class exposes milliwatts.
"""
from __future__ import annotations

import logging

from .base import Device, DeviceState

log = logging.getLogger("onn.laser")

try:
    import serial  # pyserial
    _HAS_SERIAL = True
except ImportError:  # pragma: no cover
    _HAS_SERIAL = False


class LaserOBIS(Device):
    name = "laser"

    def __init__(self, port: str = "COM3", baudrate: int = 9600,
                 default_power_mw: float = 10.0, max_power_mw: float = 30.0,
                 timeout: float = 1.0):
        super().__init__()
        if not _HAS_SERIAL:
            raise ImportError("pyserial not installed: pip install pyserial")
        self.port, self.baudrate, self.timeout = port, baudrate, timeout
        self.default_power_mw = default_power_mw
        self.max_power_mw = max_power_mw
        self._ser = None

    # -- low level ---------------------------------------------------
    def _query(self, cmd: str) -> str:
        self._ser.write((cmd + "\r\n").encode())
        reply = self._ser.readline().decode().strip()
        # OBIS follows replies with an "OK" handshake line
        hs = self._ser.readline().decode().strip()
        if hs and hs != "OK" and not reply:
            reply, hs = hs, ""
        if reply.startswith("ERR"):
            raise RuntimeError(f"laser error for {cmd!r}: {reply}")
        return reply

    def _command(self, cmd: str):
        self._ser.write((cmd + "\r\n").encode())
        hs = self._ser.readline().decode().strip()
        if hs.startswith("ERR"):
            raise RuntimeError(f"laser error for {cmd!r}: {hs}")

    # -- lifecycle -----------------------------------------------------
    def connect(self):
        self._ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
        self._command("SYST:COMM:HAND ON")          # enable OK handshaking
        model = self._query("SYST:INF:MOD?")
        log.info("connected to %s on %s", model, self.port)
        self.set_power_mw(self.default_power_mw)     # enforce bench default
        self.state = DeviceState.READY
        return self

    def disconnect(self):
        if self._ser:
            try:
                self.emission_off()
            finally:
                self._ser.close()
                self._ser = None
        self.state = DeviceState.DISCONNECTED

    # -- control ---------------------------------------------------------
    def set_power_mw(self, mw: float):
        self._require_serial()
        if mw > self.max_power_mw:
            raise ValueError(f"{mw} mW exceeds software cap {self.max_power_mw} mW")
        self._command(f"SOUR:POW:LEV:IMM:AMPL {mw / 1000.0:.6f}")

    def get_power_mw(self) -> float:
        return float(self._query("SOUR:POW:LEV?")) * 1000.0

    def emission_on(self):
        self._command("SOUR:AM:STAT ON")
        self.state = DeviceState.RUNNING

    def emission_off(self):
        self._command("SOUR:AM:STAT OFF")
        self.state = DeviceState.READY

    # -- telemetry --------------------------------------------------------
    def status(self) -> dict:
        self._require_serial()
        return {
            "state": self.state.value,
            "wavelength_nm": float(self._query("SYST:INF:WAV?")),
            "power_mw": self.get_power_mw(),
            "power_setpoint_mw": float(self._query("SOUR:POW:LEV:IMM:AMPL?")) * 1000.0,
            "diode_temp_c": float(self._query("SOUR:TEMP:DIOD?")),
            "baseplate_temp_c": float(self._query("SOUR:TEMP:BAS?")),
            "faults": self._query("SYST:FAUL?"),
        }

    def _require_serial(self):
        if self._ser is None:
            raise RuntimeError("laser: not connected")
