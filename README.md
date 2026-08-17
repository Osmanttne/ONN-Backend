# ONN Bench Backend

Notebook-first backend for the unified ONN control project (Pégard Lab).
Core contract: temporal input matrix X (T, h, w) -> DMD -> optics -> camera -> prediction matrix Y (T, gh, gw).

## Layout
- `onn_backend.ipynb` — run this; drives everything, ships pre-executed in sim mode
- `hardware/` — device classes: `laser.py` (OBIS/SCPI), `dmd.py` (ViALUX/ALP4lib), `slm.py` (Meadowlark/slmsuite, coverglass lockout), `camera.py` (FLIR/PySpin), `simulators.py` (no-hardware twins)
- `onn/` — `patterns.py` (named DMD/SLM patterns), `forward.py` (ONNForward: X -> Y, save_result)
- `config/onn_nico.yaml` — bench profile; TODO markers = fill from onn-nico-parameters
- `data/` — saved runs (.npz: X, Y, timestamps, frames, metadata)

## Quick start (any machine, no hardware)
pip install numpy matplotlib pyyaml pillow jupyter
jupyter notebook onn_backend.ipynb   # SIM = True at the top

## Bench PC
Install pyserial + ALP4lib + slmsuite + PySpin (vendor SDKs required),
fill in config TODOs, set SIM = False, run top to bottom.
# ONN-Backend
<img width="3947" height="8105" alt="diagram" src="https://github.com/user-attachments/assets/33ae7646-94f3-430d-bf5d-80f9e38c80c2" />
