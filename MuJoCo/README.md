# MuJoCo Sim-to-Sim

This folder contains the MuJoCo side of the IsaacLab -> MuJoCo validation.

Files:
- `run_policy_mujoco.py`: load the exported ONNX policy and replay it in MuJoCo
- `compare_traces.py`: compare IsaacLab traces with MuJoCo traces
- `convert_stl.py`: helper for mesh conversion
- `screenshot.png`: visual reference for the current model

Notes:
- The policy input is 42-dim and must match the IsaacLab observation order.
- Joint order is aligned to the IsaacLab trace metadata.
- The policy outputs joint position targets, not direct torques.
- If observation/action layout changes, retrain and re-export the policy.
