# MuJoCo Sim-to-Sim

This folder contains the MuJoCo side of the IsaacLab -> MuJoCo validation for MyRobot.

## Files

- `run_policy_mujoco.py`: loads the MJCF model and exported ONNX policy, builds the 42-dim observation, and runs the policy in MuJoCo.
- `compare_traces.py`: compares IsaacLab and MuJoCo trace CSV files.
- `convert_stl.py`: helper for mesh conversion.
- `screenshot.png`: visual reference for the current MuJoCo model.

## Run

```bash
conda activate mujoco_rl
cd /home/user/Devonte_file/Reinforcement-learning-in-isaaclab-for-HuaZhong_one_robot/MuJoCo

python run_policy_mujoco.py \
  --mjcf_path /home/user/Devonte_file/Reinforcement-learning-in-isaaclab-for-HuaZhong_one_robot/Huazhong1/urdf/Huazhong1.xml \
  --onnx_path /home/user/Devonte_file/IsaacLab/logs/rsl_rl/my_robot_flat/2026-06-18_15-46-24/exported/policy.onnx \
  --command_vel 1.0 0.0 0.0
```

## Important Alignment Rules

- The policy input is 42-dim and must match the IsaacLab observation order.
- Joint order is aligned to the IsaacLab trace metadata, not guessed from the MJCF file.
- The policy outputs joint position commands, not direct torques.
- The target position is `default_joint_pos + 0.5 * action`.
- Actuator ids are mapped explicitly from MuJoCo actuators to the policy joint order.
- If the observation/action layout changes, retrain or fine-tune and re-export the policy.

## Trace Comparison

First export an IsaacLab trace with:

```bash
conda activate env_isaaclab
cd /home/user/Devonte_file/IsaacLab

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_trace.py \
  --task Isaac-Velocity-Flat-MyRobot-Play-v0 \
  --num_envs 1 \
  --checkpoint /home/user/Devonte_file/IsaacLab/logs/rsl_rl/my_robot_flat/2026-06-18_15-46-24/model_1499.pt \
  --headless \
  --trace_path /home/user/Devonte_file/isaaclab_dynamics_trace.csv \
  --trace_steps 100
```

Run MuJoCo with an IsaacLab action trace:

```bash
python run_policy_mujoco.py \
  --headless \
  --control_mode position \
  --replay_trace /home/user/Devonte_file/isaaclab_dynamics_trace.csv \
  --trace_path /home/user/Devonte_file/mujoco_position_replay_trace.csv \
  --trace_steps 100 \
  --duration 2.1
```

Compare traces:

```bash
python compare_traces.py \
  --isaac /home/user/Devonte_file/isaaclab_dynamics_trace.csv \
  --mujoco /home/user/Devonte_file/mujoco_position_replay_trace.csv
```
