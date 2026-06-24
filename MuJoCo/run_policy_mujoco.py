#!/usr/bin/env python3
"""
Run IsaacLab-trained policy in MuJoCo for Sim-to-Sim validation.
"""

import argparse
import csv
import os
import signal
import numpy as np
import mujoco
import mujoco.viewer
import onnxruntime as ort
import time

# ========== 配置 ==========
MJCF_PATH = "/home/user/Devonte_file/Reinforcement-learning-in-isaaclab-for-HuaZhong_one_robot/Huazhong1/urdf/Huazhong1.xml"
ONNX_PATH = "/home/user/Devonte_file/IsaacLab/logs/rsl_rl/my_robot_flat/2026-06-18_15-46-24/exported/policy.onnx"

JOINT_NAMES = [
    "Left_leg_roll_joint",
    "Right_leg_roll_joint",
    "Left_leg_pitch_joint",
    "Right_leg_pitch_joint",
    "Left_knee_pitch_joint",
    "Right_knee_pitch_joint",
    "Left_ankle_roll_joint",
    "Right_ankle_roll_joint",
    "Left_ankle_pitch_joint",
    "Right_ankle_pitch_joint",
]

# 默认关节位置（与 IsaacLab action/observation joint order 对齐）
DEFAULT_JOINT_POS = np.array([
    0.0,   # Left_leg_roll
    0.0,   # Right_leg_roll
    -0.3,  # Left_leg_pitch
    -0.3,  # Right_leg_pitch
    0.6,   # Left_knee_pitch
    0.6,   # Right_knee_pitch
    0.0,   # Left_ankle_roll
    0.0,   # Right_ankle_roll
    -0.3,  # Left_ankle_pitch
    -0.3,  # Right_ankle_pitch
])
INIT_BASE_POS = np.array([0.0, 0.0, 1.15])
# Keep IsaacLab's init_state height by default. Enable this only to debug contact geometry.
AUTO_FIT_BASE_TO_FLOOR = False
FOOT_CLEARANCE = 0.002

# 动作缩放（与 IsaacLab 配置一致）
ACTION_SCALE = 0.5  # 需要根据你的配置调整
# IsaacLab/RSL-RL log says clip_actions=null, so do not clip raw policy actions here.
ACTION_CLIP = None

KP = np.array([100.0, 100.0, 200.0, 200.0, 180.0, 180.0, 20.0, 20.0, 20.0, 20.0], dtype=np.float32)
KD = np.array([2.5, 2.5, 2.5, 2.5, 5.0, 5.0, 0.2, 0.2, 0.2, 0.2], dtype=np.float32)
EFFORT_LIMIT = np.array([264.0, 264.0, 264.0, 264.0, 264.0, 264.0, 150.0, 150.0, 150.0, 150.0], dtype=np.float32)
VELOCITY_LIMIT = np.full(len(JOINT_NAMES), 20.0, dtype=np.float32)
SATURATION_EFFORT = np.array([540.0, 540.0, 540.0, 540.0, 540.0, 540.0, 240.0, 240.0, 240.0, 240.0], dtype=np.float32)

# IsaacLab: sim.dt=0.005, decimation=4, so the policy runs every 0.02s.
CONTROL_DT = 0.02

# Add --debug_obs to print full observation/action slices.
DEBUG_OBS = False
DEBUG_EVERY_POLICY_STEPS = 10

# 速度指令（与训练时一致）
COMMAND_VEL = np.array([1.0, 0.0, 0.0])  # [vx, vy, yaw_rate]

# 观测历史（用于 actions 部分）
last_action = np.zeros(len(JOINT_NAMES), dtype=np.float32)
stop_requested = False


def parse_args():
    """Parse command-line options for viewer/headless trace runs."""
    parser = argparse.ArgumentParser(description="Run IsaacLab-trained policy in MuJoCo.")
    parser.add_argument("--mjcf_path", type=str, default=None, help="Override MJCF model path.")
    parser.add_argument("--onnx_path", type=str, default=None, help="Override ONNX policy path.")
    parser.add_argument(
        "--replay_trace",
        type=str,
        default=None,
        help="Replay action_i columns from an IsaacLab trace CSV instead of running the ONNX policy.",
    )
    parser.add_argument(
        "--control_mode",
        choices=("dc_motor", "position"),
        default="position",
        help="Use IsaacLab-style explicit DCMotor torque control or MuJoCo position actuators.",
    )
    parser.add_argument("--headless", action="store_true", help="Run without launching the MuJoCo viewer.")
    parser.add_argument("--duration", type=float, default=20.0, help="Simulation duration in seconds.")
    parser.add_argument(
        "--mujoco_timestep",
        type=float,
        default=None,
        help="Override MuJoCo physics timestep. IsaacLab uses 0.005 for this task.",
    )
    parser.add_argument(
        "--command_vel",
        type=float,
        nargs=3,
        default=None,
        metavar=("VX", "VY", "WZ"),
        help="Velocity command [vx, vy, yaw_rate] used in the policy observation.",
    )
    parser.add_argument("--trace_path", type=str, default=None, help="Optional CSV path for obs/action tracing.")
    parser.add_argument(
        "--trace_steps",
        type=int,
        default=None,
        help="Number of policy steps to record. Defaults to 100 when --trace_path is set.",
    )
    parser.add_argument(
        "--real_time",
        action="store_true",
        help="Sleep to wall-clock time. Headless mode otherwise runs as fast as possible.",
    )
    parser.add_argument("--debug_obs", action="store_true", help="Print full observation/action slices while running.")
    return parser.parse_args()


def load_replay_trace(trace_path):
    """Load actions and velocity commands from an IsaacLab/MuJoCo trace CSV."""
    with open(trace_path, newline="") as trace_file:
        rows = list(csv.DictReader(trace_file))
    if not rows:
        raise ValueError(f"Replay trace is empty: {trace_path}")

    actions = np.asarray(
        [[float(row[f"action_{i}"]) for i in range(len(JOINT_NAMES))] for row in rows],
        dtype=np.float32,
    )
    commands = np.asarray([[float(row[f"obs_{i}"]) for i in range(9, 12)] for row in rows], dtype=np.float32)
    return actions, commands


def get_debug_header(action_dim):
    """Build extra trace columns after obs/action."""
    return (
        [f"target_pos_{i}" for i in range(action_dim)]
        + [f"joint_pos_abs_{i}" for i in range(action_dim)]
        + [f"joint_vel_abs_{i}" for i in range(action_dim)]
        + [f"computed_torque_{i}" for i in range(action_dim)]
        + [f"applied_torque_{i}" for i in range(action_dim)]
        + [f"root_pos_{i}" for i in range(3)]
        + [f"root_quat_{i}" for i in range(4)]
        + [f"root_lin_vel_b_{i}" for i in range(3)]
        + [f"root_ang_vel_b_{i}" for i in range(3)]
        + [f"left_foot_force_{i}" for i in range(3)]
        + [f"right_foot_force_{i}" for i in range(3)]
        + ["left_foot_force_norm", "right_foot_force_norm"]
    )


def request_stop(signum=None, frame=None):
    """Request a clean shutdown from Ctrl+C/SIGTERM."""
    global stop_requested
    stop_requested = True
    print("\n[INFO] Stop requested, closing MuJoCo viewer...")


def box_geom_bottom_z(model, data, geom_id):
    """Return the world-frame bottom z of a box geom."""
    half_size = model.geom_size[geom_id]
    rot = data.geom_xmat[geom_id].reshape(3, 3)
    world_z_extent = np.sum(np.abs(rot[2, :]) * half_size)
    return data.geom_xpos[geom_id, 2] - world_z_extent


def fit_base_height_to_feet(model, data, joint_qposadr):
    """Shift the floating base so both foot collision boxes start on the floor."""
    foot_geom_names = ["Left_ankle_pitch_collision", "Right_ankle_pitch_collision", "Left_foot", "Right_foot"]
    foot_geom_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in foot_geom_names
    ]
    foot_geom_ids = [geom_id for geom_id in foot_geom_ids if geom_id >= 0]
    if not foot_geom_ids:
        print("[WARN] No foot collision geoms found; using INIT_BASE_POS without height fitting.")
        return

    mujoco.mj_forward(model, data)
    min_bottom_z = min(box_geom_bottom_z(model, data, geom_id) for geom_id in foot_geom_ids)
    z_shift = FOOT_CLEARANCE - min_bottom_z
    data.qpos[2] += z_shift
    mujoco.mj_forward(model, data)
    print(
        f"[INFO] Auto-fit base height: z shift {z_shift:+.4f} m, "
        f"base z {data.qpos[2]:.4f} m"
    )


def print_initial_state(model, data, base_body_id, joint_qposadr, joint_dofadr):
    obs = get_observation(model, data, base_body_id, joint_qposadr, joint_dofadr)
    print(f"\n{'=' * 72}")
    print("[INITIAL STATE]")
    print(f"  Base pos:             {data.qpos[0:3]}")
    print(f"  Base quat wxyz:       {data.qpos[3:7]}")
    print(f"  Joint order:          {JOINT_NAMES}")
    print(f"  Joint pos:            {data.qpos[joint_qposadr]}")
    print(f"  Joint vel:            {data.qvel[joint_dofadr]}")
    print(f"  base_lin_vel obs:     {obs[0:3]}")
    print(f"  base_ang_vel obs:     {obs[3:6]}")
    print(f"  projected_gravity:    {obs[6:9]}")
    print(f"  velocity_commands:    {obs[9:12]}")
    print(f"  joint_pos_rel:        {obs[12:22]}")
    print(f"  joint_vel_rel:        {obs[22:32]}")
    print(f"  last_action:          {obs[32:42]}")
    print(f"{'=' * 72}\n")


# ========== 加载模型 ==========
def load_models():
    """Load MuJoCo model and ONNX policy."""
    # 检查文件
    if not os.path.exists(MJCF_PATH):
        raise FileNotFoundError(f"MJCF not found: {MJCF_PATH}")
    if not os.path.exists(ONNX_PATH):
        raise FileNotFoundError(f"ONNX not found: {ONNX_PATH}\n"
                                f"Please run play.py with --checkpoint to export policy first.")
    
    # 加载 MuJoCo
    model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    data = mujoco.MjData(model)

    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    if base_body_id < 0:
        raise ValueError("MuJoCo model does not contain body 'base_link'.")

    joint_qposadr = []
    joint_dofadr = []
    actuator_ids = []
    for joint_name in JOINT_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"MuJoCo model does not contain joint '{joint_name}'.")
        joint_qposadr.append(model.jnt_qposadr[joint_id])
        joint_dofadr.append(model.jnt_dofadr[joint_id])

        matching_actuators = np.where(model.actuator_trnid[:, 0] == joint_id)[0]
        if len(matching_actuators) != 1:
            raise ValueError(
                f"Expected exactly one actuator for joint '{joint_name}', found {len(matching_actuators)}."
            )
        actuator_ids.append(int(matching_actuators[0]))

    joint_qposadr = np.asarray(joint_qposadr, dtype=np.int32)
    joint_dofadr = np.asarray(joint_dofadr, dtype=np.int32)
    actuator_ids = np.asarray(actuator_ids, dtype=np.int32)
    
    # 加载 ONNX 策略
    session = ort.InferenceSession(ONNX_PATH)
    input_name = session.get_inputs()[0].name
    
    print(f"[INFO] MuJoCo model loaded: {model.nq} DOF, {model.nu} actuators")
    print(f"[INFO] MuJoCo total mass: {np.sum(model.body_mass):.3f} kg")
    print(f"[INFO] ONNX policy loaded: input shape {session.get_inputs()[0].shape}")
    print(f"[INFO] Policy control dt: {CONTROL_DT:.3f}s")
    
    return model, data, session, input_name, base_body_id, joint_qposadr, joint_dofadr, actuator_ids


# ========== 观测构建 ==========
def get_observation(model, data, base_body_id, joint_qposadr, joint_dofadr):
    """Build 42-dim observation vector matching IsaacLab format."""
    global last_action
    
    # --- 1. Base linear velocity (3) in body frame ---
    # IsaacLab returns root COM velocity in the root frame. MuJoCo returns
    # spatial velocity as [angular, linear] when flg_local=1.
    rot_mat = data.xmat[base_body_id].reshape(3, 3)
    base_velocity = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, base_body_id, base_velocity, 1)
    base_ang_vel = base_velocity[0:3]
    base_lin_vel = base_velocity[3:6]
    
    # --- 2. Base angular velocity (3) in body frame ---
    # Computed above from MuJoCo's local object velocity.
    
    # --- 3. Projected gravity (3) ---
    # IsaacLab normalizes gravity before projecting it, so this must be a unit vector.
    gravity_world = np.array([0.0, 0.0, -1.0])
    projected_gravity = rot_mat.T @ gravity_world
    
    # --- 4. Velocity commands (3) ---
    velocity_commands = COMMAND_VEL
    
    # --- 5. Joint positions (10) relative to default ---
    joint_pos = data.qpos[joint_qposadr]
    joint_pos_rel = joint_pos - DEFAULT_JOINT_POS
    
    # --- 6. Joint velocities (10) ---
    joint_vel = data.qvel[joint_dofadr]
    
    # --- 7. Last actions (10) ---
    actions = last_action
    
    # Concatenate
    obs = np.concatenate([
        base_lin_vel,
        base_ang_vel,
        projected_gravity,
        velocity_commands,
        joint_pos_rel,
        joint_vel,
        actions
    ])

    return obs.astype(np.float32)


def configure_torque_actuators(model):
    """Make MuJoCo actuators behave like direct torque motors at runtime."""
    if model.nu != len(JOINT_NAMES):
        raise ValueError(f"Expected {len(JOINT_NAMES)} actuators, got {model.nu}.")

    model.actuator_gaintype[:] = mujoco.mjtGain.mjGAIN_FIXED
    model.actuator_biastype[:] = mujoco.mjtBias.mjBIAS_NONE
    model.actuator_dyntype[:] = mujoco.mjtDyn.mjDYN_NONE
    model.actuator_gainprm[:, :] = 0.0
    model.actuator_biasprm[:, :] = 0.0
    model.actuator_dynprm[:, :] = 0.0
    model.actuator_gainprm[:, 0] = 1.0
    print("[INFO] MuJoCo actuators converted to direct torque motors for DCMotor control.")


def compute_dc_motor_torque(data, joint_qposadr, joint_dofadr, target_pos):
    """Compute IsaacLab DCMotor torque from target position and current joint state."""
    joint_pos = data.qpos[joint_qposadr].astype(np.float32)
    joint_vel = data.qvel[joint_dofadr].astype(np.float32)

    computed_torque = KP * (target_pos.astype(np.float32) - joint_pos) - KD * joint_vel

    vel_at_effort_lim = VELOCITY_LIMIT * (1.0 + EFFORT_LIMIT / SATURATION_EFFORT)
    clipped_joint_vel = np.clip(joint_vel, -vel_at_effort_lim, vel_at_effort_lim)
    torque_speed_top = SATURATION_EFFORT * (1.0 - clipped_joint_vel / VELOCITY_LIMIT)
    torque_speed_bottom = SATURATION_EFFORT * (-1.0 - clipped_joint_vel / VELOCITY_LIMIT)
    max_effort = np.minimum(torque_speed_top, EFFORT_LIMIT)
    min_effort = np.maximum(torque_speed_bottom, -EFFORT_LIMIT)
    applied_torque = np.clip(computed_torque, min_effort, max_effort)

    return computed_torque.astype(np.float32), applied_torque.astype(np.float32)


def get_foot_body_ids(model):
    """Resolve MuJoCo foot bodies used for contact-force diagnostics."""
    left_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Left_ankle_pitch_link")
    right_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Right_ankle_pitch_link")
    return left_body_id, right_body_id


def get_foot_contact_forces(model, data, foot_body_ids):
    """Return approximate external contact forces on left/right foot bodies."""
    mujoco.mj_rnePostConstraint(model, data)
    forces = []
    for body_id in foot_body_ids:
        if body_id < 0:
            forces.append(np.full(3, np.nan, dtype=np.float32))
        else:
            forces.append(np.asarray(data.cfrc_ext[body_id, 3:6], dtype=np.float32).copy())
    return forces[0], forces[1]


def get_debug_values(
    model,
    data,
    base_body_id,
    joint_qposadr,
    joint_dofadr,
    target_pos,
    computed_torque,
    applied_torque,
    foot_body_ids,
):
    """Collect post-step state/action diagnostics in the same column order as play_trace.py."""
    post_obs = get_observation(model, data, base_body_id, joint_qposadr, joint_dofadr)
    left_foot_force, right_foot_force = get_foot_contact_forces(model, data, foot_body_ids)
    values = np.concatenate(
        [
            target_pos.astype(np.float32),
            data.qpos[joint_qposadr].astype(np.float32),
            data.qvel[joint_dofadr].astype(np.float32),
            computed_torque.astype(np.float32),
            applied_torque.astype(np.float32),
            data.qpos[0:3].astype(np.float32),
            data.qpos[3:7].astype(np.float32),
            post_obs[0:3].astype(np.float32),
            post_obs[3:6].astype(np.float32),
            left_foot_force.astype(np.float32),
            right_foot_force.astype(np.float32),
            np.asarray([np.linalg.norm(left_foot_force), np.linalg.norm(right_foot_force)], dtype=np.float32),
        ]
    )
    expected = len(get_debug_header(len(JOINT_NAMES)))
    if values.shape[0] != expected:
        raise RuntimeError(f"Unexpected debug value count: {values.shape[0]} != {expected}.")
    return values


# ========== 动作执行 ==========
def apply_action(model, data, action, joint_qposadr, joint_dofadr, actuator_ids, control_mode):
    """Convert policy action to MuJoCo control."""
    global last_action
    
    raw_action = action.astype(np.float32)
    if ACTION_CLIP is None:
        action_to_apply = raw_action
    else:
        action_to_apply = np.clip(raw_action, -ACTION_CLIP, ACTION_CLIP)

    # IsaacLab action is typically: target_pos = default_pos + action * scale
    target_pos = DEFAULT_JOINT_POS + action_to_apply * ACTION_SCALE

    if control_mode == "dc_motor":
        computed_torque, applied_torque = compute_dc_motor_torque(data, joint_qposadr, joint_dofadr, target_pos)
        data.ctrl[actuator_ids] = applied_torque
    else:
        data.ctrl[actuator_ids] = target_pos
        computed_torque = np.full(len(JOINT_NAMES), np.nan, dtype=np.float32)
        applied_torque = np.full(len(JOINT_NAMES), np.nan, dtype=np.float32)
    
    # IsaacLab's last_action observation is the raw action sent to the action manager.
    last_action = raw_action.copy()

    return action_to_apply, target_pos, computed_torque, applied_torque


def print_policy_debug(policy_step_count, data, obs, raw_action, action_to_apply, target_pos, joint_qposadr):
    """Print observation/action slices in the same order as IsaacLab policy input."""
    print(f"\n{'=' * 72}")
    print(f"[POLICY STEP {policy_step_count}]  sim time: {data.time:.3f}s")
    print(f"  base height:          {data.qpos[2]:.4f} m")
    print(f"  base quat wxyz:       {data.qpos[3:7]}")
    print(f"  base_lin_vel obs:     {obs[0:3]}")
    print(f"  base_ang_vel obs:     {obs[3:6]}")
    print(f"  projected_gravity:    {obs[6:9]}")
    print(f"  velocity_commands:    {obs[9:12]}")
    print(f"  joint_pos_rel:        {obs[12:22]}")
    print(f"  joint_vel_rel:        {obs[22:32]}")
    print(f"  last_action:          {obs[32:42]}")
    print(f"  raw_action:           {raw_action}")
    print(f"  action_to_apply:      {action_to_apply}")
    print(f"  target_pos:           {target_pos}")
    print(f"  current_joint_pos:    {data.qpos[joint_qposadr]}")
    print(f"{'=' * 72}")


# ========== 主循环 ==========
def main():
    global COMMAND_VEL, DEBUG_OBS, MJCF_PATH, ONNX_PATH, stop_requested
    args = parse_args()
    if args.mjcf_path is not None:
        MJCF_PATH = args.mjcf_path
    if args.onnx_path is not None:
        ONNX_PATH = args.onnx_path
    if args.trace_path and args.trace_steps is None:
        args.trace_steps = 100

    replay_actions = None
    replay_commands = None
    if args.replay_trace is not None:
        replay_actions, replay_commands = load_replay_trace(args.replay_trace)
        if args.trace_steps is None:
            args.trace_steps = len(replay_actions)
        print(f"[INFO] Replaying {len(replay_actions)} actions from: {args.replay_trace}")

    if args.command_vel is not None:
        COMMAND_VEL = np.asarray(args.command_vel, dtype=np.float32)
    elif replay_commands is not None:
        COMMAND_VEL = replay_commands[0].copy()
    DEBUG_OBS = args.debug_obs

    stop_requested = False
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    model, data, session, input_name, base_body_id, joint_qposadr, joint_dofadr, actuator_ids = load_models()
    if args.mujoco_timestep is not None:
        model.opt.timestep = args.mujoco_timestep
    if args.control_mode == "dc_motor":
        configure_torque_actuators(model)
    foot_body_ids = get_foot_body_ids(model)
    control_decimation = max(1, int(round(CONTROL_DT / model.opt.timestep)))
    actual_control_dt = control_decimation * model.opt.timestep
    print(f"[INFO] MuJoCo timestep: {model.opt.timestep:.4f}s")
    print(f"[INFO] MuJoCo steps per policy action: {control_decimation} ({actual_control_dt:.4f}s)\n")
    
    # 重置到初始姿态
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = INIT_BASE_POS
    data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
    data.qpos[joint_qposadr] = DEFAULT_JOINT_POS
    data.qvel[:] = 0.0
    if args.control_mode == "dc_motor":
        data.ctrl[:] = 0.0
    else:
        data.ctrl[actuator_ids] = DEFAULT_JOINT_POS
    mujoco.mj_forward(model, data)
    if AUTO_FIT_BASE_TO_FLOOR:
        fit_base_height_to_feet(model, data, joint_qposadr)
    print_initial_state(model, data, base_body_id, joint_qposadr, joint_dofadr)

    trace_file = None
    trace_writer = None
    if args.trace_path:
        trace_path = os.path.abspath(args.trace_path)
        trace_dir = os.path.dirname(trace_path)
        if trace_dir:
            os.makedirs(trace_dir, exist_ok=True)
        trace_file = open(trace_path, "w", newline="")
        trace_writer = csv.writer(trace_file)
        trace_writer.writerow(
            ["step", "time"]
            + [f"obs_{i}" for i in range(42)]
            + [f"action_{i}" for i in range(len(JOINT_NAMES))]
            + get_debug_header(len(JOINT_NAMES))
        )
        print(f"[INFO] Writing MuJoCo policy trace to: {trace_path}")
    
    print("\n[INFO] Starting simulation...")
    viewer = None
    if args.headless:
        print("[INFO] Headless mode enabled; no MuJoCo viewer will be opened.\n")
    else:
        print("[INFO] Press 'Space' in viewer to pause/resume")
        print("[INFO] Close viewer window to exit\n")
        viewer = mujoco.viewer.launch_passive(model, data)
        viewer.cam.distance = 4.0
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -20
        viewer.cam.lookat[:] = np.array([0.0, 0.0, 0.6])

    start_time = time.time()
    step_count = 0
    policy_step_count = 0

    try:
        while not stop_requested and data.time < args.duration:
            if viewer is not None and not viewer.is_running():
                break
            if replay_actions is not None and policy_step_count >= len(replay_actions):
                break

            if replay_commands is not None:
                COMMAND_VEL = replay_commands[policy_step_count].copy()

            # 1. 获取观测
            obs = get_observation(model, data, base_body_id, joint_qposadr, joint_dofadr)
            
            # 2. 策略推理，或从 IsaacLab trace 中开环回放动作
            if replay_actions is None:
                obs_batch = obs.reshape(1, -1)
                raw_action = session.run(None, {input_name: obs_batch})[0].flatten()
            else:
                raw_action = replay_actions[policy_step_count].copy()

            # 3. 应用动作
            action_to_apply, target_pos, computed_torque, applied_torque = apply_action(
                model,
                data,
                raw_action,
                joint_qposadr,
                joint_dofadr,
                actuator_ids,
                args.control_mode,
            )
            
            # 4. 在两个 policy action 之间保持同一个目标位置
            for _ in range(control_decimation):
                if stop_requested or data.time >= args.duration:
                    break
                if viewer is not None and not viewer.is_running():
                    break
                if args.control_mode == "dc_motor":
                    computed_torque, applied_torque = compute_dc_motor_torque(
                        data,
                        joint_qposadr,
                        joint_dofadr,
                        target_pos,
                    )
                    data.ctrl[actuator_ids] = applied_torque
                mujoco.mj_step(model, data)
                if viewer is not None:
                    viewer.sync()
                step_count += 1

                if args.real_time or viewer is not None:
                    elapsed = time.time() - start_time
                    sleep_time = data.time - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)

            if trace_writer is not None and (
                args.trace_steps is None or policy_step_count < args.trace_steps
            ):
                debug_values = get_debug_values(
                    model,
                    data,
                    base_body_id,
                    joint_qposadr,
                    joint_dofadr,
                    target_pos,
                    computed_torque,
                    applied_torque,
                    foot_body_ids,
                )
                trace_writer.writerow(
                    [
                        policy_step_count,
                        policy_step_count * CONTROL_DT,
                        *obs.tolist(),
                        *raw_action.tolist(),
                        *debug_values.tolist(),
                    ]
                )
                trace_file.flush()
	            
            # 打印状态
            policy_step_count += 1

            if trace_writer is not None and args.trace_steps is not None and policy_step_count >= args.trace_steps:
                break

            if DEBUG_OBS and policy_step_count % DEBUG_EVERY_POLICY_STEPS == 0:
                print_policy_debug(
                    policy_step_count,
                    data,
                    obs,
                    raw_action,
                    action_to_apply,
                    target_pos,
                    joint_qposadr,
                )
            elif policy_step_count % 10 == 0:
                print(f"Time: {data.time:.2f}s | "
                      f"Base height: {data.qpos[2]:.3f}m | "
                      f"Raw action mean: {np.mean(np.abs(raw_action)):.3f} | "
                      f"Obs gravity z: {obs[8]:.3f}")
    except KeyboardInterrupt:
        request_stop()
    finally:
        if trace_file is not None:
            trace_file.close()
        if viewer is not None:
            try:
                viewer.close()
            except Exception:
                pass
    
    print(f"\n[INFO] Simulation completed: {step_count} steps, {data.time:.2f}s")
    print(f"[INFO] Average FPS: {step_count / (time.time() - start_time):.1f}")


if __name__ == "__main__":
    main()
    
