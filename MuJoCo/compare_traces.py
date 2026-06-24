#!/usr/bin/env python3
"""Compare IsaacLab and MuJoCo obs/action/dynamics traces."""

import argparse
import csv
from pathlib import Path

import numpy as np


DEFAULT_JOINT_POS = np.array([0.0, 0.0, -0.3, -0.3, 0.6, 0.6, 0.0, 0.0, -0.3, -0.3], dtype=np.float64)


def load_csv(path):
    with Path(path).open(newline="") as trace_file:
        return list(csv.DictReader(trace_file))


def vec(row, prefix, count):
    return np.array([float(row[f"{prefix}_{i}"]) for i in range(count)], dtype=np.float64)


def named_vec(row, base, count):
    return np.array([float(row[f"{base}_{i}"]) for i in range(count)], dtype=np.float64)


def has_columns(row, names):
    return all(name in row for name in names)


def max_abs(a, b):
    return float(np.max(np.abs(a - b)))


def compare_pre_step(isaac_rows, mujoco_rows, steps):
    print("\n[pre-step obs/action]")
    print("step | obs | action | command | joint_pos | joint_vel | gravity")
    for step in steps:
        if step >= len(isaac_rows) or step >= len(mujoco_rows):
            continue
        isaac = isaac_rows[step]
        mujoco = mujoco_rows[step]
        isaac_obs = vec(isaac, "obs", 42)
        mujoco_obs = vec(mujoco, "obs", 42)
        isaac_action = vec(isaac, "action", 10)
        mujoco_action = vec(mujoco, "action", 10)
        print(
            f"{step:>4} | "
            f"{max_abs(isaac_obs, mujoco_obs):>7.4f} | "
            f"{max_abs(isaac_action, mujoco_action):>7.4f} | "
            f"{max_abs(isaac_obs[9:12], mujoco_obs[9:12]):>7.4f} | "
            f"{max_abs(isaac_obs[12:22], mujoco_obs[12:22]):>9.4f} | "
            f"{max_abs(isaac_obs[22:32], mujoco_obs[22:32]):>9.4f} | "
            f"{max_abs(isaac_obs[6:9], mujoco_obs[6:9]):>7.4f}"
        )


def compare_post_step_fallback(isaac_rows, mujoco_rows, steps):
    print("\n[post-step dynamics: MuJoCo row k vs IsaacLab obs row k+1]")
    print("step | joint_pos | joint_vel | root_lin | root_ang")
    for step in steps:
        if step + 1 >= len(isaac_rows) or step >= len(mujoco_rows):
            continue
        isaac_next = isaac_rows[step + 1]
        mujoco = mujoco_rows[step]
        mujoco_joint_rel = named_vec(mujoco, "joint_pos_abs", 10) - DEFAULT_JOINT_POS
        mujoco_joint_vel = named_vec(mujoco, "joint_vel_abs", 10)
        mujoco_root_lin = named_vec(mujoco, "root_lin_vel_b", 3)
        mujoco_root_ang = named_vec(mujoco, "root_ang_vel_b", 3)
        isaac_joint_rel = vec(isaac_next, "obs", 42)[12:22]
        isaac_joint_vel = vec(isaac_next, "obs", 42)[22:32]
        isaac_root_lin = vec(isaac_next, "obs", 42)[0:3]
        isaac_root_ang = vec(isaac_next, "obs", 42)[3:6]
        print(
            f"{step:>4} | "
            f"{max_abs(mujoco_joint_rel, isaac_joint_rel):>9.4f} | "
            f"{max_abs(mujoco_joint_vel, isaac_joint_vel):>9.4f} | "
            f"{max_abs(mujoco_root_lin, isaac_root_lin):>8.4f} | "
            f"{max_abs(mujoco_root_ang, isaac_root_ang):>8.4f}"
        )


def compare_post_step_debug(isaac_rows, mujoco_rows, steps):
    print("\n[post-step dynamics: debug columns row-to-row]")
    print("step | target | joint_pos | joint_vel | computed_tau | applied_tau | root_pos | contact_norm")
    for step in steps:
        if step >= len(isaac_rows) or step >= len(mujoco_rows):
            continue
        isaac = isaac_rows[step]
        mujoco = mujoco_rows[step]
        contact_isaac = np.array(
            [float(isaac["left_foot_force_norm"]), float(isaac["right_foot_force_norm"])], dtype=np.float64
        )
        contact_mujoco = np.array(
            [float(mujoco["left_foot_force_norm"]), float(mujoco["right_foot_force_norm"])], dtype=np.float64
        )
        print(
            f"{step:>4} | "
            f"{max_abs(named_vec(isaac, 'target_pos', 10), named_vec(mujoco, 'target_pos', 10)):>7.4f} | "
            f"{max_abs(named_vec(isaac, 'joint_pos_abs', 10), named_vec(mujoco, 'joint_pos_abs', 10)):>9.4f} | "
            f"{max_abs(named_vec(isaac, 'joint_vel_abs', 10), named_vec(mujoco, 'joint_vel_abs', 10)):>9.4f} | "
            f"{max_abs(named_vec(isaac, 'computed_torque', 10), named_vec(mujoco, 'computed_torque', 10)):>12.4f} | "
            f"{max_abs(named_vec(isaac, 'applied_torque', 10), named_vec(mujoco, 'applied_torque', 10)):>11.4f} | "
            f"{max_abs(named_vec(isaac, 'root_pos', 3), named_vec(mujoco, 'root_pos', 3)):>8.4f} | "
            f"{max_abs(contact_isaac, contact_mujoco):>12.4f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Compare IsaacLab and MuJoCo trace CSV files.")
    parser.add_argument("--isaac", default="/home/user/Devonte_file/isaaclab_debug_trace.csv")
    parser.add_argument("--mujoco", default="/home/user/Devonte_file/mujoco_position_replay_trace.csv")
    parser.add_argument("--steps", type=int, nargs="*", default=[0, 1, 2, 5, 10, 20, 50, 90])
    args = parser.parse_args()

    isaac_rows = load_csv(args.isaac)
    mujoco_rows = load_csv(args.mujoco)
    if not isaac_rows or not mujoco_rows:
        raise RuntimeError("Both trace files must contain at least one row.")

    print(f"Isaac rows: {len(isaac_rows)}")
    print(f"MuJoCo rows: {len(mujoco_rows)}")
    compare_pre_step(isaac_rows, mujoco_rows, args.steps)

    debug_columns = ["joint_pos_abs_0", "applied_torque_0", "left_foot_force_norm"]
    if has_columns(isaac_rows[0], debug_columns) and has_columns(mujoco_rows[0], debug_columns):
        compare_post_step_debug(isaac_rows, mujoco_rows, args.steps)
    elif has_columns(mujoco_rows[0], debug_columns):
        compare_post_step_fallback(isaac_rows, mujoco_rows, args.steps)
    else:
        print("\nNo dynamics debug columns found in MuJoCo trace.")


if __name__ == "__main__":
    main()
