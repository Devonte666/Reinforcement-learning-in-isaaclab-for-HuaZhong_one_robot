# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np  # ← 新增
import torch

from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # ========== 新增：接触力统计 ==========
    # 获取接触力数据 (batch, history, bodies, 3)
    net_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    force_mag = torch.norm(net_forces, dim=-1)  # (batch, history, bodies)
    max_force_per_foot = torch.max(force_mag, dim=1)[0]  # (batch, bodies)

    # Episode 级接触力统计
    if not hasattr(env, "_ep_contact_forces"):
        env._ep_contact_forces = []  # 当前 episode 的所有接触力
        env._ep_count = 0

    # 收集接触的力值
    is_contact = max_force_per_foot > 1.0
    if is_contact.any():
        # 只收集实际接触的力，转为 CPU list
        contact_values = max_force_per_foot[is_contact].cpu().tolist()
        env._ep_contact_forces.extend(contact_values)

    # 检测 episode 结束（有环境被 reset）
    done = env.termination_manager.terminated | env.termination_manager.time_outs
    if done.any() and len(env._ep_contact_forces) > 0:
        env._ep_count += done.sum().item()

        # 计算统计
        avg_force = sum(env._ep_contact_forces) / len(env._ep_contact_forces)
        max_force = max(env._ep_contact_forces)
        # 95th percentile
        import numpy as np
        p95 = np.percentile(env._ep_contact_forces, 95)

        print(f"\n{'='*65}")
        print(f"[Episode {env._ep_count}] Foot Contact Force Statistics")
        print(f"  Average Force:     {avg_force:>8.1f} N")
        print(f"  Peak Force:        {max_force:>8.1f} N")
        print(f"  95th Percentile:   {p95:>8.1f} N")
        print(f"  Total Contacts:    {len(env._ep_contact_forces):>8}")
        print(f"{'='*65}\n")

        # 清空，开始新 episode
        env._ep_contact_forces = []
    # ======================================

    # 原有逻辑
    contacts = is_contact  # 复用前面的计算
    asset = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def feet_air_time_asymmetry_penalty(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, max_penalty: float = 1.0
) -> torch.Tensor:
    """Penalize asymmetric air time between left and right feet.

    This function computes the absolute difference between the current air time of the left and right feet.
    It penalizes the agent when one foot stays in the air significantly longer than the other,
    which encourages a symmetric gait pattern.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # get current air time for the specified body ids (assumed [left, right])
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    # compute absolute difference between left and right foot air time
    diff = torch.abs(air_time[:, 0] - air_time[:, 1])
    # square the difference to penalize large asymmetries more heavily
    penalty = torch.square(diff)
    # clamp to avoid excessive penalty
    penalty = torch.clamp(penalty, max=max_penalty)
    return penalty


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned
    robot frame using an exponential kernel.
    """
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def stand_still_joint_deviation_l1(
    env, command_name: str, command_threshold: float = 0.06, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize offsets from the default joint positions when the command is very small."""
    command = env.command_manager.get_command(command_name)
    # Penalize motion when command is nearly zero.
    return mdp.joint_deviation_l1(env, asset_cfg) * (torch.norm(command[:, :2], dim=1) < command_threshold)

def knee_pitch_soft_limit(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    limit_deg: float = 35.0,
    max_penalty: float = 1.0,
) -> torch.Tensor:
    """Soft limit penalty for knee pitch joints.

    Penalizes knee angles beyond a specified limit (in degrees).
    The penalty increases quadratically as the angle exceeds the limit.
    """
    asset = env.scene[asset_cfg.name]
    # find knee pitch joints
    knee_indices = asset.find_joints(".*_knee_pitch_joint")[0]

    # get current knee positions (in radians)
    knee_pos = asset.data.joint_pos[:, knee_indices]

    # convert limit to radians
    limit_rad = limit_deg * 3.141592653589793 / 180.0

    # penalty only applies when knee_pos > limit_rad
    # use ReLU: max(0, knee_pos - limit_rad)
    excess = torch.clamp(knee_pos - limit_rad, min=0.0)

    # quadratic penalty: excess^2, scaled
    penalty = torch.square(excess)

    # clamp max penalty per joint, then sum across joints
    penalty = torch.clamp(penalty, max=max_penalty)

    # sum penalty across all knee joints (left + right)
    return torch.sum(penalty, dim=1)