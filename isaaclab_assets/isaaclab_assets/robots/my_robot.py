# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for my_robot humanoid robot."""

import os
import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg

##
# Configuration
##

# USD 路径（使用绝对路径）
MY_ROBOT_USD_PATH = r"D:\Devonte_file\Robot\Unitree\IsaacLab\source\isaaclab_assets\data\Robots\MyRobot\my_robot.usd"
"""Path to the my_robot robot USD file."""


MY_ROBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=MY_ROBOT_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.15),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "Left_leg_yaw_joint": 0.0,
            "Left_leg_roll_joint": 0.0,
            "Left_leg_pitch_joint": -0.3,
            "Left_knee_pitch_joint": 0.6,
            "Left_ankle_roll_joint": 0.0,
            "Left_ankle_pitch_joint": -0.3,
            "Right_leg_yaw_joint": 0.0,
            "Right_leg_roll_joint": 0.0,
            "Right_leg_pitch_joint": -0.3,
            "Right_knee_pitch_joint": 0.6,
            "Right_ankle_roll_joint": 0.0,
            "Right_ankle_pitch_joint": -0.3,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": DCMotorCfg(
            joint_names_expr=[
                ".*_leg_yaw_joint",
                ".*_leg_roll_joint",
                ".*_leg_pitch_joint",
                ".*_knee_pitch_joint",
            ],
            effort_limit={
                ".*_leg_yaw_joint": 264.0,      # 88 * 3
                ".*_leg_roll_joint": 264.0,       # 88 * 3
                ".*_leg_pitch_joint": 264.0,    # 88 * 3
                ".*_knee_pitch_joint": 417.0,   # 139 * 3
            },
            velocity_limit={
                ".*_leg_yaw_joint": 20.0,
                ".*_leg_roll_joint": 20.0,
                ".*_leg_pitch_joint": 20.0,
                ".*_knee_pitch_joint": 20.0,
            },
            stiffness={
                ".*_leg_yaw_joint": 100.0,
                ".*_leg_roll_joint": 100.0,
                ".*_leg_pitch_joint": 200.0,
                ".*_knee_pitch_joint": 200.0,
            },
            damping={
                ".*_leg_yaw_joint": 2.5,
                ".*_leg_roll_joint": 2.5,
                ".*_leg_pitch_joint": 2.5,
                ".*_knee_pitch_joint": 5.0,
            },
            armature={
                ".*_leg_yaw_joint": 0.03,
                ".*_leg_roll_joint": 0.03,
                ".*_leg_pitch_joint": 0.03,
                ".*_knee_pitch_joint": 0.03,
            },
            saturation_effort=540.0,  # 180 * 3
        ),
        "feet": DCMotorCfg(
            joint_names_expr=[
                ".*_ankle_roll_joint",
                ".*_ankle_pitch_joint",
            ],
            stiffness={
                ".*_ankle_roll_joint": 20.0,
                ".*_ankle_pitch_joint": 20.0,
            },
            damping={
                ".*_ankle_roll_joint": 0.2,
                ".*_ankle_pitch_joint": 0.2,
            },
            effort_limit={
                ".*_ankle_roll_joint": 150.0,    # 50 * 3
                ".*_ankle_pitch_joint": 150.0,   # 50 * 3
            },
            velocity_limit={
                ".*_ankle_roll_joint": 20.0,
                ".*_ankle_pitch_joint": 20.0,
            },
            armature=0.03,
            saturation_effort=240.0,  # 80 * 3
        ),
    },
    prim_path="/World/envs/env_.*/Robot",
)
"""Configuration for the my_robot humanoid robot."""


MY_ROBOT_FIXED_BASE_CFG = MY_ROBOT_CFG.copy()
MY_ROBOT_FIXED_BASE_CFG.spawn.articulation_props.fix_root_link = True
"""Configuration for my_robot with fixed base for upper body debugging."""