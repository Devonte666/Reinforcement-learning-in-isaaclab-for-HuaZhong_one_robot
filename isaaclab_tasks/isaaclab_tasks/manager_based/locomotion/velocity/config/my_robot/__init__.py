# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

print(">>> [DEBUG] 开始加载 MyRobot 任务注册表...")

from . import agents
from .flat_env_cfg import MyRobotFlatEnvCfg, MyRobotFlatEnvCfg_PLAY
from .rough_env_cfg import MyRobotRoughEnvCfg, MyRobotRoughEnvCfg_PLAY

print(">>> [DEBUG] 正在执行 gym.register...")

##
# 平地任务注册
##
gym.register(
    id="Isaac-Velocity-Flat-MyRobot-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:MyRobotFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MyRobotFlatPPORunnerCfg",
    },
)

##
# 崎岖地形任务注册
##
gym.register(
    id="Isaac-Velocity-Rough-MyRobot-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:MyRobotRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MyRobotRoughPPORunnerCfg",
    },
)

##
# PLAY 版本注册
##
gym.register(
    id="Isaac-Velocity-Flat-MyRobot-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:MyRobotFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MyRobotFlatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-MyRobot-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:MyRobotRoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MyRobotRoughPPORunnerCfg",
    },
)

print(">>> [DEBUG] MyRobot 所有任务注册完毕！")