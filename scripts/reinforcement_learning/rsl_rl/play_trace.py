# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--trace_path",
    type=str,
    default="/home/user/Devonte_file/isaaclab_debug_trace.csv",
    help="Path to save the policy observation/action trace as CSV.",
)
parser.add_argument(
    "--trace_meta_path",
    type=str,
    default=None,
    help="Optional path to save trace metadata such as IsaacLab joint ordering.",
)
parser.add_argument("--trace_steps", type=int, default=100, help="Number of policy steps to record before exiting.")
parser.add_argument("--trace_env_id", type=int, default=0, help="Vectorized environment index to record.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for installed RSL-RL version."""

import importlib.metadata as metadata

from packaging import version

installed_version = metadata.version("rsl-rl-lib")

"""Rest everything follows."""

import csv
import os
import time

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
    handle_deprecated_rsl_rl_cfg,
    handle_deprecated_rsl_rl_checkpoint,
)
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)


def _get_policy_observation_tensor(obs):
    """Extract the actor observation tensor from IsaacLab/RSL-RL observations."""
    if hasattr(obs, "keys"):
        if "policy" in obs.keys():
            return obs["policy"]
        if "obs" in obs.keys():
            return obs["obs"]
    return obs


def _get_joint_ids_from_action_term(action_term):
    """Return joint ids in the same order as the policy action term."""
    joint_ids = getattr(action_term, "_joint_ids", slice(None))
    return joint_ids


def _get_foot_contact_forces(base_env, env_id: int):
    """Return summed left/right foot normal contact forces in world frame."""
    left_force = torch.zeros(3, device=base_env.device)
    right_force = torch.zeros(3, device=base_env.device)
    try:
        contact_sensor = base_env.scene["contact_forces"]
        forces = contact_sensor.data.net_forces_w[env_id]
        for body_id, body_name in enumerate(contact_sensor.body_names):
            body_name_lower = body_name.lower()
            is_foot_body = "foot" in body_name_lower or "ankle_pitch" in body_name_lower
            if not is_foot_body:
                continue
            if "left" in body_name_lower:
                left_force += forces[body_id]
            elif "right" in body_name_lower:
                right_force += forces[body_id]
    except Exception:
        left_force[:] = torch.nan
        right_force[:] = torch.nan
    return left_force, right_force


def _get_debug_header(action_dim: int):
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


def _get_debug_values(base_env, env_id: int, action_dim: int):
    """Collect post-step state/action diagnostics for one environment."""
    robot = base_env.scene["robot"]
    action_term = base_env.action_manager.get_term("joint_pos")
    joint_ids = _get_joint_ids_from_action_term(action_term)

    target_pos = action_term.processed_actions[env_id]
    joint_pos_abs = robot.data.joint_pos[env_id, joint_ids]
    joint_vel_abs = robot.data.joint_vel[env_id, joint_ids]
    computed_torque = robot.data.computed_torque[env_id, joint_ids]
    applied_torque = robot.data.applied_torque[env_id, joint_ids]
    root_pos = robot.data.root_pos_w[env_id] - base_env.scene.env_origins[env_id]
    root_quat = robot.data.root_quat_w[env_id]
    root_lin_vel_b = robot.data.root_lin_vel_b[env_id]
    root_ang_vel_b = robot.data.root_ang_vel_b[env_id]
    left_foot_force, right_foot_force = _get_foot_contact_forces(base_env, env_id)

    values = torch.cat(
        [
            target_pos.reshape(-1),
            joint_pos_abs.reshape(-1),
            joint_vel_abs.reshape(-1),
            computed_torque.reshape(-1),
            applied_torque.reshape(-1),
            root_pos.reshape(-1),
            root_quat.reshape(-1),
            root_lin_vel_b.reshape(-1),
            root_ang_vel_b.reshape(-1),
            left_foot_force.reshape(-1),
            right_foot_force.reshape(-1),
            torch.linalg.norm(left_foot_force).reshape(1),
            torch.linalg.norm(right_foot_force).reshape(1),
        ]
    )
    if values.numel() != len(_get_debug_header(action_dim)):
        raise RuntimeError(f"Unexpected debug value count: {values.numel()} for action_dim={action_dim}.")
    return values.detach().cpu().numpy()


def _write_trace_metadata(base_env, trace_path: str, trace_meta_path: str | None):
    """Save IsaacLab joint/contact ordering next to the CSV trace."""
    if trace_meta_path is None:
        trace_meta_path = f"{trace_path}.meta.txt"
    trace_meta_path = os.path.abspath(trace_meta_path)
    action_term = base_env.action_manager.get_term("joint_pos")
    robot = base_env.scene["robot"]

    lines = [
        "IsaacLab trace metadata",
        f"action_term_joint_names={getattr(action_term, '_joint_names', None)}",
        f"action_term_joint_ids={getattr(action_term, '_joint_ids', None)}",
        f"robot_data_joint_names={robot.data.joint_names}",
        f"robot_data_body_names={robot.data.body_names}",
    ]
    try:
        contact_sensor = base_env.scene["contact_forces"]
        lines.append(f"contact_sensor_body_names={contact_sensor.body_names}")
    except Exception as exc:
        lines.append(f"contact_sensor_body_names_error={exc}")

    with open(trace_meta_path, "w") as meta_file:
        meta_file.write("\n".join(lines))
        meta_file.write("\n")
    print(f"[INFO] Saved IsaacLab trace metadata to: {trace_meta_path}")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # handle deprecated configurations
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    # convert pre-5.0 published checkpoints to the layout expected by rsl-rl >= 5.0 (no-op otherwise)
    resume_path = handle_deprecated_rsl_rl_checkpoint(resume_path, installed_version)
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # export the trained policy to JIT and ONNX formats
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")

    if version.parse(installed_version) >= version.parse("4.0.0"):
        # use the new export functions for rsl-rl >= 4.0.0
        runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
        runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")
    else:
        # extract the neural network for rsl-rl < 4.0.0
        if version.parse(installed_version) >= version.parse("2.3.0"):
            policy_nn = runner.alg.policy
        else:
            policy_nn = runner.alg.actor_critic

        # extract the normalizer
        if hasattr(policy_nn, "actor_obs_normalizer"):
            normalizer = policy_nn.actor_obs_normalizer
        elif hasattr(policy_nn, "student_obs_normalizer"):
            normalizer = policy_nn.student_obs_normalizer
        else:
            normalizer = None

        # export to JIT and ONNX
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt
    trace_path = os.path.abspath(args_cli.trace_path)
    trace_dir = os.path.dirname(trace_path)
    if trace_dir:
        os.makedirs(trace_dir, exist_ok=True)

    _write_trace_metadata(env.unwrapped, trace_path, args_cli.trace_meta_path)
    trace_file = open(trace_path, "w", newline="")
    trace_writer = None

    # reset environment
    obs = env.get_observations()
    timestep = 0

    try:
        # simulate environment
        while simulation_app.is_running():
            start_time = time.time()
            # run everything in inference mode
            with torch.inference_mode():
                # agent stepping
                actions = policy(obs)
                policy_obs = _get_policy_observation_tensor(obs)

                if args_cli.trace_env_id >= policy_obs.shape[0]:
                    raise ValueError(
                        f"trace_env_id={args_cli.trace_env_id} is out of range for num_envs={policy_obs.shape[0]}."
                    )

                if timestep < args_cli.trace_steps:
                    obs_to_log = policy_obs[args_cli.trace_env_id].detach().cpu().numpy()
                    action_to_log = actions[args_cli.trace_env_id].detach().cpu().numpy()

                    if trace_writer is None:
                        trace_writer = csv.writer(trace_file)
                        trace_writer.writerow(
                            ["step", "time"]
                            + [f"obs_{i}" for i in range(obs_to_log.shape[0])]
                            + [f"action_{i}" for i in range(action_to_log.shape[0])]
                            + _get_debug_header(action_to_log.shape[0])
                        )

                # env stepping
                obs, _, dones, _ = env.step(actions)

                if timestep < args_cli.trace_steps:
                    debug_values = _get_debug_values(env.unwrapped, args_cli.trace_env_id, action_to_log.shape[0])
                    trace_writer.writerow(
                        [
                            timestep,
                            timestep * dt,
                            *obs_to_log.tolist(),
                            *action_to_log.tolist(),
                            *debug_values.tolist(),
                        ]
                    )
                    trace_file.flush()

                # reset recurrent states for episodes that have terminated
                if version.parse(installed_version) >= version.parse("4.0.0"):
                    policy.reset(dones)
                else:
                    policy_nn.reset(dones)

            timestep += 1

            if timestep >= args_cli.trace_steps:
                break

            if args_cli.video and timestep == args_cli.video_length:
                break

            # time delay for real-time evaluation
            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        trace_file.close()
        env.close()

    print(f"[INFO] Saved IsaacLab policy trace to: {trace_path}")


if __name__ == "__main__":
    try:
        # run the main function
        main()
    finally:
        # close sim app
        simulation_app.close()
