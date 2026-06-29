# MyRobot for Isaac Lab
[录屏 2026年06月18日 18时23分50秒.webm](https://github.com/user-attachments/assets/a75d2fca-7026-44c0-ae31-bf5659f3504a)


自定义机器人 MyRobot 在 Isaac Lab 中的训练配置。

## 📂 文件说明

| 路径 | 说明 |
|------|------|
| `Huazhong1/urdf/Huazhong1.urdf` | 机器人 URDF，当前版本固定了左右髋 yaw 关节，并调整了部分关节限位和碰撞体 |
| `isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/my_robot/` | Velocity 任务配置（flat/rough + PPO） |
| `isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/rewards.py` | 自定义 locomotion reward / penalty 函数 |
| `isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/__init__.py` | 导出自定义 MDP 函数，使 `mdp.xxx` 可以在配置中调用 |
| `isaaclab_assets/isaaclab_assets/data/Robots/MyRobot/` | USD 模型文件 + 配置 |
| `isaaclab_assets/isaaclab_assets/robots/my_robot.py` | 机器人 ArticulationCfg 定义 |
| `isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/__init__.py` | 修改：进入 velocity 任务发现层并加载 config |
| `isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/__init__.py` | 修改：添加 `from . import my_robot` |
| `isaaclab_assets/isaaclab_assets/robots/__init__.py` | 修改：添加 `my_robot` 导入 |
| `MuJoCo/` | IsaacLab 训练策略迁移到 MuJoCo 的 sim-to-sim 验证脚本 |
| `scripts/reinforcement_learning/rsl_rl/play_trace.py` | IsaacLab 侧 trace 导出脚本，用于和 MuJoCo 逐项对比 |

## 🛠️ 安装方式

把这些文件夹复制到 Isaac Lab 源码对应位置，覆盖官方文件：

```bash
ISAACLAB=/home/user/Devonte_file/IsaacLab

# 1. 复制任务配置
cp -r isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/my_robot \
  $ISAACLAB/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/

# 2. 复制自定义 MDP reward 函数
cp -r isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp \
  $ISAACLAB/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/

# 3. 复制 USD 模型
cp -r isaaclab_assets/isaaclab_assets/data/Robots/MyRobot \
  $ISAACLAB/source/isaaclab_assets/isaaclab_assets/data/Robots/

# 4. 复制机器人定义
cp isaaclab_assets/isaaclab_assets/robots/my_robot.py \
  $ISAACLAB/source/isaaclab_assets/isaaclab_assets/robots/

# 5. 覆盖修改过的官方入口文件（注意备份）
cp isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/__init__.py \
  $ISAACLAB/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/
cp isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/__init__.py \
  $ISAACLAB/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/
cp isaaclab_assets/isaaclab_assets/robots/__init__.py \
  $ISAACLAB/source/isaaclab_assets/isaaclab_assets/robots/

# 6. 可选：复制 sim-to-sim trace 调试脚本
cp scripts/reinforcement_learning/rsl_rl/play_trace.py \
  $ISAACLAB/scripts/reinforcement_learning/rsl_rl/
```

## 奖励与惩罚设计

`MyRobotFlatEnvCfg` 继承自 `MyRobotRoughEnvCfg`，所以 rough 中定义的奖励项默认也会在 flat 环境里生效，除非在 flat 配置中显式修改或关闭。

| 名称 | 权重 / 状态 | 作用 |
|------|-------------|------|
| `termination_penalty` | `-200.0` | 机器人触发终止条件时给较大惩罚，降低摔倒或非法接触策略的收益 |
| `track_lin_vel_xy_exp` | `1.0` | 使用 yaw frame 下的线速度误差指数核，鼓励机器人跟踪前进速度命令 |
| `track_ang_vel_z_exp` | rough: `2.0`, flat: `1.0` | 使用 world frame 下的 yaw 角速度误差指数核，鼓励跟踪转向命令 |
| `feet_air_time` | rough: `0.25`, flat: `0.75` | 使用 `feet_air_time_positive_biped`，鼓励双足机器人形成单脚支撑和合理抬脚时间 |
| `feet_slide` | `-0.1` | 脚接触地面时，根据脚部水平速度惩罚打滑 |
| `feet_air_time_asymmetry` | `-0.6` | 根据左右脚当前腾空时间差的平方惩罚不对称步态，鼓励左右脚交替更均衡 |
| `dof_pos_limits` | `-1.0` | 对脚踝 pitch / roll 关节接近或超过关节位置限制的行为进行惩罚 |
| `joint_deviation_hip` | `-0.1` | 惩罚髋 roll 关节偏离默认姿态 |
| `joint_deviation_knee` | `-0.1` | 惩罚膝 pitch 关节偏离默认姿态 |
| `flat_orientation_l2` | `-1.0` | 惩罚机身姿态倾斜，鼓励 base 保持稳定 |
| `action_rate_l2` | `-0.005` | 惩罚动作变化过快，使控制输出更平滑 |
| `dof_acc_l2` | rough: `-1.25e-6`, flat: `-1.0e-7` | 惩罚腿部、膝部、脚踝关节加速度，减少高频抖动 |
| `dof_torques_l2` | rough: `-1.5e-6`, flat: `-2.0e-6` | 惩罚腿部、膝部、脚踝关节力矩，降低能耗和过大控制力 |
| `lin_vel_z_l2` | rough: `0.0`, flat: `-0.2` | flat 环境中惩罚 base 垂直方向速度，减少上下跳动 |

自定义 `rewards.py` 中还加入了这些辅助函数：

- `track_lin_vel_xy_yaw_frame_exp`：在机器人 yaw 对齐坐标系下计算线速度跟踪奖励。
- `track_ang_vel_z_world_exp`：在世界坐标系下计算 yaw 角速度跟踪奖励。
- `feet_air_time_asymmetry_penalty`：计算左右脚腾空时间差，作为步态不对称惩罚。
- `feet_slide`：在原脚底滑动惩罚基础上，额外打印每个 episode 的脚部接触力统计，包括平均力、峰值、95 分位数和接触次数。
- `knee_pitch_soft_limit`：膝关节软限位惩罚函数已经实现，但当前在 `rough_env_cfg.py` 中还是注释状态，需要时可以替代或补充 `joint_deviation_knee`。

当前速度命令范围主要用于训练直行：`lin_vel_x=(0.0, 1.0)`，`lin_vel_y=(0.0, 0.0)`，`ang_vel_z=(0.0, 0.0)`。

## 🔄 导出 USD 模型（从 URDF）
如果你的机器人是 URDF 格式，需要先转换为 USD：
```bash
.\isaaclab.bat -p scripts\tools\convert_urdf.py `
  D:\Devonte_file\backup-humanoid-gym-ourrobot\resources\robots\Huazhong1\urdf\Huazhong1.urdf `
  source\isaaclab_assets\data\Robots\MyRobot\my_robot.usd `
  --merge-joints `
  --joint-stiffness 0.0 `
  --joint-damping 0.0 `
  --joint-target-type none
```

参数说明：
--merge-joints：合并固定关节
--joint-stiffness 0.0：关节刚度设为 0
--joint-damping 0.0：关节阻尼设为 0
--joint-target-type none：不设置关节目标类型

## 🚀 训练命令

```bash
# 训练（2048 个并行环境，无头模式）
.\isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\train.py `
  --task Isaac-Velocity-Flat-MyRobot-v0 `
  --num_envs 2048 `
  --headless

python scripts/rsl_rl/train.py \
  --task HuaZhong-Velocity-Flat-MyRobot-v2 \
  --num_envs 4096 \
  --device cuda:0 \
  --headless \
  --max_iterations 1500

  --run_name standing_reward_test
```

## 🎮 播放/测试命令
```bash
# 加载训练好的模型进行测试
.\isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\play.py `
  --task Isaac-Velocity-Flat-MyRobot-v0

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Velocity-Flat-MyRobot-v0
```


## MuJoCo Sim-to-Sim 验证

本仓库同时包含 IsaacLab -> MuJoCo 的 sim-to-sim 验证代码，位于 `MuJoCo/`。

### 文件说明

| 路径 | 说明 |
|------|------|
| `MuJoCo/run_policy_mujoco.py` | 在 MuJoCo 中加载 IsaacLab 导出的 ONNX policy 并运行机器人 |
| `MuJoCo/compare_traces.py` | 对比 IsaacLab trace 和 MuJoCo trace，定位 observation/action/dynamics 差异 |
| `MuJoCo/convert_stl.py` | 辅助转换 STL mesh |
| `MuJoCo/screenshot.png` | MuJoCo 当前模型显示参考图 |
| `Huazhong1/urdf/Huazhong1.xml` | 由 URDF 转换得到的 MJCF 模型 |

### 运行方式

先导出 IsaacLab policy 为 ONNX，然后在 MuJoCo 环境中运行：

```bash
conda activate mujoco_rl
cd /home/user/Devonte_file/Reinforcement-learning-in-isaaclab-for-HuaZhong_one_robot/MuJoCo

python run_policy_mujoco.py \
  --mjcf_path /home/user/Devonte_file/Reinforcement-learning-in-isaaclab-for-HuaZhong_one_robot/Huazhong1/urdf/Huazhong1.xml \
  --onnx_path /home/user/Devonte_file/IsaacLab/logs/rsl_rl/my_robot_flat/2026-06-18_15-46-24/exported/policy.onnx \
  --command_vel 1.0 0.0 0.0
```

当前验证通过的 policy 输入为 42 维：

```text
base_lin_vel       3
base_ang_vel       3
projected_gravity  3
velocity_commands  3
joint_pos          10
joint_vel          10
last_action        10
```

policy 输出不是直接力矩，而是 10 维关节位置 action：

```text
target_joint_pos = default_joint_pos + 0.5 * action
```

底层再由 PD/DCMotor actuator 根据 `kp/kd` 计算关节力矩。

### 本次迁移做了什么

1. 将 `Huazhong1.urdf` 转换为 MuJoCo MJCF：`Huazhong1/urdf/Huazhong1.xml`。
2. 编写 `run_policy_mujoco.py`，完成 MJCF 加载、ONNX policy 加载、42 维 observation 构造、action 执行和 viewer/headless 运行。
3. 编写 `play_trace.py` 和 `compare_traces.py`，用于把 IsaacLab 前几秒的 observation/action/dynamics 导出并和 MuJoCo 逐项对比。
4. 修正 MuJoCo 中的关节顺序，使其与 IsaacLab action term 顺序一致。
5. 增加 actuator 到 joint 的显式映射，避免 MuJoCo XML actuator 顺序和 policy joint 顺序不一致。
6. 修正默认关节角、action scale、PD 参数、control dt，使 MuJoCo 执行逻辑和 IsaacLab 对齐。
7. 在 MJCF 里补充/调整 armature 和可视化 mesh，使模型显示和动力学表现更接近 IsaacLab。

### 遇到的问题和解决方式

| 问题 | 原因 | 解决方式 |
|------|------|----------|
| MuJoCo 中机器人很快倒地 | policy action 顺序和 MuJoCo joint 顺序不一致 | 使用 IsaacLab trace metadata 确认真实 joint order，并重排 `JOINT_NAMES` |
| action/target 对不上 | MuJoCo actuator 顺序不能假设等于 joint 顺序 | 根据 `model.actuator_trnid` 建立 actuator 到 joint 的映射 |
| 模型只显示碰撞块或视觉效果和 URDF 不同 | MJCF 转换后 mesh/visual 路径和 MuJoCo 显示方式不同 | 检查 MJCF mesh 路径，补充 `base_link.obj` 并保留必要 visual/collision |
| Ctrl+C 不容易退出 viewer | MuJoCo viewer 主循环没有处理 stop signal | 在脚本中加入 `SIGINT/SIGTERM` stop flag 和 clean shutdown |
| 运行结束后出现 `GLXBadContext` | viewer 关闭时 OpenGL context 释放警告 | 仿真已完成后出现，不影响结果 |
| `fetch first` 导致 Git push 被拒绝 | GitHub 远端已有本地没有的新提交 | 先 `git fetch origin`，再 `git rebase origin/main`，最后重新 push |

### Trace 对比流程

在 IsaacLab 里导出 trace：

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

在 MuJoCo 中重放 IsaacLab action：

```bash
conda activate mujoco_rl
cd /home/user/Devonte_file/Reinforcement-learning-in-isaaclab-for-HuaZhong_one_robot/MuJoCo

python run_policy_mujoco.py \
  --headless \
  --control_mode position \
  --replay_trace /home/user/Devonte_file/isaaclab_dynamics_trace.csv \
  --trace_path /home/user/Devonte_file/mujoco_position_replay_trace.csv \
  --trace_steps 100 \
  --duration 2.1
```

对比两边 trace：

```bash
python compare_traces.py \
  --isaac /home/user/Devonte_file/isaaclab_dynamics_trace.csv \
  --mujoco /home/user/Devonte_file/mujoco_position_replay_trace.csv
```

播放策略命令

```bash
conda activate mujoco_rl
cd /home/user/Devonte_file/MuJoCo

python run_policy_mujoco.py

```

### 后续迁移注意事项

- observation 顺序必须和训练时完全一致，尤其是 `joint_pos`、`joint_vel`、`last_action` 的关节顺序。
- action 顺序必须和 IsaacLab action term 顺序一致，不能只按 URDF/MJCF 文件里的顺序猜。
- `default_joint_pos`、`ACTION_SCALE`、`CONTROL_DT`、`kp/kd`、effort limit 要和 IsaacLab 配置保持一致。
- `velocity_commands` 是目标速度命令，不是机器人当前速度。
- `projected_gravity` 应由 IMU 姿态估计得到，用来表示机身相对重力方向的姿态。
- 当前 policy 不使用脚踝六维力传感器；如果未来加入双脚 6D wrench，输入会从 42 维变为 54 维，需要重新训练或 fine-tune。
- 真机部署时应优先保证零位、关节方向、单位、坐标系、延迟和滤波与仿真一致。
- MuJoCo 中能跑通只说明 sim-to-sim 链路成立，真机部署仍需要低速、限幅、急停和安全绳等保护。
