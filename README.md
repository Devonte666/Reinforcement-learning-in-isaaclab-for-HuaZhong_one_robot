# MyRobot for Isaac Lab
[录屏 2026年06月18日 18时23分50秒.webm](https://github.com/user-attachments/assets/be079582-ed04-4448-a441-844c02c66ea9)

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
```

## 🎮 播放/测试命令
```bash
# 加载训练好的模型进行测试
.\isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\play.py `
  --task Isaac-Velocity-Flat-MyRobot-v0
```
