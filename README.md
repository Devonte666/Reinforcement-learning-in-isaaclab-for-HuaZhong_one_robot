@'
# MyRobot for Isaac Lab

自定义机器人 MyRobot 在 Isaac Lab 中的训练配置。

## 文件说明

| 路径 | 说明 |
|------|------|
| `isaaclab_tasks/.../my_robot/` | Velocity 任务配置（flat/rough + PPO） |
| `isaaclab_assets/data/Robots/MyRobot/` | USD 模型文件 + 配置 |
| `isaaclab_assets/robots/my_robot.py` | 机器人 ArticulationCfg 定义 |
| `isaaclab_tasks/.../velocity/config/__init__.py` | 修改：添加 `from . import my_robot` |
| `isaaclab_assets/robots/__init__.py` | 修改：添加 `my_robot` 导入 |

## 安装方式

把这些文件夹复制到 Isaac Lab 源码对应位置，覆盖官方文件：

```bash
# 1. 复制任务配置
cp -r isaaclab_tasks/.../my_robot/ <ISAACLAB>/source/isaaclab_tasks/.../velocity/config/

# 2. 复制 USD 模型
cp -r isaaclab_assets/data/Robots/MyRobot/ <ISAACLAB>/source/isaaclab_assets/data/Robots/

# 3. 复制机器人定义
cp isaaclab_assets/robots/my_robot.py <ISAACLAB>/source/isaaclab_assets/robots/

# 4. 覆盖修改过的官方文件（注意备份！）
cp isaaclab_tasks/.../velocity/config/__init__.py <ISAACLAB>/source/isaaclab_tasks/.../velocity/config/
cp isaaclab_assets/robots/__init__.py <ISAACLAB>/source/isaaclab_assets/robots/

# 5.训练命令

.\isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Velocity-Flat-MyRobot-v0 --headless