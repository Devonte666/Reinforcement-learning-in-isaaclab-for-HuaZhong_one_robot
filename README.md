# MyRobot for Isaac Lab

自定义机器人 MyRobot 在 Isaac Lab 中的训练配置。

## 📂 文件说明

| 路径 | 说明 |
|------|------|
| `isaaclab_tasks/.../my_robot/` | Velocity 任务配置（flat/rough + PPO） |
| `isaaclab_assets/data/Robots/MyRobot/` | USD 模型文件 + 配置 |
| `isaaclab_assets/robots/my_robot.py` | 机器人 ArticulationCfg 定义 |
| `isaaclab_tasks/.../velocity/config/__init__.py` | 修改：添加 `from . import my_robot` |
| `isaaclab_assets/robots/__init__.py` | 修改：添加 `my_robot` 导入 |

## 🛠️ 安装方式

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
```

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
