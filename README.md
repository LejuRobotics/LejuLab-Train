# LejuRobot Lab

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![IsaacLab](https://img.shields.io/badge/IsaacLab-2.1-blue.svg)](https://github.com/isaac-sim/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![RSL-RL](https://img.shields.io/badge/RSL--RL-IsaacLab%202.1-green.svg)](https://github.com/leggedrobotics/rsl_rl)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![License](https://img.shields.io/badge/license-Apache--2.0-yellow.svg)](https://opensource.org/license/apache-2-0)

[English](#english) | [中文](#中文)

---

## English

### Overview

LejuRobot Lab is a robotics simulation and reinforcement learning framework built on top of Isaac Lab. It provides tools for motion imitation, training RL agents, and replaying motion data for humanoid robots such as RobanS14 and KuavoS52.

### Features

- **Multi-Robot Support**: Supports multiple robot models (RobanS14, KuavoS52, etc.)
- **Motion Imitation**: Train RL agents to imitate reference motions from NPZ files
- **Data Conversion**: Convert motion data between CSV and NPZ formats
- **Motion Replay**: Visualize and replay motion sequences in Isaac Sim
- **RL Training**: Train policies using RSL-RL framework
- **Flexible Configuration**: Support for local motion files and WandB registry

### Requirements

This framework is built on the following core dependencies with specific versions:

- **Python**: >= 3.10
- **Isaac Sim**: 4.5.0
- **Isaac Lab**: 2.1
- **RSL-RL**: Included with Isaac Lab 2.1 (via `isaaclab_rl` package)
- **PyTorch**: Compatible with Isaac Lab 2.1 requirements
- **CUDA**: Required for GPU acceleration (compatible with Isaac Sim 4.5.0)

**Note**: The RSL-RL library is integrated into Isaac Lab 2.1 as part of the `isaaclab_rl` package.

### Installation

1. **Install Isaac Lab** following the official documentation

2. **Install the package**:
   ```bash
   cd source/leju_robot
   pip install -e .
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt  # If available
   ```

4. **Configure IDE Type Checking** (Optional but Recommended):
   
   For proper IDE support (autocomplete, type checking) with VS Code/Pyright, you need to configure `extraPaths` in `pyproject.toml`:
   
   ```toml
   [tool.pyright]
   extraPaths = [
       "/path/to/IsaacLab2.1/source/isaaclab",
       "/path/to/IsaacLab2.1/source/isaaclab_assets",
       "/path/to/IsaacLab2.1/source/isaaclab_mimic",
       "/path/to/IsaacLab2.1/source/isaaclab_rl",
       "/path/to/IsaacLab2.1/source/isaaclab_tasks",
       "/path/to/IsaacLab2.1/IsaacLabExtensionRoban/source/ext_kuavo",
       "/path/to/isaac-sim-4.5/exts/omni.isaac.ml_archive/pip_prebundle",
   ]
   ```
   
   **Why this is needed:**
   - These paths point to Isaac Lab source packages that are not installed as standard Python packages
   - Pyright (VS Code's type checker) needs these paths to resolve imports and provide autocomplete
   - Without this configuration, you may see import errors and missing type hints in your IDE
   - **Important**: Update the paths to match your actual Isaac Lab installation directory
   
   **Note**: This configuration only affects IDE type checking and does not impact runtime execution. The actual imports work at runtime because Isaac Lab packages are added to `PYTHONPATH` during execution.

### Project Structure

```
LejuRobot_lab/
├── source/leju_robot/              # Main package
│   ├── leju_robot/                 # Core robot modules
│   │   ├── assets/                 # Robot asset definitions
│   │   ├── actuators/              # Actuator configurations
│   │   └── tasks/                  # Task definitions
│   │       ├── tracking/           # Motion tracking tasks (dance, standup)
│   │       │   ├── agents/         # RL agent configs
│   │       │   ├── mdp/            # MDP components
│   │       │   └── config/          # Robot-specific configs
│   │       │       ├── robanS14/    # RobanS14 configs (dance, standup)
│   │       │       └── kuavoS52/   # KuavoS52 configs (dance)
│   │       └── locomotion/         # Locomotion velocity tasks
│   │           └── velocity/        # Velocity control tasks
│   │               ├── agents/     # RL agent configs
│   │               ├── mdp/        # MDP components
│   │               └── config/      # Robot-specific configs
│   │                   ├── robanS14/ # RobanS14 velocity configs
│   │                   └── kuavoS52/ # KuavoS52 velocity configs
│   └── leju_data/                  # Robot data (URDF, meshes, etc.)
├── scripts/                        # Utility & training scripts
│   ├── motion_tool/                # Motion data tools
│   │   ├── csv_to_npz&deploycsv.py # CSV → NPZ & deploy-CSV converter
│   │   ├── pkl_to_npz&deploycsv.py # PKL → NPZ & deploy-CSV converter
│   │   ├── replay_npz.py           # Single NPZ replay
│   │   └── replay_npz_list.py      # Multiple NPZ replay
│   └── reinforcement_learning/     # RL training & play scripts
│       └── rsl_rl/                 # RSL-RL training scripts
│           ├── train.py            # Train policies (all task types)
│           └── play.py             # Play trained policies
└── docker/                         # Docker configuration
```

### Usage

#### 1. Convert CSV to NPZ

Convert motion data from CSV format to NPZ format:

```bash
python scripts/motion_tool/csv_to_npz&deploycsv.py \
    --input_file path/to/motion.csv \
    --input_fps 30 \
    --output_fps 50 \
    --robot robanS14 \
    --npz_output output/motion.npz \
    --csv_output output/motion_deploy.csv
```

**Parameters:**
- `--input_file`: Path to input CSV file (required)
- `--input_fps`: FPS of input motion (default: 30)
- `--output_fps`: FPS of output motion (default: 50)
- `--frame_range START END`: Optional frame range to extract
- `--npz_output`: Output NPZ file path
- `--csv_output`: Optional deploy CSV output path
- `--robot`: Robot model name (robanS14 or kuavoS52)

#### 2. Replay Motion (Single File)

Replay a single NPZ motion file:

```bash
python scripts/motion_tool/replay_npz.py \
    --motion_file path/to/motion.npz \
    --robot robanS14
```

**Parameters:**
- `--motion_file`: Path to NPZ motion file
- `--robot`: Robot model name (default: robanS14)

#### 3. Replay Motion (Multiple Files)

Replay multiple NPZ files in sequence:

```bash
python scripts/motion_tool/replay_npz_list.py \
    --motion_file path/to/motion1.npz \
    --robot robanS14
```

Or edit `MOTION_FILES` list in the script to specify multiple files.

#### 4. Train RL Agent

Train a reinforcement learning agent for different task types:

**Tracking Task (Motion Imitation):**
```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task Tracking-Dance-Flat-RobanS14 \
    --motion_file path/to/motion.npz \
    --num_envs 8192 \
    --headless \
    --max_iterations 25000
```

**Velocity Task (Locomotion):**
```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task Velocity-Flat-RobanS14 \
    --num_envs 8192 \
    --headless \
    --max_iterations 25000
```

**Parameters:**
- `--task`: Task name (e.g., `Tracking-Dance-Flat-RobanS14`, `Velocity-Flat-RobanS14`)
- `--motion_file`: Path to reference motion NPZ file (required for tracking tasks)
- `--num_envs`: Number of parallel environments
- `--max_iterations`: Maximum training iterations
- `--headless`: Run without GUI
- `--resume`: Resume training from checkpoint
- `--load_run`: Run ID to load checkpoint from
- `--checkpoint`: Checkpoint filename (e.g., `model_25000.pt`)

#### 5. Play Trained Policy

Test a trained policy:

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
    --task Tracking-Dance-Flat-RobanS14-Play \
    --load_run 2026-02-05_15-18-56 \
    --checkpoint model_52500.pt \
    --num_envs 1
```

**Parameters:**
- `--task`: Task name with `-Play` suffix
- `--load_run`: Run ID from training logs
- `--checkpoint`: Checkpoint filename
- `--num_envs`: Number of environments (typically 1 for visualization)

#### 6. Using VS Code Debug Configuration

The project includes pre-configured VS Code debug configurations in `.vscode/launch.json` for easy one-click training and testing.

**How to use:**

1. **Set Python Interpreter** (Important!):
   - Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac) to open the command palette
   - Type "Python: Select Interpreter" and select it
   - Choose the Python interpreter from your virtual environment (e.g., `venv/bin/python` or `conda envs/your_env/bin/python`)
   - Alternatively, click on the Python version in the bottom-right corner of VS Code and select the correct interpreter
   - **This step is required** - VS Code must use the same Python environment where Isaac Lab and project dependencies are installed

2. **Open VS Code** in the project root directory

3. **Go to Run and Debug**:
   - Press `F5` or click the "Run and Debug" icon in the sidebar
   - Or use the menu: `Run > Start Debugging`

4. **Select a configuration** from the dropdown at the top:
   - **Motion Tools:**
     - `csv to npz`: Convert CSV motion files to NPZ format
     - `pkl to npz`: Convert PKL motion files to NPZ format
     - `replay npz`: Replay a single NPZ motion file
     - `replay npz list`: Replay multiple NPZ motion files
   
   - **Training Configurations:**
     - `train robanS14 walk`: Train RobanS14 velocity control task
     - `train robanS14 dance`: Train RobanS14 dance tracking task
     - `train robanS14 standup`: Train RobanS14 standup tracking task
     - `train kuavoS52 walk`: Train KuavoS52 velocity control task
     - `train kuavoS52 dance`: Train KuavoS52 dance tracking task
   
   - **Play Configurations:**
     - `play robanS14 walk`: Test trained RobanS14 velocity policy
     - `play robanS14 dance`: Test trained RobanS14 dance policy
     - `play robanS14 standup`: Test trained RobanS14 standup policy
     - `play kuavoS52 walk`: Test trained KuavoS52 velocity policy
     - `play kuavoS52 dance`: Test trained KuavoS52 dance policy

5. **Customize parameters** (optional):
   - Edit `.vscode/launch.json` to modify arguments
   - Uncomment/comment lines to enable/disable options
   - Update `--load_run` and `--checkpoint` for play configurations

**Tips:**
- Set breakpoints in your code for debugging
- Use `--headless` flag for training without GUI (faster)
- Adjust `--num_envs` based on your GPU memory
- For play configurations, update `--load_run` with your training run ID

### Supported Robots

- **RobanS14**: 21-DOF humanoid robot
- **KuavoS52**: Humanoid robot

### Available Tasks

**Tracking Tasks (Motion Imitation):**
- `Tracking-Dance-Flat-RobanS14` / `Tracking-Dance-Flat-RobanS14-Play`
- `Tracking-Standup-Flat-RobanS14` / `Tracking-Standup-Flat-RobanS14-Play`
- `Tracking-Dance-Flat-KuavoS52` / `Tracking-Dance-Flat-KuavoS52-Play`

**Velocity Tasks (Locomotion):**
- `Velocity-Flat-RobanS14` / `Velocity-Flat-RobanS14-Play`
- `Velocity-Rough-RobanS14` / `Velocity-Rough-RobanS14-Play`
- `Velocity-Flat-KuavoS52` / `Velocity-Flat-KuavoS52-Play`
- `Velocity-Rough-KuavoS52` / `Velocity-Rough-KuavoS52-Play`

### Motion Data Format

NPZ files should contain the following arrays:
- `joint_pos`: Joint positions (T, num_joints)
- `joint_vel`: Joint velocities (T, num_joints)
- `body_pos_w`: Body positions in world frame (T, num_bodies, 3)
- `body_quat_w`: Body quaternions in world frame (T, num_bodies, 4)
- `body_lin_vel_w`: Body linear velocities (T, num_bodies, 3)
- `body_ang_vel_w`: Body angular velocities (T, num_bodies, 3)
- `fps`: Frame rate (scalar)

### Configuration

Robot configurations are defined in:
- `source/leju_robot/leju_robot/tasks/{task_type}/config/{robot_name}/`

Each robot has its own configuration including:
- Environment settings
- MDP components (observations, rewards, events, etc.)
- Agent configurations
- Task-specific parameters

### Docker Support

Docker configuration is available in the `docker/` directory for containerized deployment.

### License

Apache 2.0

### Contributing

Contributions are welcome! Please follow the project's coding standards and submit pull requests.

---

## 中文

### 概述

LejuRobot Lab 是一个基于 Isaac Lab 构建的机器人仿真和强化学习框架。它为 RobanS14 和 KuavoS52 等类人机器人提供动作模仿、训练 RL 智能体以及回放动作数据的工具。

### 功能特性

- **多机器人支持**：支持多种机器人模型（RobanS14、KuavoS52 等）
- **动作模仿**：训练 RL 智能体模仿来自 NPZ 文件的参考动作
- **数据转换**：在 CSV 和 NPZ 格式之间转换动作数据
- **动作回放**：在 Isaac Sim 中可视化和回放动作序列
- **RL 训练**：使用 RSL-RL 框架训练策略
- **灵活配置**：支持本地动作文件和 WandB 注册表

### 系统要求

本框架基于以下核心依赖的特定版本构建：

- **Python**: >= 3.10
- **Isaac Sim**: 4.5.0
- **Isaac Lab**: 2.1
- **RSL-RL**: 随 Isaac Lab 2.1 包含（通过 `isaaclab_rl` 包）
- **PyTorch**: 与 Isaac Lab 2.1 要求兼容
- **CUDA**: 需要用于 GPU 加速（与 Isaac Sim 4.5.0 兼容）

**注意**：RSL-RL 库已集成到 Isaac Lab 2.1 中，作为 `isaaclab_rl` 包的一部分。

### 安装

1. **安装 Isaac Lab**，按照官方文档操作

2. **安装包**：
   ```bash
   cd source/leju_robot
   pip install -e .
   ```

3. **安装依赖**：
   ```bash
   pip install -r requirements.txt  # 如果存在
   ```

4. **配置 IDE 类型检查**（可选但推荐）：
   
   为了在 VS Code/Pyright 中获得正确的 IDE 支持（自动补全、类型检查），您需要在 `pyproject.toml` 中配置 `extraPaths`：
   
   ```toml
   [tool.pyright]
   extraPaths = [
       "/path/to/IsaacLab2.1/source/isaaclab",
       "/path/to/IsaacLab2.1/source/isaaclab_assets",
       "/path/to/IsaacLab2.1/source/isaaclab_mimic",
       "/path/to/IsaacLab2.1/source/isaaclab_rl",
       "/path/to/IsaacLab2.1/source/isaaclab_tasks",
       "/path/to/IsaacLab2.1/IsaacLabExtensionRoban/source/ext_kuavo",
       "/path/to/isaac-sim-4.5/exts/omni.isaac.ml_archive/pip_prebundle",
   ]
   ```
   
   **为什么需要这个配置：**
   - 这些路径指向 Isaac Lab 的源代码包，它们不是作为标准 Python 包安装的
   - Pyright（VS Code 的类型检查器）需要这些路径来解析导入并提供自动补全
   - 没有此配置，您可能会在 IDE 中看到导入错误和缺少类型提示
   - **重要**：请更新路径以匹配您实际的 Isaac Lab 安装目录
   
   **注意**：此配置仅影响 IDE 类型检查，不会影响运行时执行。实际的导入在运行时可以正常工作，因为 Isaac Lab 包在执行时被添加到 `PYTHONPATH` 中。

### 项目结构

```
LejuRobot_lab/
├── source/leju_robot/                  # 主包
│   ├── leju_robot/                     # 核心机器人模块
│   │   ├── assets/                     # 机器人资源定义
│   │   ├── actuators/                  # 执行器配置
│   │   └── tasks/                      # 任务定义
│   │       ├── tracking/               # 动作跟踪任务（dance, standup）
│   │       │   ├── agents/             # RL 智能体配置
│   │       │   ├── mdp/                # MDP 组件
│   │       │   └── config/              # 机器人特定配置
│   │       │       ├── robanS14/        # RobanS14 配置（dance, standup）
│   │       │       └── kuavoS52/       # KuavoS52 配置（dance）
│   │       └── locomotion/             # 运动速度任务
│   │           └── velocity/            # 速度控制任务
│   │               ├── agents/         # RL 智能体配置
│   │               ├── mdp/            # MDP 组件
│   │               └── config/          # 机器人特定配置
│   │                   ├── robanS14/   # RobanS14 速度配置
│   │                   └── kuavoS52/   # KuavoS52 速度配置
│   └── leju_data/                      # 机器人数据（URDF、网格等）
├── scripts/                            # 工具脚本 & 训练脚本
│   ├── motion_tool/                    # 动作数据工具
│   │   ├── csv_to_npz&deploycsv.py     # CSV 转 NPZ & deploy-CSV 转换器
│   │   ├── pkl_to_npz&deploycsv.py     # PKL 转 NPZ & deploy-CSV 转换器
│   │   ├── replay_npz.py               # 单个 NPZ 回放
│   │   └── replay_npz_list.py          # 多个 NPZ 回放
│   └── reinforcement_learning/         # 强化学习训练 & 回放
│       └── rsl_rl/                     # RSL-RL 训练脚本
│           ├── train.py                # 训练策略（所有任务类型）
│           └── play.py                 # 运行训练好的策略
└── docker/                             # Docker 配置
```

### 使用方法

#### 1. CSV 转 NPZ

将动作数据从 CSV 格式转换为 NPZ 格式：

```bash
python scripts/motion_tool/csv_to_npz&deploycsv.py \
    --input_file path/to/motion.csv \
    --input_fps 30 \
    --output_fps 50 \
    --robot robanS14 \
    --npz_output output/motion.npz \
    --csv_output output/motion_deploy.csv
```

**参数说明：**
- `--input_file`: 输入 CSV 文件路径（必需）
- `--input_fps`: 输入动作的帧率（默认：30）
- `--output_fps`: 输出动作的帧率（默认：50）
- `--frame_range START END`: 可选，要提取的帧范围
- `--npz_output`: 输出 NPZ 文件路径
- `--csv_output`: 可选的部署 CSV 输出路径
- `--robot`: 机器人型号名称（robanS14 或 kuavoS52）

#### 2. 回放动作（单个文件）

回放单个 NPZ 动作文件：

```bash
python scripts/motion_tool/replay_npz.py \
    --motion_file path/to/motion.npz \
    --robot robanS14
```

**参数说明：**
- `--motion_file`: NPZ 动作文件路径
- `--robot`: 机器人型号名称（默认：robanS14）

#### 3. 回放动作（多个文件）

按顺序回放多个 NPZ 文件：

```bash
python scripts/motion_tool/replay_npz_list.py \
    --motion_file path/to/motion1.npz \
    --robot robanS14
```

或在脚本中编辑 `MOTION_FILES` 列表以指定多个文件。

#### 4. 训练 RL 智能体

训练强化学习智能体，支持不同类型的任务：

**跟踪任务（动作模仿）：**
```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task Tracking-Dance-Flat-RobanS14 \
    --motion_file path/to/motion.npz \
    --num_envs 8192 \
    --headless \
    --max_iterations 25000
```

**速度任务（运动控制）：**
```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task Velocity-Flat-RobanS14 \
    --num_envs 8192 \
    --headless \
    --max_iterations 25000
```

**参数说明：**
- `--task`: 任务名称（如 `Tracking-Dance-Flat-RobanS14`、`Velocity-Flat-RobanS14`）
- `--motion_file`: 参考动作 NPZ 文件路径（跟踪任务必需）
- `--num_envs`: 并行环境数量
- `--max_iterations`: 最大训练迭代次数
- `--headless`: 无 GUI 运行
- `--resume`: 从检查点恢复训练
- `--load_run`: 要加载的检查点的运行 ID
- `--checkpoint`: 检查点文件名（如 `model_25000.pt`）

#### 5. 运行训练好的策略

测试训练好的策略：

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
    --task Tracking-Dance-Flat-RobanS14-Play \
    --load_run 2026-02-05_15-18-56 \
    --checkpoint model_52500.pt \
    --num_envs 1
```

**参数说明：**
- `--task`: 任务名称，需带 `-Play` 后缀
- `--load_run`: 训练日志中的运行 ID
- `--checkpoint`: 检查点文件名
- `--num_envs`: 环境数量（通常为 1 用于可视化）

#### 6. 使用 VS Code Debug 配置

项目在 `.vscode/launch.json` 中包含了预配置的 VS Code 调试配置，可以一键启动训练和测试任务。

**使用方法：**

1. **设置 Python 解释器**（重要！）：
   - 按 `Ctrl+Shift+P`（Mac 上为 `Cmd+Shift+P`）打开命令面板
   - 输入 "Python: Select Interpreter" 并选择
   - 选择虚拟环境中的 Python 解释器（例如 `venv/bin/python` 或 `conda envs/your_env/bin/python`）
   - 或者点击 VS Code 右下角的 Python 版本，选择正确的解释器
   - **此步骤是必需的** - VS Code 必须使用安装了 Isaac Lab 和项目依赖的相同 Python 环境

2. **在 VS Code 中打开项目**：
   - 在项目根目录打开 VS Code

3. **进入运行和调试**：
   - 按 `F5` 或点击侧边栏的"运行和调试"图标
   - 或使用菜单：`运行 > 启动调试`

4. **从顶部下拉菜单中选择配置**：
   - **动作工具：**
     - `csv to npz`: 将 CSV 动作文件转换为 NPZ 格式
     - `pkl to npz`: 将 PKL 动作文件转换为 NPZ 格式
     - `replay npz`: 回放单个 NPZ 动作文件
     - `replay npz list`: 回放多个 NPZ 动作文件
   
   - **训练配置：**
     - `train robanS14 walk`: 训练 RobanS14 速度控制任务
     - `train robanS14 dance`: 训练 RobanS14 舞蹈跟踪任务
     - `train robanS14 standup`: 训练 RobanS14 站立跟踪任务
     - `train kuavoS52 walk`: 训练 KuavoS52 速度控制任务
     - `train kuavoS52 dance`: 训练 KuavoS52 舞蹈跟踪任务
   
   - **运行配置：**
     - `play robanS14 walk`: 测试训练好的 RobanS14 速度策略
     - `play robanS14 dance`: 测试训练好的 RobanS14 舞蹈策略
     - `play robanS14 standup`: 测试训练好的 RobanS14 站立策略
     - `play kuavoS52 walk`: 测试训练好的 KuavoS52 速度策略
     - `play kuavoS52 dance`: 测试训练好的 KuavoS52 舞蹈策略

5. **自定义参数**（可选）：
   - 编辑 `.vscode/launch.json` 以修改参数
   - 取消注释/注释行以启用/禁用选项
   - 更新运行配置中的 `--load_run` 和 `--checkpoint`

**提示：**
- 在代码中设置断点进行调试
- 训练时使用 `--headless` 标志以无 GUI 模式运行（更快）
- 根据 GPU 内存调整 `--num_envs`
- 对于运行配置，使用您的训练运行 ID 更新 `--load_run`

### 支持的机器人

- **RobanS14**：21 自由度类人机器人
- **KuavoS52**：类人机器人

### 可用任务

**跟踪任务（动作模仿）：**
- `Tracking-Dance-Flat-RobanS14` / `Tracking-Dance-Flat-RobanS14-Play`
- `Tracking-Standup-Flat-RobanS14` / `Tracking-Standup-Flat-RobanS14-Play`
- `Tracking-Dance-Flat-KuavoS52` / `Tracking-Dance-Flat-KuavoS52-Play`

**速度任务（运动控制）：**
- `Velocity-Flat-RobanS14` / `Velocity-Flat-RobanS14-Play`
- `Velocity-Rough-RobanS14` / `Velocity-Rough-RobanS14-Play`
- `Velocity-Flat-KuavoS52` / `Velocity-Flat-KuavoS52-Play`
- `Velocity-Rough-KuavoS52` / `Velocity-Rough-KuavoS52-Play`

### 动作数据格式

NPZ 文件应包含以下数组：
- `joint_pos`: 关节位置 (T, num_joints)
- `joint_vel`: 关节速度 (T, num_joints)
- `body_pos_w`: 世界坐标系中的身体位置 (T, num_bodies, 3)
- `body_quat_w`: 世界坐标系中的身体四元数 (T, num_bodies, 4)
- `body_lin_vel_w`: 身体线速度 (T, num_bodies, 3)
- `body_ang_vel_w`: 身体角速度 (T, num_bodies, 3)
- `fps`: 帧率（标量）

### 配置

机器人配置定义在：
- `source/leju_robot/leju_robot/tasks/{task_type}/config/{robot_name}/`

每个机器人都有自己的配置，包括：
- 环境设置
- MDP 组件（观测、奖励、事件等）
- 智能体配置
- 任务特定参数

### Docker 支持

`docker/` 目录中提供了 Docker 配置，用于容器化部署。

### 许可证

Apache 2.0

### 贡献

欢迎贡献！请遵循项目的编码标准并提交 Pull Request。

