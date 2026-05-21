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

LejuRobot Lab is a robotics simulation and reinforcement learning framework built on top of Isaac Lab. It provides tools for motion imitation, training RL agents, and replaying motion data for humanoid robots such as RobanS14, RobanS17, KuavoS53, and KuavoS54.

### Features

- **Multi-Robot Support**: Supports multiple robot models (RobanS14, RobanS17, KuavoS53, KuavoS54, KuavoS46, etc.)
- **Motion Imitation**: Train RL agents to imitate reference motions from NPZ files
- **Data Conversion**: Convert motion data between CSV, PKL, and NPZ formats
- **Motion Replay**: Visualize and replay motion sequences in Isaac Sim
- **RL Training**: Train policies using RSL-RL and SKRL frameworks
- **Flexible Configuration**: Support for local motion files and WandB registry

### Requirements

This framework is built on the following core dependencies with specific versions:

- **Python**: >= 3.10
- **Isaac Sim**: 4.5.0
- **Isaac Lab**: 2.1
- **RSL-RL**: Included with Isaac Lab 2.1 (via `isaaclab_rl` package)
- **SKRL**: Customized based on version 1.4.3
- **PyTorch**: Compatible with Isaac Lab 2.1 requirements
- **CUDA**: Required for GPU acceleration (compatible with Isaac Sim 4.5.0)

**Note**: The RSL-RL library is integrated into Isaac Lab 2.1 as part of the `isaaclab_rl` package. The SKRL library is custom-modified for AMP tasks in this project, rather than using the standard upstream distribution.

### Installation

1. **Install Isaac Lab** following the official documentation

2. **Install the package**:
   ```bash
   cd source/leju_robot
   pip install -e .
   ```

3. **Install dependencies**:
   ```bash
   # Run this command from the repository root directory
   pip install -e source/leju_robot/leju_robot/tasks/amp/skrl  # Local customized SKRL
   ```

   There is no separate `requirements.txt` in this repository; runtime dependencies come from Isaac Lab and the editable installs above.

4. **Configure IDE Type Checking** (Optional but Recommended):

   The repository does not ship with Pyright paths pre-configured. For proper IDE support (autocomplete, type checking) with VS Code/Pyright, add `extraPaths` in the project root `pyproject.toml` or in `.vscode/settings.json` under `python.analysis.extraPaths`:

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
lejurobot_lab/
├── source/leju_robot/              # Main package
│   └── leju_robot/                 # Core robot modules
│       ├── assets/
│       │   ├── robots/             # URDF / USD robot models
│       │   └── motion_data/mimic/  # Example csv / npz / deploy data
│       ├── actuators/              # Actuator configurations
│       └── tasks/                  # Task definitions
│           ├── amp/                # AMP base task modules
│           │   ├── envs/           # AMP manager-based environment
│           │   └── skrl/           # Customized SKRL (AMP)
│           ├── tracking/           # Motion tracking (dance, standup, etc.)
│           │   ├── agents/
│           │   ├── mdp/
│           │   └── config/
│           │       ├── robanS14/   # dance, standup, new_year_dance
│           │       ├── robanS17/   # dance
│           │       ├── kuavoS53/   # dance
│           │       └── kuavoS54/   # dance
│           └── locomotion/
│               ├── velocity/       # RSL-RL velocity tasks
│               │   └── config/
│               │       ├── robanS14/
│               │       ├── kuavoS53/
│               │       ├── kuavoS54/
│               │       └── kuavoS46/  # rough only
│               └── velocity_amp/   # SKRL AMP velocity
│                   └── config/robanS17/
├── scripts/
│   ├── list_envs.py                # List registered tasks
│   ├── motion_tool/
│   │   ├── csv_to_npz&deploycsv.py
│   │   ├── pkl_to_npz&deploycsv.py
│   │   ├── replay_npz.py
│   │   └── replay_npz_list.py
│   └── reinforcement_learning/
│       ├── skrl/                   # SKRL train / play / export
│       └── rsl_rl/                 # RSL-RL train / play
└── .vscode/                        # launch.json debug configs
```

### Usage

#### 1. Convert CSV to NPZ

Convert motion data from CSV format to NPZ format:

```bash
python "scripts/motion_tool/csv_to_npz&deploycsv.py" \
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
- `--robot`: Robot model name (`robanS14`, `robanS17`, `kuavoS53`, `kuavoS54`)
- `--input_quat_xyzw`: If set, input CSV quaternions are in xyzw order and will be converted to wxyz

#### 2. Replay Motion (Single File)

Replay a single NPZ motion file:

```bash
python scripts/motion_tool/replay_npz.py \
    --motion_file path/to/motion.npz \
    --robot robanS14
```

**Parameters:**
- `--motion_file`: Path to NPZ motion file
- `--robot`: Robot model name (`robanS14`, `robanS17`, `kuavoS53`, `kuavoS54`; default: `robanS14`)

#### 3. Replay Motion (Multiple Files)

Replay multiple NPZ files in sequence:

```bash
python scripts/motion_tool/replay_npz_list.py \
    --motion_file path/to/motion1.npz \
    --robot robanS14
```

Or edit the `MOTION_FILES` list in the script to specify multiple files.

#### 4. List Registered Environments

Print all registered `Tracking-*` and `Velocity-*` tasks:

```bash
python scripts/list_envs.py
```

#### 5. Train RL Agent

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

**Tracking Task (New Year Dance, RobanS14):**
```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task Tracking-New-Year-Dance-Flat-RobanS14 \
    --motion_file path/to/motion.npz \
    --num_envs 8192 \
    --headless \
    --max_iterations 25000
```

**Tracking Task (RobanS17):**
```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task Tracking-Dance-Flat-RobanS17 \
    --motion_file path/to/motion.npz \
    --num_envs 8192 \
    --headless \
    --max_iterations 25000
```

**Velocity Task (Locomotion, RSL-RL):**
```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task Velocity-Flat-RobanS14 \
    --num_envs 8192 \
    --headless \
    --max_iterations 25000
```

**Velocity Task (Locomotion with AMP, SKRL):**
```bash
python scripts/reinforcement_learning/skrl/train.py \
    --task Velocity-AMP-RobanS17 \
    --num_envs 4096 \
    --algorithm AMP \
    --max_iterations 30000 \
    --headless
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
- `--seed`: Random seed (optional)
- `--logger`: Logger backend for RSL-RL (`wandb`, `tensorboard`, or `neptune`)

#### 6. Play Trained Policy

Test a trained RSL-RL policy:

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
    --task Tracking-Dance-Flat-RobanS14-Play \
    --motion_file path/to/motion.npz \
    --load_run 2026-02-05_15-18-56 \
    --checkpoint model_52500.pt \
    --num_envs 1
```

Test a trained SKRL Velocity AMP policy (use absolute path or path under `logs/skrl/.../checkpoints/`):

```bash
python scripts/reinforcement_learning/skrl/play.py \
    --task Velocity-AMP-RobanS17-Play \
    --num_envs 32 \
    --algorithm AMP \
    --checkpoint /path/to/logs/skrl/.../checkpoints/agent_24000.pt \
    --output_name agent_24000.onnx
```

SKRL checkpoints are saved as `agent_{timestep}.pt` under the experiment `checkpoints/` directory.

**Parameters:**
- `--task`: Task name with `-Play` suffix
- `--load_run`: Run ID from training logs (RSL-RL)
- `--checkpoint`: Checkpoint file path or filename
- `--num_envs`: Number of environments (typically 1 for visualization)

#### 7. Using VS Code Debug Configuration

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
     - `train robanS14 walk` / `train robanS14 dance` / `train robanS14 standup`
     - `train robanS17 dance` / `train robanS17 AMP walk`
     - `train kuavoS53 walk` / `train kuavoS53 dance`
     - `train kuavoS54 walk` / `train kuavoS54 dance`
     - `train kuavoS46 rough walk`

   - **Play Configurations:**
     - `play robanS14 walk` / `play robanS14 dance` / `play robanS14 standup`
     - `play robanS17 dance` / `play robanS17 AMP walk`
     - `play kuavoS53 walk` / `play kuavoS53 dance`
     - `play kuavoS54 walk` / `play kuavoS54 dance`
     - `play kuavoS46 rough walk`

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
- **RobanS17**: Humanoid robot (tracking dance, AMP velocity)
- **KuavoS53**: Humanoid robot
- **KuavoS54**: Humanoid robot
- **KuavoS46**: Humanoid robot (rough-terrain velocity only; robot assets under `assets/robots/kuavoS46/` are not included in this repository—provide them locally before use)

> **Note:** `KuavoS52` is no longer used in this codebase. Use `KuavoS53` or `KuavoS54` instead.

### Available Tasks

**Tracking Tasks (Motion Imitation, RSL-RL):**
- `Tracking-Dance-Flat-RobanS14` / `Tracking-Dance-Flat-RobanS14-Play`
- `Tracking-New-Year-Dance-Flat-RobanS14` / `Tracking-New-Year-Dance-Flat-RobanS14-Play`
- `Tracking-Standup-Flat-RobanS14` / `Tracking-Standup-Flat-RobanS14-Play`
- `Tracking-Dance-Flat-RobanS17` / `Tracking-Dance-Flat-RobanS17-Play`
- `Tracking-Dance-Flat-KuavoS53` / `Tracking-Dance-Flat-KuavoS53-Play`
- `Tracking-Dance-Flat-KuavoS54` / `Tracking-Dance-Flat-KuavoS54-Play`

**Velocity Tasks (Locomotion, RSL-RL):**
- `Velocity-Flat-RobanS14` / `Velocity-Flat-RobanS14-Play`
- `Velocity-Rough-RobanS14` / `Velocity-Rough-RobanS14-Play`
- `Velocity-Flat-KuavoS53` / `Velocity-Flat-KuavoS53-Play`
- `Velocity-Rough-KuavoS53` / `Velocity-Rough-KuavoS53-Play`
- `Velocity-Flat-KuavoS54` / `Velocity-Flat-KuavoS54-Play`
- `Velocity-Rough-KuavoS54` / `Velocity-Rough-KuavoS54-Play`
- `Velocity-Rough-KuavoS46` / `Velocity-Rough-KuavoS46-Play`

**Velocity AMP Tasks (SKRL):**
- `Velocity-AMP-RobanS17` / `Velocity-AMP-RobanS17-Play`

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

Robot configurations are organized as follows:

- **Tracking:** `source/leju_robot/leju_robot/tasks/tracking/config/<robot>/<variant>/` (e.g., `dance`, `standup`, `new_year_dance`)
- **Velocity:** `source/leju_robot/leju_robot/tasks/locomotion/velocity/config/<robot>/`
- **Velocity AMP:** `source/leju_robot/leju_robot/tasks/locomotion/velocity_amp/config/robanS17/`

Each configuration includes environment settings, MDP components (observations, rewards, events, etc.), agent settings, and task-specific parameters.

### License

Apache 2.0

### Contributing

Contributions are welcome! Please follow the project's coding standards and submit pull requests.

---

## 中文

### 概述

LejuRobot Lab 是一个基于 Isaac Lab 构建的机器人仿真和强化学习框架。它为 RobanS14、RobanS17、KuavoS53、KuavoS54 等人形机器人提供动作模仿、训练 RL 智能体以及回放动作数据的工具。

### 功能特性

- **多机器人支持**：支持多种机器人模型（RobanS14、RobanS17、KuavoS53、KuavoS54、KuavoS46 等）
- **动作模仿**：训练 RL 智能体模仿来自 NPZ 文件的参考动作
- **数据转换**：在 CSV、PKL 和 NPZ 格式之间转换动作数据
- **动作回放**：在 Isaac Sim 中可视化和回放动作序列
- **RL 训练**：使用 RSL-RL、SKRL 等框架训练策略
- **灵活配置**：支持本地动作文件和 WandB 注册表

### 系统要求

本框架基于以下核心依赖的特定版本构建：

- **Python**: >= 3.10
- **Isaac Sim**: 4.5.0
- **Isaac Lab**: 2.1
- **RSL-RL**: 随 Isaac Lab 2.1 包含（通过 `isaaclab_rl` 包）
- **SKRL**: 在 1.4.3 版本基础上的本地定制版
- **PyTorch**: 与 Isaac Lab 2.1 要求兼容
- **CUDA**: 需要用于 GPU 加速（与 Isaac Sim 4.5.0 兼容）

**注意**：RSL-RL 库已集成到 Isaac Lab 2.1 中，作为 `isaaclab_rl` 包的一部分。SKRL 库针对本项目中的 AMP 任务进行了定制化修改，不使用一般的发行版。

### 安装

1. **安装 Isaac Lab**，按照官方文档操作

2. **安装包**：
   ```bash
   cd source/leju_robot
   pip install -e .
   ```

3. **安装依赖**：
   ```bash
   # 请在仓库根目录执行以下命令
   pip install -e source/leju_robot/leju_robot/tasks/amp/skrl  # 安装本地定制版 SKRL
   ```

   本仓库无独立 `requirements.txt`，运行时依赖由 Isaac Lab 及上述可编辑安装提供。

4. **配置 IDE 类型检查**（可选但推荐）：

   仓库未内置 Pyright 路径。为了在 VS Code/Pyright 中获得正确的 IDE 支持（自动补全、类型检查），请在项目根目录 `pyproject.toml` 或 `.vscode/settings.json` 的 `python.analysis.extraPaths` 中添加 `extraPaths`：

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
lejurobot_lab/
├── source/leju_robot/                  # 主包
│   └── leju_robot/                     # 核心机器人模块
│       ├── assets/
│       │   ├── robots/                 # URDF / USD 机器人模型
│       │   └── motion_data/mimic/      # 示例 csv / npz / deploy 数据
│       ├── actuators/                  # 执行器配置
│       └── tasks/                      # 任务定义
│           ├── amp/                    # AMP 基础任务模块
│           │   ├── envs/
│           │   └── skrl/               # 定制 SKRL（AMP）
│           ├── tracking/               # 动作跟踪（dance、standup 等）
│           │   ├── agents/
│           │   ├── mdp/
│           │   └── config/
│           │       ├── robanS14/       # dance、standup、new_year_dance
│           │       ├── robanS17/       # dance
│           │       ├── kuavoS53/       # dance
│           │       └── kuavoS54/       # dance
│           └── locomotion/
│               ├── velocity/           # RSL-RL 速度任务
│               │   └── config/
│               │       ├── robanS14/
│               │       ├── kuavoS53/
│               │       ├── kuavoS54/
│               │       └── kuavoS46/   # 仅 rough
│               └── velocity_amp/       # SKRL AMP 速度
│                   └── config/robanS17/
├── scripts/
│   ├── list_envs.py                    # 列出已注册任务
│   ├── motion_tool/
│   │   ├── csv_to_npz&deploycsv.py
│   │   ├── pkl_to_npz&deploycsv.py
│   │   ├── replay_npz.py
│   │   └── replay_npz_list.py
│   └── reinforcement_learning/
│       ├── skrl/
│       └── rsl_rl/
└── .vscode/                            # launch.json 调试配置
```

### 使用方法

#### 1. CSV 转 NPZ

将动作数据从 CSV 格式转换为 NPZ 格式：

```bash
python "scripts/motion_tool/csv_to_npz&deploycsv.py" \
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
- `--robot`: 机器人型号（`robanS14`、`robanS17`、`kuavoS53`、`kuavoS54`）
- `--input_quat_xyzw`: 若指定，表示输入 CSV 四元数为 xyzw 顺序，将转换为 wxyz

#### 2. 回放动作（单个文件）

回放单个 NPZ 动作文件：

```bash
python scripts/motion_tool/replay_npz.py \
    --motion_file path/to/motion.npz \
    --robot robanS14
```

**参数说明：**
- `--motion_file`: NPZ 动作文件路径
- `--robot`: 机器人型号（`robanS14`、`robanS17`、`kuavoS53`、`kuavoS54`；默认：`robanS14`）

#### 3. 回放动作（多个文件）

按顺序回放多个 NPZ 文件：

```bash
python scripts/motion_tool/replay_npz_list.py \
    --motion_file path/to/motion1.npz \
    --robot robanS14
```

或在脚本中编辑 `MOTION_FILES` 列表以指定多个文件。

#### 4. 列出已注册环境

打印所有已注册的 `Tracking-*` 与 `Velocity-*` 任务：

```bash
python scripts/list_envs.py
```

#### 5. 训练 RL 智能体

**跟踪任务（动作模仿）：**
```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task Tracking-Dance-Flat-RobanS14 \
    --motion_file path/to/motion.npz \
    --num_envs 8192 \
    --headless \
    --max_iterations 25000
```

**跟踪任务（RobanS14 新年舞）：**
```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task Tracking-New-Year-Dance-Flat-RobanS14 \
    --motion_file path/to/motion.npz \
    --num_envs 8192 \
    --headless \
    --max_iterations 25000
```

**跟踪任务（RobanS17）：**
```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task Tracking-Dance-Flat-RobanS17 \
    --motion_file path/to/motion.npz \
    --num_envs 8192 \
    --headless \
    --max_iterations 25000
```

**速度任务（运动控制，RSL-RL）：**
```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task Velocity-Flat-RobanS14 \
    --num_envs 8192 \
    --headless \
    --max_iterations 25000
```

**速度任务（AMP 风格化运动控制，SKRL）：**
```bash
python scripts/reinforcement_learning/skrl/train.py \
    --task Velocity-AMP-RobanS17 \
    --num_envs 4096 \
    --algorithm AMP \
    --max_iterations 30000 \
    --headless
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
- `--seed`: 随机种子（可选）
- `--logger`: RSL-RL 日志后端（`wandb`、`tensorboard` 或 `neptune`）

#### 6. 运行训练好的策略

测试 RSL-RL 训练好的策略：

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
    --task Tracking-Dance-Flat-RobanS14-Play \
    --motion_file path/to/motion.npz \
    --load_run 2026-02-05_15-18-56 \
    --checkpoint model_52500.pt \
    --num_envs 1
```

测试 SKRL Velocity AMP 策略（请使用绝对路径或 `logs/skrl/.../checkpoints/` 下的路径）：

```bash
python scripts/reinforcement_learning/skrl/play.py \
    --task Velocity-AMP-RobanS17-Play \
    --num_envs 32 \
    --algorithm AMP \
    --checkpoint /path/to/logs/skrl/.../checkpoints/agent_24000.pt \
    --output_name agent_24000.onnx
```

SKRL 检查点默认保存为实验目录下 `checkpoints/agent_{timestep}.pt`。

**参数说明：**
- `--task`: 任务名称，需带 `-Play` 后缀
- `--load_run`: 训练日志中的运行 ID（RSL-RL）
- `--checkpoint`: 检查点文件路径或文件名
- `--num_envs`: 环境数量（可视化时通常为 1）

#### 7. 使用 VS Code Debug 配置

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
     - `train robanS14 walk` / `train robanS14 dance` / `train robanS14 standup`
     - `train robanS17 dance` / `train robanS17 AMP walk`
     - `train kuavoS53 walk` / `train kuavoS53 dance`
     - `train kuavoS54 walk` / `train kuavoS54 dance`
     - `train kuavoS46 rough walk`

   - **运行配置：**
     - `play robanS14 walk` / `play robanS14 dance` / `play robanS14 standup`
     - `play robanS17 dance` / `play robanS17 AMP walk`
     - `play kuavoS53 walk` / `play kuavoS53 dance`
     - `play kuavoS54 walk` / `play kuavoS54 dance`
     - `play kuavoS46 rough walk`

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

- **RobanS14**：21 自由度人形机器人
- **RobanS17**：人形机器人（跟踪舞蹈、AMP 速度控制）
- **KuavoS53**：人形机器人
- **KuavoS54**：人形机器人
- **KuavoS46**：人形机器人（仅粗糙地形速度任务；`assets/robots/kuavoS46/` 资源未包含在本仓库中，使用前请自行准备）

> **说明：** 代码库中已不再使用 `KuavoS52`，请改用 `KuavoS53` 或 `KuavoS54`。

### 可用任务

**跟踪任务（动作模仿，RSL-RL）：**
- `Tracking-Dance-Flat-RobanS14` / `Tracking-Dance-Flat-RobanS14-Play`
- `Tracking-New-Year-Dance-Flat-RobanS14` / `Tracking-New-Year-Dance-Flat-RobanS14-Play`
- `Tracking-Standup-Flat-RobanS14` / `Tracking-Standup-Flat-RobanS14-Play`
- `Tracking-Dance-Flat-RobanS17` / `Tracking-Dance-Flat-RobanS17-Play`
- `Tracking-Dance-Flat-KuavoS53` / `Tracking-Dance-Flat-KuavoS53-Play`
- `Tracking-Dance-Flat-KuavoS54` / `Tracking-Dance-Flat-KuavoS54-Play`

**速度任务（运动控制，RSL-RL）：**
- `Velocity-Flat-RobanS14` / `Velocity-Flat-RobanS14-Play`
- `Velocity-Rough-RobanS14` / `Velocity-Rough-RobanS14-Play`
- `Velocity-Flat-KuavoS53` / `Velocity-Flat-KuavoS53-Play`
- `Velocity-Rough-KuavoS53` / `Velocity-Rough-KuavoS53-Play`
- `Velocity-Flat-KuavoS54` / `Velocity-Flat-KuavoS54-Play`
- `Velocity-Rough-KuavoS54` / `Velocity-Rough-KuavoS54-Play`
- `Velocity-Rough-KuavoS46` / `Velocity-Rough-KuavoS46-Play`

**速度 AMP 任务（SKRL）：**
- `Velocity-AMP-RobanS17` / `Velocity-AMP-RobanS17-Play`

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

机器人配置路径如下：

- **跟踪：** `source/leju_robot/leju_robot/tasks/tracking/config/<robot>/<variant>/`（如 `dance`、`standup`、`new_year_dance`）
- **速度：** `source/leju_robot/leju_robot/tasks/locomotion/velocity/config/<robot>/`
- **速度 AMP：** `source/leju_robot/leju_robot/tasks/locomotion/velocity_amp/config/robanS17/`

每个配置包括环境设置、MDP 组件（观测、奖励、事件等）、智能体配置及任务特定参数。

### 许可证

Apache 2.0

### 贡献

欢迎贡献！请遵循项目的编码标准并提交 Pull Request。
