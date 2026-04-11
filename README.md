# 机械臂远程操控无线链路仿真系统

## 项目概述

本项目实现了一个机械臂远程操控的无线链路仿真系统，集成了Sionna无线链路仿真库，用于研究无线通信质量对机械臂控制性能的影响。系统支持不同信道模型、调制方式、码率等参数的配置，可用于分析通信误差和时延对控制系统稳定性的作用机理。

## 目录结构

```
.
├── main.py                 # 主程序，协调无线模块和机械臂模块
├── wireless_module.py      # 无线链路仿真模块，实现不同调制方式和码率的无线传输
├── robot_module.py         # 机械臂相关类和函数，包括关节空间轨迹规划、力传感器等
└── model/                  # 机械臂模型文件
    ├── universal_robots_ur5e/
    │   └── scene.xml       # 机械臂场景文件
    └── ur5e.urdf           # 机械臂URDF模型文件
```

## 系统架构

系统由三个主要模块组成：

1. **主程序模块** (`main.py`)
   - 协调无线模块和机械臂模块
   - 控制机械臂运动
   - 统计误码率和无线链路时延

2. **无线链路模块** (`wireless_module.py`)
   - 基本无线链路：AWGN信道 + QPSK调制 + LDPC编码
   - 高级无线链路：3GPP CDL信道模型 + OFDM + QPSK调制 + LDPC编码
   - 支持不同码率、调制方式、天线配置等参数

3. **机械臂模块** (`robot_module.py`)
   - 关节空间轨迹规划
   - 力传感器数据采集与可视化
   - 机械臂控制

## 核心功能

1. **无线链路仿真**
   - 支持AWGN信道和3GPP CDL信道模型
   - 支持QPSK调制
   - 支持LDPC编码，码率可配置（0.3/0.5/0.7）
   - 计算误码率和无线链路时延

2. **机械臂控制**
   - 关节空间轨迹规划
   - 逆运动学计算
   - 力传感器数据采集与可视化

3. **仿真数据分析**
   - 统计误码率
   - 计算无线链路时延
   - 分析通信质量对控制效果的影响

## 安装与依赖

### 依赖项

- Python 3.7+
- NumPy
- TensorFlow 2.0+
- Sionna (无线链路仿真库)
- MuJoCo (机械臂物理仿真)
- IKPy (逆运动学计算)
- transforms3d (坐标变换)
- matplotlib (力传感器数据可视化)

### 安装步骤

1. 安装Python 3.7+
2. 安装依赖包：
   ```bash
   pip install numpy tensorflow sionna mujoco ikpy transforms3d matplotlib
   ```
3. 下载机械臂模型文件到`model`目录

## 使用方法

### 基本用法

```python
# 运行高级无线链路仿真
python main.py

# 运行基本无线链路仿真
python main.py --use_advanced_wireless=False

# 运行理想情况（无无线链路）
python main.py --use_wireless=False
```

### 参数配置

可以通过修改`main.py`中的参数来配置仿真环境：

- `use_wireless`: 是否使用无线链路
- `ebno_db`: 信噪比(dB)
- `use_force_sensor`: 是否启用力传感器可视化
- `use_advanced_wireless`: 是否使用高级无线链路
- `cdl_model`: CDL信道模型(A/B/C/D/E)
- `speed`: 移动速度(m/s)
- `delay_spread`: 延迟扩展(s)
- `bs_antennas`: BS天线数量
- `subcarrier_spacing`: 子载波间隔(Hz)
- `fft_size`: FFT大小

### 示例

```python
# 使用高级无线链路，CDL模型C，速度10m/s，天线数量4
main(use_advanced_wireless=True, cdl_model="C", speed=10.0, bs_antennas=4)

# 使用基本无线链路，码率0.3，信噪比0dB
main(use_advanced_wireless=False, ebno_db=0.0)
```

## 仿真环境与参数

### 仿真环境

1. **传播距离**：10公里（机械臂到基站的距离）
2. **物理层处理能力**：
   - 编码/解码速度：100 Mbps
   - 调制/解调速度：100 Msymbols/s（QPSK调制时相当于200 Mbps）
   - OFDM处理速度：1 GHz
3. **传输参数**：
   - 基本无线链路：50 Mbps带宽
   - 高级无线链路：50 Mbps带宽
4. **信道模型**：
   - 基本无线链路：AWGN信道
   - 高级无线链路：3GPP CDL信道模型
5. **天线配置**：
   - UT（机械臂端）：1根天线
   - BS（控制端）：可配置4/8/16根天线
6. **数据传输**：
   - 6个关节角度，每个16位精度，共96比特
   - LDPC编码，码率可配置（0.3/0.5/0.7）
   - QPSK调制
7. **时延计算**：
   - 物理层处理时延：编码/解码、调制/解调、OFDM处理
   - 传播时延：基于光速（300,000 km/s）
   - 传输时延：基于带宽和数据量

### 关键参数

| 参数 | 说明 | 可选值 |
|------|------|--------|
| cdl_model | CDL信道模型 | A/B/C/D/E |
| speed | 移动速度 | 0-72 m/s |
| delay_spread | 延迟扩展 | 5e-8 - 5e-7 s |
| bs_antennas | BS天线数量 | 4/8/16 |
| subcarrier_spacing | 子载波间隔 | 15e3/30e3/60e3 Hz |
| fft_size | FFT大小 | 76/156/312 |
| coderate | LDPC码率 | 0.3/0.5/0.7 |
| ebno_db | 信噪比 | -5.0 - 15.0 dB |



## 代码说明

### main.py

主程序文件，负责协调无线模块和机械臂模块，控制机械臂运动，统计误码率和无线链路时延。

### wireless_module.py

无线链路仿真模块，实现了两种无线链路：

- **WirelessLink**：基本无线链路，使用AWGN信道、QPSK调制和LDPC编码
- **AdvancedWirelessLink**：高级无线链路，使用3GPP CDL信道模型、OFDM、QPSK调制和LDPC编码

### robot_module.py

机械臂相关类和函数：

- **JointSpaceTrajectory**：关节空间坐标系下的线性插值轨迹
- **ForceSensor**：力传感器数据采集与滤波
- **ForcePlotter**：力传感器数据可视化
- **viewer_init**：渲染器的摄像头视角初始化
- **initialize_robot**：初始化机械臂模型

## 总结

本项目成功实现了机械臂远程操控的无线链路仿真系统，集成了Sionna无线链路仿真库，支持多种信道模型和参数配置。通过仿真分析，我们发现了无线通信质量对机械臂控制性能的影响规律，为实际系统设计提供了参考。

该系统可以进一步扩展，例如添加更多调制方式、信道模型，或者与实际机械臂硬件集成，以实现更真实的远程操控场景。