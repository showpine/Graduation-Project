import time
import numpy as np
import mujoco
from robot_module import (
    ForceSensor,
    ForcePlotter,
    viewer_init,
    initialize_robot
)
from wireless_module import WirelessLink, AdvancedWirelessLink


def main(use_wireless=True, ebno_db=10.0, use_force_sensor=False, use_advanced_wireless=True, cdl_model="C", speed=10.0, delay_spread=100e-9, 
         bs_antennas=4, subcarrier_spacing=30e3, fft_size=76):
    """主程序
    
    仿真环境与条件说明：
    1. 传播距离：10公里（机械臂到基站的距离）
    2. 物理层处理能力：
       - 编码/解码速度：100 Mbps
       - 调制/解调速度：100 Msymbols/s（QPSK调制时相当于200 Mbps）
       - OFDM处理速度：1 GHz
    3. 传输参数：
       - 基本无线链路：50 Mbps带宽
       - 高级无线链路：50 Mbps带宽
    4. 信道模型：
       - 基本无线链路：AWGN信道
       - 高级无线链路：3GPP CDL信道模型
    5. 天线配置：
       - UT（机械臂端）：1根天线
       - BS（控制端）：可配置4/8/16根天线
    6. 数据传输：
       - 6个关节角度，每个16位精度，共96比特
       - LDPC编码，码率可配置（0.3/0.5/0.7）
       - QPSK调制
    7. 时延计算：
       - 物理层处理时延：编码/解码、调制/解调、OFDM处理
       - 传播时延：基于光速（300,000 km/s）
       - 传输时延：基于带宽和数据量
    
    Args:
        use_wireless: 是否使用无线链路
        ebno_db: 信噪比(dB)
        use_force_sensor: 是否启用力传感器可视化
        use_advanced_wireless: 是否使用高级无线链路
        cdl_model: CDL信道模型(A/B/C/D/E)
        speed: 移动速度(m/s)
        delay_spread: 延迟扩展(s)
        bs_antennas: BS天线数量
        subcarrier_spacing: 子载波间隔(Hz)
        fft_size: FFT大小
    """
    if use_advanced_wireless:
        print(f"使用CDL信道模型: {cdl_model}")
        print(f"移动速度: {speed} m/s")
        print(f"延迟扩展: {delay_spread * 1e9:.0f} ns")
        print(f"UT天线数量: 1")
        print(f"BS天线数量: {bs_antennas}")
        print(f"子载波间隔: {subcarrier_spacing/1000:.0f} kHz")
        print(f"FFT大小: {fft_size}")
    start_time = time.time()

    if use_wireless:
        print(f"使用无线链路，Eb/No = {ebno_db} dB")
        if use_advanced_wireless:
            print("使用高级无线链路（3GPP CDL信道模型 + OFDM）")
            wireless_link = AdvancedWirelessLink(cdl_model=cdl_model, speed=speed, delay_spread=delay_spread, 
                                               bs_antennas=bs_antennas, 
                                               subcarrier_spacing=subcarrier_spacing, fft_size=fft_size)
        else:
            print("使用基本无线链路（AWGN信道）")
            wireless_link = WirelessLink(coderate=0.3)
    else:
        print("理想情况（无无线链路）")
        wireless_link = None

    # 初始化机械臂
    model, data, joint_trajectory, end_joints, force_sensor, force_plotter = initialize_robot(use_force_sensor=use_force_sensor)

    # 启动viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer_init(viewer)
        print("进入主循环...")
        step_count = 0
        total_wireless_delay = 0.0  # 总实际无线链路时延（仿真值）
        total_transmissions = 0  # 总传输次数
        ber_sum = 0.0  # 总误码率

        while viewer.is_running() and step_count < 200:  # 限制步数以便调试
            step_count += 1
            waypoint = joint_trajectory.get_next_waypoint(data.qpos[:6])

            # 控制机械臂
            if use_wireless and wireless_link:
                # 通过无线链路传输关节角度
                transmitted_waypoint, ber, wireless_delay = wireless_link.transmit(waypoint, ebno_db=ebno_db)
                total_wireless_delay += wireless_delay
                data.ctrl[:6] = transmitted_waypoint
                ber_sum += ber
                total_transmissions += 1
                if step_count % 20 == 0:  # 每20步输出一次，控制输出量
                    print(f"步骤 {step_count}: 原始路径点: {waypoint}")
                    print(f"步骤 {step_count}: 传输后路径点: {transmitted_waypoint}")
                    # 计算传输前后的差异
                    diff = np.abs(np.array(waypoint) - np.array(transmitted_waypoint))
                    print(f"步骤 {step_count}: 传输差异: {diff}")
                    print(f"步骤 {step_count}: 误码率: {ber:.6f}")
                    print(f"步骤 {step_count}: 无线链路时延: {wireless_delay*1000:.4f}ms")
            else:
                # 直接控制机械臂
                data.ctrl[:6] = waypoint
                if step_count % 20 == 0:  # 每20步输出一次，控制输出量
                    print(f"步骤 {step_count}: 直接控制路径点: {waypoint}")

            # 获取并显示力传感器数据（可选）
            if force_sensor is not None and force_plotter is not None:
                filtered_force = force_sensor.filter()
                force_plotter.plot_force_vector(filtered_force)

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.002)  # 添加小延迟，确保机械臂运动流畅

        # 计算并输出统计信息
        if total_transmissions > 0:
            average_ber = ber_sum / total_transmissions
            print(f"\n平均误码率: {average_ber:.6f}")
            print(f"总无线链路时延: {total_wireless_delay*1000:.4f}ms")
        print("退出主循环...")


if __name__ == "__main__":

    # main(use_wireless=True, ebno_db=0.0, use_force_sensor=True, use_advanced_wireless=False)
    main(use_advanced_wireless=True, use_force_sensor=True, ebno_db=0.0, fft_size=312)
    # main(use_wireless=False, use_force_sensor=True)
