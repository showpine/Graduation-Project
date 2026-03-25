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


def main(use_wireless=True, ebno_db=10.0, use_force_sensor=False, use_advanced_wireless=True):
    """主程序
    
    Args:
        use_wireless: 是否使用无线链路
        ebno_db: 信噪比(dB)
        use_force_sensor: 是否启用力传感器可视化
        use_advanced_wireless: 是否使用高级无线链路
    """
    start_time = time.time()

    if use_wireless:
        print(f"使用无线链路，Eb/No = {ebno_db} dB")
        if use_advanced_wireless:
            print("使用高级无线链路（3GPP CDL信道模型 + OFDM）")
            wireless_link = AdvancedWirelessLink()
        else:
            print("使用基本无线链路（AWGN信道）")
            wireless_link = WirelessLink()
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
        total_control_time = 0.0  # 总控制时间
        total_transmissions = 0  # 总传输次数
        ber_sum = 0.0  # 总误码率

        while viewer.is_running() and step_count < 200:  # 限制步数以便调试
            step_count += 1
            waypoint = joint_trajectory.get_next_waypoint(data.qpos[:6])

            # 控制机械臂
            control_start_time = time.time()
            if use_wireless and wireless_link:
                # 通过无线链路传输关节角度
                transmitted_waypoint, ber = wireless_link.transmit(waypoint, ebno_db=ebno_db)
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
            else:
                # 直接控制机械臂
                data.ctrl[:6] = waypoint
                if step_count % 20 == 0:  # 每20步输出一次，控制输出量
                    print(f"步骤 {step_count}: 直接控制路径点: {waypoint}")

            # 计算控制时间
            control_time = time.time() - control_start_time
            total_control_time += control_time

            # 获取并显示力传感器数据（可选）
            if force_sensor is not None and force_plotter is not None:
                filtered_force = force_sensor.filter()
                force_plotter.plot_force_vector(filtered_force)

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.002)  # 添加小延迟，确保机械臂运动流畅

        # 计算总运行时间
        total_run_time = time.time() - start_time

        # 计算并输出统计信息
        if total_transmissions > 0:
            average_ber = ber_sum / total_transmissions
            print(f"\n平均误码率: {average_ber:.6f}")
        average_control_time = total_control_time / step_count if step_count > 0 else 0.0
        print(f"平均控制时间: {average_control_time*1000:.2f}ms")
        print(f"总运行时间: {total_run_time:.2f}s")
        print("退出主循环...")


if __name__ == "__main__":
    ''' 
    使用高级无线链路（3GPP CDL信道模型 + OFDM）use_advanced_wireless=True
    使用基本无线链路（AWGN信道） use_advanced_wireless=False
    使用力传感器 use_force_sensor=True
    不使用力传感器 use_force_sensor=False
    '''

    # main(use_wireless=True, ebno_db=10.0, use_force_sensor=True, use_advanced_wireless=True)

    main(use_wireless=False, use_force_sensor=True)
