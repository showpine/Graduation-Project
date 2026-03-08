import time
import numpy as np
import mujoco
from robot_module import initialize_robot, viewer_init
from wireless_module import WirelessLink

def main(use_wireless=True, ebno_db=10.0):
    """主程序"""
    print("开始初始化...")
    start_time = time.time()  # 记录开始时间
    
    # 初始化机械臂
    model, data, joint_trajectory = initialize_robot()
    
    # 初始化无线链路
    wireless_link = None
    if use_wireless:
        print(f"初始化无线链路 (Eb/No = {ebno_db} dB)...")
        wireless_link = WirelessLink()
    else:
        print("不使用无线链路，直接控制机械臂...")

    print("启动Mujoco查看器...")
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
    # 测试不同的Eb/No值
    ebno_values = [0.0, 5.0, 10.0, 15.0, 20.0]
    
    # 先测试理想情况（无无线链路）
    print("=== 测试：理想情况（无无线链路）===")
    main(use_wireless=False)
    
    # 测试不同Eb/No值的无线链路情况
    for ebno in ebno_values:
        input(f"按Enter键开始测试 Eb/No = {ebno} dB 的情况...")
        print(f"\n=== 测试：Eb/No = {ebno} dB ===")
        main(use_wireless=True, ebno_db=ebno)