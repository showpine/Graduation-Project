import mujoco.viewer
import ikpy.chain
import transforms3d as tf3d
import numpy as np

class JointSpaceTrajectory:
    """关节空间坐标系下的线性插值轨迹"""

    def __init__(self, start_joints, end_joints, steps):
        self.start_joints = np.array(start_joints)
        self.end_joints = np.array(end_joints)
        self.steps = steps
        self.step = (self.end_joints - self.start_joints) / self.steps
        self.trajectory = self._generate_trajectory()
        self.waypoint = self.start_joints
        self.current_step = 0

    def _generate_trajectory(self):
        for i in range(self.steps + 1):
            yield self.start_joints + self.step * i
        # 确保最后精确到达目标关节值
        yield self.end_joints

    def get_next_waypoint(self, qpos):
        # 每次调用都返回下一个轨迹点，不等待机械臂到达当前点
        try:
            self.waypoint = next(self.trajectory)
            self.current_step += 1
            return self.waypoint
        except StopIteration:
            return self.waypoint

def viewer_init(viewer):
    """渲染器的摄像头视角初始化"""
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    viewer.cam.lookat[:] = [0, 0.5, 0.5]
    viewer.cam.distance = 2.5
    viewer.cam.azimuth = 180
    viewer.cam.elevation = -30

def initialize_robot():
    """初始化机械臂模型"""
    model = mujoco.MjModel.from_xml_path('model/universal_robots_ur5e/scene.xml')
    data = mujoco.MjData(model)
    
    # 加载URDF文件
    my_chain = ikpy.chain.Chain.from_urdf_file("model/ur5e.urdf",
                                               active_links_mask=[False, False] + [True] * 6 + [False])

    # 设置初始关节角度
    start_joints = np.array([-1.57, -1.34, 2.65, -1.3, 1.55, 0])  # 对应机械臂初始位姿[-0.14, 0.3, 0.1, 3.14, 0, 1.57]
    data.qpos[:6] = start_joints  # 确保渲染一开始机械臂便处于起始位置，而非MJCF中的默认位置

    # 设置目标点
    ee_pos = [-0.13, 0.6, 0.1]
    ee_euler = [3.14, 0, 1.57]
    ref_pos = [0, 0, -1.57, -1.34, 2.65, -1.3, 1.55, 0, 0]
    ee_orientation = tf3d.euler.euler2mat(*ee_euler)

    # 计算逆运动学
    joint_angles = my_chain.inverse_kinematics(ee_pos, ee_orientation, "all", initial_position=ref_pos)
    end_joints = joint_angles[2:-1]

    # 生成轨迹
    joint_trajectory = JointSpaceTrajectory(start_joints, end_joints, steps=100)

    return model, data, joint_trajectory