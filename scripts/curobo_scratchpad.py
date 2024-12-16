import os
import torch
from time import sleep

# from jrl.robots import Panda
import numpy as np
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel, CudaRobotModelConfig, CudaRobotModelState
from curobo.cuda_robot_model.types import JointLimits
from curobo.types.base import TensorDeviceType
from curobo.types.robot import RobotConfig
from curobo.util_file import get_robot_path, join_path, load_yaml, get_content_path, get_assets_path
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig
from scipy.spatial.transform import Rotation as ScipyRotation

from robomeshcat import Scene as RmcScene, Robot as RmcRobot, Object as RmcObject


def oscillate_joints(robot: RmcRobot, robot_world: RobotWorld, joint_limits: JointLimits):
    """Move the robot around"""

    inc = 0.005
    time_p_loop = 1 / 60  # 60Hz, in theory

    upper = joint_limits.position[0, :]
    lower = joint_limits.position[1, :]
    q = (upper + lower) / 2
    ndof = 7

    def update_robot(_q):
        for i in range(ndof):
            robot[i] = _q[i]
        sleep(time_p_loop)
        print(robot_world.get_world_self_collision_distance_from_joints(_q.view(1, 7)))


    for i in range(ndof):
        l = lower[i]
        u = lower[i]

        while q[i] < u:
            q[i] += inc
            update_robot(q)

        while q[i] > l:
            q[i] -= inc
            update_robot(q)

        while q[i] < (u + l) / 2.0:
            q[i] += inc
            update_robot(q)


if __name__ == "__main__":

    tensor_args = TensorDeviceType()
    config_file = load_yaml("thirdparty/curobo/src/curobo/content/configs/robot/franka.yml")
    urdf_file = config_file["robot_cfg"]["kinematics"]["urdf_path"]
    base_link = config_file["robot_cfg"]["kinematics"]["base_link"]
    ee_link = config_file["robot_cfg"]["kinematics"]["ee_link"]

    robot_cfg: RobotConfig = RobotConfig.from_basic(urdf_file, base_link, ee_link, tensor_args)
    kin_model = CudaRobotModel(robot_cfg.kinematics)
    joint_limits = robot_cfg.kinematics.get_joint_limits()

    # Sample FK
    # q = torch.rand((10, kin_model.get_dof()), **(tensor_args.as_torch_dict()))
    # out: CudaRobotModelState = kin_model.get_state(q)

    world_config = {
        "cuboid": {
            # Note: CuRobo's quaternion format is wxyz
            "table": {"dims": [2, 2, 0.2], "pose": [0.4, 0.0, -0.1, 1, 0, 0, 0]},
            "cube_1": {"dims": [0.1, 0.1, 0.2], "pose": [0.4, 0.0, 0.5, 1, 0, 0, 0]},
            "cube_2": {"dims": [0.1, 0.1, 0.2], "pose": [0.0, 0.4, 0.5, 1, 0, 0, 0]},
        },
        # "mesh": {
        #     "scene": {
        #         "pose": [1.5, 0.080, 1.6, 0.043, -0.471, 0.284, 0.834],
        #         "file_path": "scene/nvblox/srl_ur10_bins.obj",
        #     }
        # },
    }

    config = RobotWorldConfig.load_from_config(robot_cfg, world_config, collision_activation_distance=0.0)
    robot_world = RobotWorld(config)

    # Setup RoboMeshCat
    rmc_scene = RmcScene()
    rmc_robot = RmcRobot(
        # urdf_path="/home/jstm/.cache/jrl/urdfs/panda_arm_hand_formatted_link_filepaths_absolute.urdf",
        # ^ has an exact mesh filepath, so works witout mesh_folder_path
        # urdf_path="/home/jstm/Projects/Jrl/jrl/urdfs/panda/panda_arm_hand_formatted.urdf",
        # mesh_folder_path="/home/jstm/Projects/Jrl/jrl/urdfs/panda/"
        # ^ doesn't work, b/c 'panda_arm_hand_formatted.urdf' has mesh filepaths that are interpreted as absolute
        urdf_path=os.path.join(get_assets_path(), urdf_file),
        mesh_folder_path=os.path.join(get_assets_path(), "robot/franka_description/")
    )

    # Visualize
    vis_cuboids = []
    for _, cuboid in world_config["cuboid"].items():
        R = ScipyRotation.from_quat(quat=cuboid["pose"][3:], scalar_first=True) # scalar-first order: (w, x, y, z)
        T = np.eye(4)
        T[0:3, 0:3] = R.as_matrix()
        T[0:3, 3] = cuboid["pose"][0:3]
        vis_cuboids.append(RmcObject.create_cuboid(lengths=cuboid["dims"], pose=T))
        rmc_scene.add_object(vis_cuboids[-1])

    rmc_scene.add_robot(rmc_robot, verbose=True)
    rmc_scene.render()
    sleep(1)

    oscillate_joints(rmc_robot, robot_world, joint_limits)



