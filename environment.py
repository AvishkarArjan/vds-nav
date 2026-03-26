from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
import numpy as np
import pybullet as p
from gymnasium import spaces

from reward import (
    yaw_reward, 
    height_reward, 
    obstacle_reward, 
    velocity_reward,
    get_roi_unit_vectors,
    get_roi_depths
    )


"""
obs
useful info on observation state of the drone
its a (1, 20) array

Indices,Property,Description
"0, 1, 2",Position,"Current [X,Y,Z] coordinates in the world."
"3, 4, 5, 6",Quaternion,"Orientation [x,y,z,w] (how it's rotated)."
"7, 8, 9",RPY,"Roll, Pitch, and Yaw (rotation in degrees/radians)."
"10, 11, 12",Linear Velocity,"How fast it's moving in [X,Y,Z]."
"13, 14, 15",Angular Velocity,How fast it's spinning around its own axes.
"16, 17, 18, 19",Last Action,The 4 motor speeds applied in the previous step.
"""

class VDSEnv(CtrlAviary):
    def __init__(
            self, 
            drone_model = DroneModel.CF2X, 
            num_drones = 1, 
            neighbourhood_radius = np.inf, 
            initial_xyzs=None, 
            initial_rpys=None, 
            physics = Physics.PYB, 
            pyb_freq = 240, 
            ctrl_freq = 240, 
            gui=False, 
            record=False, 
            obstacles=False, 
            user_debug_gui=True, 
            output_folder='results',

            img_res = np.array((224,224)),
            img_frame_id = 0,

            # reward parameters

            yaw_weight = 1.0,
            height_weight = 0.5,
            dist_weight = 2.0,
            vel_weight = 1.0,
            z_min = 0.5,
            z_max = 2.5,
            d_min = 0.2,
            d_max = 5.0,
            smoother = 0.1,
            epsilon = 1e-6,
            m_side = 8

        ):
        super().__init__(
            drone_model, 
            num_drones, 
            neighbourhood_radius, 
            initial_xyzs, 
            initial_rpys, 
            physics, 
            pyb_freq, 
            ctrl_freq, 
            gui, 
            record, 
            obstacles, 
            user_debug_gui, 
            output_folder
            )

        self.IMG_RES = np.array(img_res) 
        self.IMG_FRAME_ID = img_frame_id

        self.drone_id = 0

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.IMG_RES[1], self.IMG_RES[0], 1),  # (H, W, C)
            dtype=np.float32
        )

        self.yaw_weight = yaw_weight
        self.height_weight = height_weight
        self.dist_weight = dist_weight
        self.vel_weight = vel_weight
        self.z_min = z_min
        self.z_max = z_max
        self.d_min = d_min
        self.d_max = d_max
        self.smoother = smoother
        self.epsilon = epsilon
        self.m_side = m_side

    def _computeObs(self):
        # override obs
        # in BaseAviary - it returns the state, I'm returning the depth img
        rgb, depth, seg = self._getDroneImages(0)
        depth = np.expand_dims(depth, axis=2)
        return depth

    def _computeReward(self):

        obs = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])[0]
        frame = self._getDroneImages(0) # 0 is the ID of the first drone
        """
        The 'frame' variable contains:
        frame[0]: RGB image
        frame[1]: Depth map (how far objects are)
        frame[2]: Segmentation mask (which object is which)
        """


        yaw_rate = obs[9]
        height = obs[2]
        # vel_body = obs[10:13]
        depth_img = frame[1]
        roi_depths =  get_roi_depths(depth_img, m_side=8)
        roi_unit_vectors = get_roi_unit_vectors()
        # vel_body_x = vel_body[0]


        ################
        pos, orn = p.getBasePositionAndOrientation(self.DRONE_IDS[0])
        lin_vel, ang_vel = p.getBaseVelocity(self.DRONE_IDS[0])
        # Convert quaternion to rotation matrix
        R = np.array(p.getMatrixFromQuaternion(orn)).reshape(3,3)
        # World velocity → Body velocity
        vel_body = R.T @ np.array(lin_vel)
        vel_body_x = vel_body[0] 

        vel_body_x = max(vel_body_x, 0.0)
        ################

        r_yaw = yaw_reward(
            self.yaw_weight,
            yaw_rate
        )
        r_height = height_reward(
            self.height_weight, 
            height, 
            self.z_min, 
            self.z_max, 
            epsilon = self.epsilon
        )
        r_dist = obstacle_reward(
            self.dist_weight, 
            self.d_min, 
            self.d_max, 
            vel_body, 
            roi_depths, 
            roi_unit_vectors, 
            self.epsilon,
            self.smoother 
        )
        r_vel = velocity_reward(
            self.vel_weight, 
            vel_body_x, 
            self.epsilon
        )

        return r_yaw + r_height + r_dist + r_vel

import numpy as np
import pybullet as p
from gymnasium import spaces
from environment import VDSEnv  # your custom env base

class HallwayNavEnv(VDSEnv):
    def __init__(self, max_steps=500, **kwargs):
        # Hallway dimensions
        self.hallway_length = 10.0
        self.hallway_width = 1.0
        self.hallway_height = 2.0
        
        self.point_A = np.array([0.5, 0.0, 1.0])
        self.point_B = np.array([self.hallway_length - 0.5, 0.0, 1.0])
        
        if 'initial_xyzs' not in kwargs:
            kwargs['initial_xyzs'] = self.point_A.reshape(1, 3)
            
        kwargs['obstacles'] = True  # MUST be true so BaseAviary triggers _addObstacles()
        
        # Obstacles: random or fixed positions (MUST BE DEFINED BEFORE super().__init__)
        self.obstacles = [
            {'pos': [2.0, 0.0, 1.0], 'size': [0.2, 0.2, 2.0]},
            {'pos': [4.5, -0.2, 1.2], 'size': [0.3, 0.3, 2.0]},
            {'pos': [7.0, 0.3, 0.8], 'size': [0.4, 0.4, 2.0]}
        ]
        
     
        self.max_steps = max_steps
        self.step_counter = 0
        
        
        self.x_bounds = (0, self.hallway_length)
        self.y_bounds = (-self.hallway_width / 2, self.hallway_width / 2)
        self.z_bounds = (0.5, self.hallway_height)
        
   
        super().__init__(**kwargs)
        
        self.pid = DSLPIDControl(drone_model=self.DRONE_MODEL)

    def _create_box(self, center, size, color=[0.7, 0.7, 0.7, 1]):
        half_extents = [size[0]/2, size[1]/2, size[2]/2]
        col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
        vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color)
        return p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=col_id,
            baseVisualShapeIndex=vis_id,
            basePosition=center
        )

    def _addObstacles(self):
        self.obstacle_ids = []
        
        self.obstacle_ids.append(self._create_box(
            center=[self.hallway_length/2, 0.0, 0.0],
            size=[self.hallway_length, self.hallway_width, 0.1],
            color=[0.3, 0.8, 0.3, 1.0] 
        ))
        
        
        self.obstacle_ids.append(self._create_box(
            center=[self.hallway_length/2, -self.hallway_width/2 - 0.05, self.hallway_height/2],
            size=[self.hallway_length, 0.1, self.hallway_height],
            color=[0.2, 0.2, 0.8, 1.0]
        ))
        
        self.obstacle_ids.append(self._create_box(
            center=[self.hallway_length/2, self.hallway_width/2 + 0.05, self.hallway_height/2],
            size=[self.hallway_length, 0.1, self.hallway_height],
            color=[0.2, 0.2, 0.8, 1.0] 
        ))

        for obs in self.obstacles:
            self.obstacle_ids.append(self._create_box(
                center=obs['pos'], 
                size=obs['size'], 
                color=[0.9, 0.1, 0.1, 1.0] 
            ))

    # def reset(self, seed=None, options=None):
    #     self.step_counter = 0
    #     return super().reset(seed=seed, options=options)

    def step(self, action):
        # 1. The PPO RL Policy outputs action within [-1, 1] for 4 continuously mapped channels
        # (v_x, v_y, v_z, yaw_rate)
        flat_action = np.array(action).flatten()
        
        # Scale the [-1, 1] network outputs to realistic meters/second velocities
        vx = flat_action[0] * 1.0
        vy = flat_action[1] * 1.0
        vz = flat_action[2] * 0.5
        yaw_rate = flat_action[3] * 1.0
        target_vel = np.array([vx, vy, vz])
        
        # 2. Extract current exact state
        state = self._getDroneStateVector(self.drone_id)
        current_pos = state[0:3]
        current_rpy = state[7:10]
        
        # 3. Integrate velocities forward by 1 simulation timestep to create structural PID targets
        dt = getattr(self, "CTRL_TIMESTEP", 1./240.) # fallback to 240Hz just in case
        target_pos = current_pos + target_vel * dt
        target_rpy = np.array([0.0, 0.0, current_rpy[2] + yaw_rate * dt])
        
        # 4. Generate literal motor RPM forces needed to hit these velocity targets smoothly
        rpms, _, _ = self.pid.computeControlFromState(
            control_timestep=dt,
            state=state,
            target_pos=target_pos,
            target_rpy=target_rpy,
            target_vel=target_vel
        )
        rpm_action = rpms.reshape(1, 4)

        # 5. Feed the RPMs mathematically back into PyBullet engine to move the Drone!
        obs, reward, terminated, truncated, info = super().step(rpm_action)
        
        self.step_counter += 1
        
        # Get current position for bounds-checking
        drone_pos = self.get_drone_state()['pos']
        
        # Check hallway boundaries
        x, y, z = drone_pos
        if not (self.x_bounds[0] <= x <= self.x_bounds[1] and
                self.y_bounds[0] <= y <= self.y_bounds[1] and
                self.z_bounds[0] <= z <= self.z_bounds[1]):
            reward -= 50.0
            terminated = True
            try:
                from rich import print as rprint
                rprint(f"[bold red]💥 Out of bounds at [x={x:.2f}, y={y:.2f}, z={z:.2f}]! Resetting...[/bold red]")
            except ImportError:
                print(f"💥 Out of bounds at [x={x:.2f}, y={y:.2f}, z={z:.2f}]! Resetting...")
        
        # VDS-Nav removes explicit goal positions and time limits in favor of pure obstacle avoidance and velocity rewards
        
        return obs, reward, terminated, truncated, info

    def get_drone_state(self):
        # Return simplified state (position + velocity)
        pos, orn = p.getBasePositionAndOrientation(self.DRONE_IDS[0])
        lin_vel, ang_vel = p.getBaseVelocity(self.DRONE_IDS[0])
        return {'pos': np.array(pos), 'vel': np.array(lin_vel), 'ang_vel': np.array(ang_vel)}

def get_env(
    IMG_RES = (224, 224),
    ):

    INIT_RPYS = np.array([[0, 0, 0]]) # np.array([[0, 0, np.pi/2]])
    env = HallwayNavEnv(
        drone_model=DroneModel.CF2X,
        num_drones=1,
        # omitted initial_xyzs to let HallwayNavEnv gracefully use its internal default `point_A`
        physics=Physics.PYB,
        initial_rpys=INIT_RPYS, # Set initial face here
        gui=True,
        obstacles=True,

        img_res = np.array(IMG_RES),
        img_frame_id = 0
    )

    return env


def get_pid():

    pid = DSLPIDControl(drone_model=DroneModel.CF2X)
    return pid


