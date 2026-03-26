""" 
A really beautiful reward function, defined in eqs 9 & 10 of the paper

I'm gonna use my own terminology for reward parameters for better understanding

yaw_weight = alpha_psi
height_weight = alpha_z
obstacle distance weight = alpha d
velocity_weight = alpha_v
height_min, height_max <- height limits

-----

The reward function has 4 components
- Yaw rate reward
- height reward
- obstacle reward
- velocity reward

"""

import numpy as np

def yaw_reward(yaw_weight, yaw_rate):
    """Encourange drone to take slow turns"""

    return -yaw_weight * abs(yaw_rate)


def height_reward(
        height_weight, 
        height, 
        height_min, 
        height_max, 
        epsilon = 1e-6
        ):
    """Encourange drone to maintain a height, not too high/low"""
    
    z_norm = (height - height_min) / (height_max - height_min)
    z_norm = np.clip(z_norm, epsilon, 1 - epsilon)
    return height_weight * (np.log(z_norm) + np.log(1 - z_norm))

def obstacle_reward(
        dist_weight, 
        d_min, 
        d_max, 
        vel_body, 
        roi_depths, 
        roi_unit_vectors, 
        epsilon = 1e-6,
        smoother = 0.1 # check value again
    ):
    
    """Avoid obstacles of course"""


    # Figure out the shortest distance bw the drone and obstacle
    # i_roi = [np.dot(vel_body, u) for u in roi_unit_vectors]
    i_roi = (roi_unit_vectors @ vel_body).flatten()
    # print("i_roi share: ", i_roi[0])
    # print("roi_depths share: ", roi_depths[0])
    # d_roi_min = min([d * max(i, smoother) for d, i in zip(roi_depths, i_roi)])
    d_roi_min = np.min(np.array(roi_depths) * np.maximum(i_roi, smoother))

    d_norm = (d_roi_min - d_min) / (d_max - d_min)
    d_norm = np.clip(d_norm, epsilon, 1 - epsilon)
    return dist_weight * (np.log(d_norm) + np.log(1 - d_norm))


def velocity_reward(vel_weight, vel_body_x, epsilon = 1e-6):
    """Keep going forward - into open spaces
    x dirn means forward"""

    # return vel_weight * np.log(vel_body_x + epsilon)
    vel_body_x = max(vel_body_x, 0.0)  # avoid negative log
    return vel_weight * np.log(vel_body_x + epsilon)


#####################################################3


def get_roi_unit_vectors(m_side=8, res=224, f=55.4):

    # based on pinhole camera model
    # 224x224 unit vectors might be a lot, so we divide the image into smaller regions of interest.
    # output shape = ( m_side^2 * 3 )

    u_roi = []
    roi_size = res // m_side
    for i in range(m_side):
        for j in range(m_side):
            # Center of the ROI
            u = (i * roi_size + roi_size / 2) - res / 2 # Horizontal offset (left-right) -> Y axis
            v = (j * roi_size + roi_size / 2) - res / 2 # Vertical offset (up-down) -> Z axis
            
            # Forward is X, Horizontal is Y, Vertical is Z
            vec = np.array([f, -u / res * f, -v / res * f])
            u_roi.append(vec / np.linalg.norm(vec))
    return np.array(u_roi)

# ROI_UNIT_VECTORS = get_roi_unit_vectors()

def get_roi_depths(depth_img, m_side=8):
    depth_img = np.squeeze(depth_img)  # remove channel dim if exists
    roi_depths = []

    H, W = depth_img.shape
    h, w = H // m_side, W // m_side

    for i in range(m_side):
        for j in range(m_side):
            roi_pixels = depth_img[i*h:(i+1)*h, j*w:(j+1)*w]
            roi_depths.append(float(np.mean(roi_pixels)))  # <- convert to float

    return roi_depths

