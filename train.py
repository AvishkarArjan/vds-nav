# %%
import os
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
import cv2
import numpy as np

class DisplayLiveDepthCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        
    def _on_step(self) -> bool:
        if hasattr(self.training_env, 'envs'):
            env = self.training_env.envs[0]
            unwrapped_env = getattr(env, "unwrapped", env)
            
            try:
                # Get the images (RGB, Depth, Segmentation)
                _, depth, _ = unwrapped_env._getDroneImages(0)
                
                # Convert depth to a displayable 8-bit image with color map
                depth_vis = (np.clip(depth, 0, 1) * 255.0).astype(np.uint8)
                depth_colormap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                
                # Resize for better visibility (400x400)
                depth_display = cv2.resize(depth_colormap, (400, 400), interpolation=cv2.INTER_NEAREST)
                
                cv2.imshow("Drone Live Depth View", depth_display)
                cv2.waitKey(1)
            except Exception as e:
                # Gracefully skip if image fails to render
                pass
        return True

from environment import get_env
from model import VDS_Nav_CNN, get_volumetric_observation_space


# %%
# paths
filename = datetime.now().strftime("vds_nav_%Y%m%d_%H%M%S")
save_path = os.path.join("results", filename)
os.makedirs(save_path, exist_ok=True)

# %%
# config

height, width = 96, 96
n = 3 # seq length of depth images
max_depth = 1 # coz hopefully, all depth distances are normalize
from gymnasium.wrappers import FrameStackObservation
train_env = get_env((height, width))
train_env = FrameStackObservation(train_env, n)

print("#############################")
print("Sanity Check")
obs, _ = train_env.reset()

print(train_env.observation_space)
print(obs.shape)
print("#############################")

policy_kwargs = {
    "features_extractor_class" : VDS_Nav_CNN,
    "features_extractor_kwargs" : {
            "out_dim":4
        }

}
model = PPO(
    "CnnPolicy", # Although we use custom extractor, we start from CnnPolicy or MultiInputPolicy
    train_env,
    policy_kwargs=policy_kwargs,
    verbose=1,
    tensorboard_log=os.path.join(save_path, "tb_logs"),
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01
)

checkpoint_callback = CheckpointCallback(
        save_freq=10000, 
        save_path=os.path.join(save_path, "checkpoints"),
        name_prefix="vds_model"
    )

# %%
# 5. Training
print("Starting training...")
try:
    model.learn(
        total_timesteps=1000000,
        callback=[checkpoint_callback, DisplayLiveDepthCallback()],
        tb_log_name="PPO_VDS"
    )
except KeyboardInterrupt:
    print("Training interrupted.")
finally:
    # Save final model
    model.save(os.path.join(save_path, "vds_final_model"))
    print(f"Model saved to {save_path}")

# %%



