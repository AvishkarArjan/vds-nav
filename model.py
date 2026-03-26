import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
import numpy as np

def get_volumetric_observation_space(height=224, width = 224, n = 3, max_depth=1): 

    return spaces.Box(
        low=0.0,
        high=max_depth,
        shape=(n, height, width),
        dtype=np.float32
    )

class VDS_Nav_CNN(BaseFeaturesExtractor):
    def __init__(
            self, 
            observation_space: spaces.Box,
            out_dim: int = 4
            ):
        super().__init__(observation_space, out_dim)


        # n_input_channels = observation_space.shape[2]
        stack, h, w, c = observation_space.shape
        n_input_channels = stack * c



        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        with torch.no_grad():
            sample = torch.zeros(1, n_input_channels, h, w)
            n_flatten = self.cnn(sample).shape[1]
            
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, 32),
            nn.ELU(),
            nn.Linear(32, 32),
            nn.ELU(),
            nn.Linear(32, out_dim) 
        )
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # print(observations.shape)
        B, S, H, W, C = observations.shape
        observations = observations.permute(0,1,4,2,3)   # (B,S,C,H,W)
        observations = observations.reshape(B, S*C, H, W)

        return self.linear(self.cnn(observations))

if __name__ == "__main__":
    from torchsummary import summary
    import numpy as np


    height, width = 224, 224
    max_depth = 1

    obs_space = spaces.Box(
        low=0.0,
        high=max_depth,          # meters
        shape=(1, height, width),
        dtype=np.float32
    )

    model = VDS_Nav_CNN(observation_space=obs_space)
    summary(model, input_size=(1, height, width))

    # test forward pass
    dummy_input = torch.randn(1, 1, height, width)
    output = model(dummy_input)
    
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print("Model verified")

