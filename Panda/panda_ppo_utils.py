import numpy as np
from PIL import Image
import io
import av

import torch
import torch.nn as nn
from torchvision import transforms

import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import Box

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor




class ObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        image_shape = (3, 224, 224)  
        image_dim = np.prod(image_shape)  
        coords_dim = 7  
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(image_dim + coords_dim,),
            dtype=np.float32
        )
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
        ])

    def observation(self, observation):
        image = self.env.render()
        image = Image.fromarray(image) 
        image = transform(image).numpy().flatten()  

        coords = np.concatenate([
            observation["observation"][:3],  
            observation["observation"][6],
            observation["observation"][7:10],  
        ])
        
        combined_obs = np.concatenate([image, coords])
        expected_size = self.observation_space.shape[0]
        if combined_obs.shape[0] != expected_size:
            raise ValueError(f"Problem: {combined_obs.shape[0]} != {expected_size}")
        
        return combined_obs


class FeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim, feature_extractor):
        super().__init__(observation_space, features_dim)
        self.image_extractor = feature_extractor  
        self.combined_fc = nn.Linear(512 + 7, features_dim)

    def forward(self, observations):
        batch = observations.shape[0]
        
        img_features = self.image_extractor(observations[:, :150528].reshape(batch, 3, 224, 224)).flatten(start_dim=1)
        coord_features = observations[:, -7:]
        
        combined_features = torch.cat((img_features, coord_features), dim=1)

        return self.combined_fc(combined_features)


class ValidationCallback(BaseCallback):
    def __init__(self, validation_env, validation_steps, output_filename="validation_video.mp4", fps=10, max_steps=200, verbose=0):
        super().__init__(verbose)
        self.validation_env = validation_env
        self.validation_steps = validation_steps
        self.output_filename = output_filename
        self.fps = fps
        self.max_steps = max_steps
        self.last_validation_step = 0

    def _on_step(self):
        if self.num_timesteps - self.last_validation_step >= self.validation_steps:
            self.last_validation_step = self.num_timesteps
            output_name = f"{self.output_filename}_{self.num_timesteps}.mp4"
            if self.verbose > 0:
                print(f"Validation at {self.num_timesteps} steps.")
                
            validate_video(
                                    env=self.validation_env,
                                    model=self.model,
                                    output_filename=output_name,
                                    fps=self.fps,
                                    max_steps=self.max_steps
            )
            
        return True


def make_video(numpy_images, output_filename="output_video.mp4", fps=10):
    container = av.open(output_filename, format='mp4', mode='w')
    height, width, _ = numpy_images[0].shape  
    stream = container.add_stream('h264', fps)
    stream.height = height
    stream.width = width
    stream.pix_fmt = 'yuv420p'

    for img in numpy_images:
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()  
            img = img.transpose(1, 2, 0)          
            img = (img * 255).clip(0, 255).astype('uint8')  
        
        frame = av.VideoFrame.from_ndarray(img, format='rgb24')
        for packet in stream.encode(frame):
            container.mux(packet)

    for packet in stream.encode(None):
        container.mux(packet)
    container.close()


def validate_video(env, model, output_filename="output_video.mp4", fps=10, max_steps=200):
    trajectory_images = []
    obs, _ = env.reset()
    done = False
    step = 0

    while not done and step < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(action)
        img = env.render() 
        if img.shape[0] == 3:  
            img = img.transpose(1, 2, 0)  
        trajectory_images.append(img)
        step += 1

    make_video(trajectory_images, output_filename, fps)