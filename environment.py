import numpy as np


# mimic a "gym" environment (this does not inherit from gym.Env)
class MyEnvironment():
    def __init__(self, dataloader):
        self.dataloader = iter(dataloader)

    def reset(self):
        image_state, self.image_target = next(self.dataloader)
        return image_state
    
    def step(self, action):
        # return observation, reward, done, reset
        pass
