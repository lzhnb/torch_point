import numpy as np
import torch

# Base Configuration Class
# Don't use this class directly. Instead, sub-class it and override
# the configurations you need to change.

class Config(object):
    """Base configuration class. For custom configurations, create a
    sub-class that inherits from this one and override properties
    that need to be changed.
    """
    # dataset parameters
    NUM_POINT = 1024
    NUM_CLASS = 40

    # model parameters
    NORMAL        = True
    LEARNING_RATE = 1e-3
    DECAY_RATE    = 1e-4
    OPTIMIZER     = "Adam"

    # GPU setting
    BATCH_SIZE_PER_GPU = 24
    _NUM_GPU            = 1

    def __init__(self):
        self.BATCH_SIZE = self.BATCH_SIZE_PER_GPU * self._NUM_GPU

    @property
    def NUM_GPU(self):
        return self._NUM_GPU
    
    @NUM_GPU.setter
    def NUM_GPU(self, value):
        if not isinstance(value, int):
            raise ValueError('GPU count must be an integer!')
        if value < 1 or value > torch.cuda.device_count():
            raise ValueError('GPU count must between 1 ~ num_of_gpu!')
        self._NUM_GPU = value
        self.BATCH_SIZE = self.BATCH_SIZE_PER_GPU * self._NUM_GPU

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")


