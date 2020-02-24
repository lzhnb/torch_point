import numpy as np

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
    BATCH_SIZE    = 24
    LEARNING_RATE = 1e-3
    DECAY_RATE    = 1e-4
    OPTIMIZER     = "Adam"

    def __init__(self):
        pass

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")

