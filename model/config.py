import os, sys
import numpy as np
import torch

# Base function for decorator
def origin_property(name):
    storage_name = '__' + name
    @property
    def prop(self):
        return getattr(self, storage_name)
    @prop.setter
    def prop(self, value):
        if value is not None:
            setattr(self, storage_name, value)
    return prop

# Base Configuration Class
# Don"t use this class directly. Instead, sub-class it and override
# the configurations you need to change.

class Config(object):
    """Base configuration class. For custom configurations, create a
    sub-class that inherits from this one and override properties
    that need to be changed.
    """
    # define as decorator
    NUM_POINT          = origin_property("NUM_POINT")
    NUM_CLASS          = origin_property("NUM_CLASS")
    NUM_WORKER         = origin_property("NUM_WORKER")
    EPOCH              = origin_property("EPOCH")
    STEP_SIZE          = origin_property("STEP_SIZE")
    LEARNING_RATE      = origin_property("LEARNING_RATE")
    LOG_DIR            = origin_property("LOG_DIR")
    DECAY_RATE         = origin_property("DECAY_RATE")
    OPTIMIZER          = origin_property("OPTIMIZER")
    MOMENTUM_ORIGINAL  = origin_property("MOMENTUM_ORIGINAL")
    MOMENTUM_DECCAY    = origin_property("MOMENTUM_DECCAY")
    MOMENTUM_CLIP      = origin_property("MOMENTUM_CLIP")
    BATCH_SIZE_PER_GPU = origin_property("BATCH_SIZE_PER_GPU")
    MODEL              = origin_property("MODEL")
    

    # dataset parameters
    NUM_POINT  = 1024
    NUM_CLASS  = 40
    NUM_PART   = 50
    NUM_WORKER = 4

    # model parameters
    EPOCH           = 200
    STEP_SIZE       = 20
    LEARNING_RATE   = 1e-3
    LOG_DIR         = None
    NORMAL          = True
    DECAY_RATE      = 1e-4
    OPTIMIZER       = "Adam"
    
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY   = 0.5
    MOMENTUM_CLIP     = 0.01

    # GPU setting
    BATCH_SIZE_PER_GPU = 24
    __NUM_GPU          = 1

    # ShapeNet set
    SEG_CLASSES = {
            "Earphone":   [16, 17, 18],
            "Motorbike":  [30, 31, 32, 33, 34, 35],
            "Rocket":     [41, 42, 43],
            "Car":        [8, 9, 10, 11],
            "Laptop":     [28, 29],
            "Cap":        [6, 7],
            "Skateboard": [44, 45, 46],
            "Mug":        [36, 37],
            "Guitar":     [19, 20, 21],
            "Bag":        [4, 5],
            "Lamp":       [24, 25, 26, 27],
            "Table":      [47, 48, 49],
            "Airplane":   [0, 1, 2, 3],
            "Pistol":     [38, 39, 40],
            "Chair":      [12, 13, 14, 15],
            "Knife":      [22, 23]
        }
    SEG_LABEL_TO_CAT = {}# {0:Airplane, 1:Airplane, ...49:Table}
    
    # model define
    __TASK = "cls"
    MODEL = "ponitnet" # pointnet or pointnet2
    # task -> dataset path
    TASK_DATA_PATH = {
            "cls":      os.path.join(os.getcwd(), "data", "modelnet40_normal_resampled"),
            "part_seg": os.path.join(os.getcwd(), "data", "shapenetcore_partanno_segmentation_benchmark_v0_normal"),
        }
    # task -> show index
    SHOW_INDEX = {
            "cls":      "Accuracy",
            "part_seg": "mIOU acc",
        }

    def __init__(self):
        self.set_label_to_cat()
        self.BATCH_SIZE = self.BATCH_SIZE_PER_GPU * self.__NUM_GPU
        self.DATA_PATH  = self.TASK_DATA_PATH[self.__TASK]

    def set_label_to_cat(self):
        for cat in self.SEG_CLASSES.keys():
            for label in self.SEG_CLASSES[cat]:
                self.SEG_LABEL_TO_CAT[label] = cat

    def update(self, cfg_key, args_value):
        if args_value is not None:
            cfg_value = args_value

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if a in ["SEG_LABEL_TO_CAT", "SEG_CLASSES", "TASK_DATA_PATH", "SHOW_INDEX"]: continue
            elif not a.startswith("_") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")

    # change relative
    @property
    def NUM_GPU(self):
        return self.__NUM_GPU
    
    @NUM_GPU.setter
    def NUM_GPU(self, value):
        if not isinstance(value, int):
            raise ValueError("GPU count must be an integer!")
        if value < 1 or value > torch.cuda.device_count():
            raise ValueError("GPU count must between 1 ~ num_of_gpu!")
        self.__NUM_GPU   = value
        self.BATCH_SIZE = self.BATCH_SIZE_PER_GPU * self.__NUM_GPU

    @property
    def TASK(self):
        return self.__TASK
    
    @TASK.setter
    def TASK(self, value):
        if value is None: return
        elif not isinstance(value, str):
            raise ValueError("TASK must be an string!")
        assert value in ["cls", "part_seg"], "Error task, task must in ['cls', 'part_seg']"

        self.__TASK     = value
        self.DATA_PATH = self.TASK_DATA_PATH[value]
        if value == "cls":
            self.NUM_CLASS = 40
            self.NUM_POINT = 1024
            self.BATCH_SIZE_PER_GPU = 96
        elif value == "part_seg":
            self.NUM_CLASS = 16
            self.NUM_PART  = 50
            self.NUM_POINT = 2500
            self.BATCH_SIZE_PER_GPU = 36



