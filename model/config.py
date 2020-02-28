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
        # define as decorator
        self.NUM_POINT          = origin_property("NUM_POINT")
        self.NUM_CLASS          = origin_property("NUM_CLASS")
        self.NUM_WORKER         = origin_property("NUM_WORKER")
        self.EPOCH              = origin_property("EPOCH")
        self.STEP_SIZE          = origin_property("STEP_SIZE")
        self.LEARNING_RATE      = origin_property("LEARNING_RATE")
        self.LOG_DIR            = origin_property("LOG_DIR")
        self.DECAY_RATE         = origin_property("DECAY_RATE")
        self.OPTIMIZER          = origin_property("OPTIMIZER")
        self.MOMENTUM_ORIGINAL  = origin_property("MOMENTUM_ORIGINAL")
        self.MOMENTUM_DECCAY    = origin_property("MOMENTUM_DECCAY")
        self.MOMENTUM_CLIP      = origin_property("MOMENTUM_CLIP")
        self.MODEL              = origin_property("MODEL")
        self.BATCH_SIZE_PER_GPU = origin_property("BATCH_SIZE_PER_GPU")

        # dataset parameters
        self.NUM_POINT  = 1024
        self.NUM_CLASS  = 40
        self.NUM_PART   = 50
        self.NUM_WORKER = 4

        # model parameters
        self.EPOCH           = 200
        self.STEP_SIZE       = 20
        self.LEARNING_RATE   = 1e-3
        self.LOG_DIR         = None
        self.NORMAL          = True
        self.DECAY_RATE      = 1e-4
        self.OPTIMIZER       = "Adam"
        
        self.MOMENTUM_ORIGINAL = 0.1
        self.MOMENTUM_DECCAY   = 0.5
        self.MOMENTUM_CLIP     = 0.01

        # GPU setting
        self.BATCH_SIZE_PER_GPU = 24
        self.__NUM_GPU          = 1
        self.BATCH_SIZE         = 24
        
        self.SEG_LABEL_TO_CAT = {}# {0:Airplane, 1:Airplane, ...49:Table}
        
        # model define
        self.__TASK = "cls"
        self.MODEL = "ponitnet" # pointnet or pointnet2


        self.set_label_to_cat()
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
            if self.MODEL in ["pointnet", "PointNet"]:
                self.MODEL = "PointNet"
                self.BATCH_SIZE_PER_GPU = 96
            if self.MODEL in ["pointnet2", "PointNet2"]:
                self.MODEL = "PointNet2"
                self.BATCH_SIZE_PER_GPU = 24
            self.NUM_CLASS = 40
            self.NUM_POINT = 1024
            self.BATCH_SIZE = self.BATCH_SIZE_PER_GPU * self.NUM_GPU
        elif value == "part_seg":
            if self.MODEL in ["pointnet", "PointNet"]:
                self.MODEL = "PointNet"
                self.BATCH_SIZE_PER_GPU = 48
            if self.MODEL in ["pointnet2", "PointNet2"]:
                self.MODEL = "PointNet2"
                self.BATCH_SIZE_PER_GPU = 12
            self.NUM_CLASS = 16
            self.NUM_PART  = 50
            self.NUM_POINT = 2500
            self.BATCH_SIZE = self.BATCH_SIZE_PER_GPU * self.NUM_GPU

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
        self.BATCH_SIZE = self.BATCH_SIZE_PER_GPU * self.NUM_GPU


