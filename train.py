"""Train the model"""

import os, shutil, argparse
import logging, datetime
from tqdm import *
from pathlib import Path
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import utils.data_utils as d_utils
import model.net as net
import model.data_loader as data_loader
from model.config import Config
from model.data_loader import ModelNetDataLoader, PartNormalDataset
from utils.model_utils import bn_momentum_adjust
from utils import gpu_utils

def parse_args():
    parser = argparse.ArgumentParser(
        description="Arguments for PointNet training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # data parameters
    parser.add_argument(
            "--gpu", dest="gpu",
            type=str, default=None,
            help="specify gpu device [default: None] type:1, 1,2(divide by ',')"
        )
    # training parameters
    parser.add_argument(
            "--epoch", dest="epoch",
            type=int, default=200,
            help="number of epoch in training [default: 200]"
        )
    parser.add_argument(
            "--step_size", dest="step_size",
            type=int,  default=20,
            help="Decay step for lr decay [default: every 20 epochs]"
        )
    parser.add_argument(
            "--learning_rate", dest="learning_rate",
            type=float, default=0.001,
            help="learning rate in training [default: 0.001]"
        )
    # train component
    parser.add_argument(
            "--model", dest="model",
            type=str, default="pointnet_cls",
            help="model name [default: pointnet]"
        )
    parser.add_argument(
            "--task", dest="task",
            type=str, default=None,
            help="model name [default: pointnet_cls]"
        )
    # relative path
    parser.add_argument(
            "--log_dir", dest="log_dir",
            type=str, default=None,
            help="experiment root"
        )
    return parser.parse_args()


class InfoLogger(object):
    def __init__(self, logger):
        super().__init__()
        self.logger = logger
    def log_string(self, str):
        self.logger.info(str)
        print(str)

def set_dir(cfg):
    timestr        = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
    experiment_dir = Path("./log/")
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath(cfg.TASK)
    experiment_dir.mkdir(exist_ok=True)
    if cfg.LOG_DIR is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(cfg.LOG_DIR)
        print("use experiment_dir: ", experiment_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath("checkpoints/")
    log_dir         = experiment_dir.joinpath("logs/")
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)
    return log_dir, checkpoints_dir

def set_logger(log_dir, cfg):
    logger       = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter    = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler("{}/{}_{}.txt".format(log_dir, cfg.MODEL, cfg.TASK))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger = InfoLogger(logger)
    logger.log_string("PARAMETER ...")
    return logger

def set_data_loader(logger, cfg):
    task      = cfg.TASK
    num_point = cfg.NUM_POINT
    normal    = cfg.NORMAL
    data_path = cfg.DATA_PATH
    
    # define default value
    weights     = None
    num_classes = None
    num_part    = None

    if task == "cls":
        TRAIN_DATASET = ModelNetDataLoader(root=data_path, npoint=cfg.NUM_POINT, split="train",
                                        normal_channel=normal)
        TEST_DATASET  = ModelNetDataLoader(root=data_path, npoint=cfg.NUM_POINT, split="test",
                                        normal_channel=normal)
        weights = None
    elif task == "part_seg":
        TRAIN_DATASET = PartNormalDataset(root=data_path, npoint=cfg.NUM_POINT, split="trainval",
                                          normal_channel=normal)
        TEST_DATASET  = PartNormalDataset(root=data_path, npoint=cfg.NUM_POINT, split="test",
                                          normal_channel=normal)
        num_classes = 16
        num_part    = 50

    
    logger.log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    logger.log_string("The number of test data is: %d" % len(TEST_DATASET))
    trainDataLoader = torch.utils.data.DataLoader(
            TRAIN_DATASET, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKER
        )
    testDataLoader  = torch.utils.data.DataLoader(
            TEST_DATASET, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKER
        )
    return trainDataLoader, testDataLoader, [weights, num_classes, num_part]

def _check_module_prefix(model_state_dict):
    """ nn.Dataparallel will output model_state_dict whose keys
    with '.module' prefix, it will fail to load this model_sta-
    te_dict. So we should check the prefix and remove it.
    """
    temp = OrderedDict()
    for key in model_state_dict.keys():
        if key[:6] == "module":
            temp[key[7:]] = model_state_dict[key]
        else:
            temp[key] = model_state_dict[key]
    return temp

def set_model(DataLoader, device_ids, checkpoints_dir, cfg):
    num_class  = cfg.NUM_CLASS
    normal     = cfg.NORMAL
    num_gpu    = cfg.NUM_GPU
    task       = cfg.TASK
    parameters = DataLoader[-1]
    weights, num_classes, num_part = parameters

    model      = net.PointNet(num_class, normal_channel=normal, task=task).cuda(device_ids[0])
    criterion  = net.get_loss(task=task, weights=weights).cuda(device_ids[0])

    if os.path.exists(os.path.join(checkpoints_dir, "best_model.pth")):
        checkpoint       = torch.load(os.path.join(checkpoints_dir, "best_model.pth"))
        start_epoch      = checkpoint["epoch"]
        model_state_dict = _check_module_prefix(checkpoint["model_state_dict"])
        model.load_state_dict(model_state_dict)
        logger.log_string("Use pretrain model, start from epoch: {}".format(start_epoch))
    else:
        logger.log_string("No existing model, starting training from scratch...")
        start_epoch = 0


    # set optimizer
    if cfg.OPTIMIZER == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr           = cfg.LEARNING_RATE,
            betas        = (0.9, 0.999),
            eps          = 1e-08,
            weight_decay = cfg.DECAY_RATE
        )
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # if num_gpu > 1:
    logger.log_string("Use {} GPU".format(num_gpu))
    assert torch.cuda.device_count() > num_gpu
    model     = nn.DataParallel(model, device_ids=device_ids)
    criterion = nn.DataParallel(criterion, device_ids=device_ids)
    
    return [model, criterion, optimizer, start_epoch, num_class]


def test(model, loader, num_class=40):
    mean_correct = []
    class_acc    = np.zeros((num_class, 3))
    for j, data in tqdm(enumerate(loader), total=len(loader)):
        points, target = data
        target         = target[:, 0]
        points         = points.transpose(2, 1)
        points         = points.cuda()
        classifier     = model.eval()
        pred, _        = classifier(points)
        pred_choice    = pred.data.max(1)[1]
        target         = target.cuda(pred_choice.device)
        for cat in np.unique(target.cpu()):
            classacc          = pred_choice[target==cat].eq(target[target==cat].long().data).cpu().sum()
            class_acc[cat,0] += classacc.item()/float(points[target==cat].size()[0])
            class_acc[cat,1] +=1
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item()/float(points.size()[0]))
    class_acc[:,2] = class_acc[:,0]/ class_acc[:,1]
    class_acc      = np.mean(class_acc[:,2])
    instance_acc   = np.mean(mean_correct)
    return instance_acc, class_acc

def train(DataLoader, ModelList, logger, checkpoints_dir, cfg):
    """Train the model on `num_steps` batches
    Args:
        DataLoader: contains trainloader and testloader
        ModelList:  contains model, criterion, optimizer and start_epoch
        logger:     logging context
    """
    # set model loading
    model, criterion, optimizer, start_epoch, num_class = ModelList
    # set dataset loading
    trainDataLoader, testDataLoader, weights = DataLoader
    # set scheduler
    scheduler         = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.STEP_SIZE, gamma=cfg.DECAY_RATE)

    global_epoch      = 0
    global_step       = 0
    best_instance_acc = 0.0
    best_class_acc    = 0.0
    mean_correct      = []

    # start training
    logger.log_string("Start training...")
    for epoch in range(start_epoch, cfg.EPOCH):
        logger.log_string("Epoch %d (%d/%s):" % (global_epoch + 1, epoch + 1, cfg.EPOCH))
        # laerning rate step
        scheduler.step()
        # batch norm momentum step
        momentum = cfg.MOMENTUM_ORIGINAL * (cfg.MOMENTUM_DECCAY ** (epoch//cfg.STEP_SIZE))
        for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            if cfg.TASK == "cls":
                points, target = data
                points         = points.data.numpy()
                points         = d_utils.random_point_dropout(points)
            elif cfg.TASK == "part_seg":
                points, label, target = data
                points                = points.data.numpy()
            
            points[:,:, 0:3] = d_utils.random_scale_point_cloud(points[:,:, 0:3])
            points[:,:, 0:3] = d_utils.shift_point_cloud(points[:,:, 0:3])
            points           = torch.Tensor(points)
            points           = points.transpose(2, 1)
            target           = target[:, 0]

            points = points.cuda(model.output_device)
            target = target.cuda(model.output_device)
            if cfg.TASK == "part_seg": label = label.cuda(model.output_device)
            optimizer.zero_grad()

            classifier       = model.train()
            if cfg.TASK == "cls":
                pred, trans_feat = model(points)
            elif cfg.TASK == "part_seg":
                seg_pred, trans_feat = model(points)
            loss             = torch.sum(criterion(pred, target.long(), trans_feat))
            pred_choice      = pred.data.max(1)[1]
            if target.device != pred_choice.device:
                pred_choice.to(target.device)
            correct          = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
            loss.backward()
            optimizer.step()
            global_step += 1

        train_instance_acc = np.mean(mean_correct)
        logger.log_string("Train Instance Accuracy: %f" % train_instance_acc)

        with torch.no_grad():
            instance_acc, class_acc = test(classifier.eval(), testDataLoader, num_class=num_class)

            if (instance_acc >= best_instance_acc):
                best_instance_acc = instance_acc
                best_epoch = epoch + 1

            if (class_acc >= best_class_acc):
                best_class_acc = class_acc
            logger.log_string("Test Instance Accuracy: %f, Class Accuracy: %f"% (instance_acc, class_acc))
            logger.log_string("Best Instance Accuracy: %f, Class Accuracy: %f"% (best_instance_acc, best_class_acc))

            if (instance_acc >= best_instance_acc):
                logger.log_string("Save model...")
                savepath = str(checkpoints_dir) + "/best_model.pth"
                logger.log_string("Saving at %s"% savepath)
                state = {
                    "epoch": best_epoch,
                    "instance_acc": instance_acc,
                    "class_acc": class_acc,
                    "model_state_dict": classifier.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }
                torch.save(state, savepath)
            global_epoch += 1

    logger.log_string("End of training...")


class TrainConfig(Config):
    # dataset parameters
    NUM_POINT  = 1024
    NUM_CLASS  = 40
    NUM_WORKER = 4

    # model parameters
    LEARNING_RATE = 1e-3
    DECAY_RATE    = 0.7
    OPTIMIZER     = "Adam"

    # GPU setting
    BATCH_SIZE_PER_GPU = 96
    NUM_GPU = 1


def update_cfg_by_args(args, cfg):
    # update gpu
    if torch.cuda.is_available(): # use GPU if available
        if args.gpu == None:
            device_ids = gpu_utils.supervise_gpu(nb_gpu=cfg.NUM_GPU)
            gpu_list   = [str(i) for i in device_ids]
        else:
            gpu_list    = args.gpu
            gpu_list    = [i for i in gpu_list.split(",")]
            device_ids  = [int(i) for i in gpu_list]
            cfg.NUM_GPU = len(device_ids)
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_list)
        print("set CUDA_VISIBLE_DEVICES: ", ",".join(gpu_list))
    
    cfg.EPOCH         = args.epoch
    cfg.STEP_SIZE     = args.step_size
    cfg.LEARNING_RATE = args.learning_rate

    cfg.TASK    = args.task
    cfg.MODEL   = args.model
    cfg.LOG_DIR = args.log_dir
    
    cfg.display()
    return device_ids



if __name__ == "__main__":
    args = parse_args()
    cfg  = TrainConfig()
    print("Called with args:")
    print(args)

    # update cfg according to args
    device_ids = update_cfg_by_args(args, cfg)

    # set the relative dir
    log_dir, checkpoints_dir = set_dir(cfg)

    # set the logger
    logger = set_logger(log_dir, cfg)

    # set dataloading
    logger.log_string("Load dataset model and optimizer ...")
    DataLoader = set_data_loader(logger, cfg)
    ModelList  = set_model(DataLoader, device_ids, checkpoints_dir, cfg)

    # start traing
    train(DataLoader, ModelList, logger, checkpoints_dir, cfg)
