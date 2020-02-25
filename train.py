"""Train the model"""

import os, shutil, argparse
import logging, datetime
from tqdm import *
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable

import utils.data_utils as d_utils
import model.net as net
import model.data_loader as data_loader
from model.config import Config
from model.data_loader import ModelNetDataLoader
from utils.train_utils import bn_momentum_adjust

def parse_args():
    parser = argparse.ArgumentParser(
        description="Arguments for PointNet training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # data parameters
    parser.add_argument(
            "--gpu", dest="gpu",
            type=str, default="0",
            help="specify gpu device [default: 0]"
        )
    # training parameters
    parser.add_argument(
            "--epoch", dest="epoch",
            type=int, default=200,
            help="number of epoch in training [default: 200]"
        )
    # train component
    parser.add_argument(
            "--model", dest="model",
            type=str, default="pointnet_cls",
            help="model name [default: pointnet_cls]"
        )
    parser.add_argument(
            "--task", dest="task",
            type=str, default="cls",
            help="model name [default: pointnet_cls]"
        )
    # relative path
    parser.add_argument(
            "--log_dir", dest="log_dir",
            type=str, default=None,
            help="experiment root"
        )
    parser.add_argument(
            "--data_dir", dest="data_dir",
            type=str, default="data/modelnet40_normal_resampled/",
            help="dataset root"
        )
    return parser.parse_args()


class InfoLogger(object):
    def __init__(self, logger):
        super().__init__()
        self.logger = logger
    def log_string(self, str):
        self.logger.info(str)
        print(str)

def set_dir(args):
    timestr        = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
    experiment_dir = Path("./log/")
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath(args.task)
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath("checkpoints/")
    log_dir         = experiment_dir.joinpath("logs/")
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)
    return log_dir, checkpoints_dir

def set_logger(log_dir, args):
    logger       = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter    = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler("%s/%s.txt" % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger = InfoLogger(logger)
    logger.log_string("PARAMETER ...")
    logger.log_string(args)
    return logger

def set_data_loader(logger, args, cfg):
    data_path     = args.data_dir
    task          = args.task
    num_point     = cfg.NUM_POINT
    normal        = cfg.NORMAL
    if task == "cls":
        TRAIN_DATASET = ModelNetDataLoader(root=data_path, npoint=cfg.NUM_POINT, split="train",
                                        normal_channel=normal)
        TEST_DATASET  = ModelNetDataLoader(root=data_path, npoint=cfg.NUM_POINT, split="test",
                                        normal_channel=normal)
        weights = None
    elif task == "sem_seg":
        TRAIN_DATASET = ModelNetDataLoader(root=data_path, npoint=cfg.NUM_POINT, split="train",
                                        normal_channel=normal)
        TEST_DATASET  = ModelNetDataLoader(root=data_path, npoint=cfg.NUM_POINT, split="test",
                                        normal_channel=normal)
        weights = torch.Tensor(TRAIN_DATASET.labelweights).cuda()
    
    logger.log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    logger.log_string("The number of test data is: %d" % len(TEST_DATASET))
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=4)
    testDataLoader  = torch.utils.data.DataLoader(TEST_DATASET, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=4)
    return trainDataLoader, testDataLoader, weights

def set_model(DataLoader, args, cfg):
    num_class = cfg.NUM_CLASS
    normal    = cfg.NORMAL
    task      = args.task
    weights   = DataLoader[-1]

    model      = net.PointNet(num_class, normal_channel=normal, task=task).cuda()
    criterion  = net.get_loss(task=task, weights=weights).cuda()

    try:
        checkpoint  = torch.load("./checkpoints/best_model.pth")
        start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.log_string("Use pretrain model")
    except:
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
    return [model, criterion, optimizer, start_epoch, num_class]


def test(model, loader, num_class=40):
    mean_correct = []
    class_acc    = np.zeros((num_class, 3))
    for j, data in tqdm(enumerate(loader), total=len(loader)):
        points, target = data
        target         = target[:, 0]
        points         = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        classifier     = model.eval()
        pred, _        = classifier(points)
        pred_choice    = pred.data.max(1)[1]
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

def train(DataLoader, ModelList, logger, checkpoints_dir, args, cfg):
    """Train the model on `num_steps` batches
    Args:
        DataLoader: contains trainloader and testloader
        ModelList:  contains model, criterion, optimizer and start_epoch
        logger:     logging context
        args:       contains argmentations we input
    """
    # set model loading
    model, criterion, optimizer, start_epoch, num_class = ModelList
    # set dataset loading
    trainDataLoader, testDataLoader, weights = DataLoader
    # set scheduler
    scheduler         = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    global_epoch      = 0
    global_step       = 0
    best_instance_acc = 0.0
    best_class_acc    = 0.0
    mean_correct      = []

    # start training
    logger.log_string("Start training...")
    for epoch in range(start_epoch,args.epoch):
        logger.log_string("Epoch %d (%d/%s):" % (global_epoch + 1, epoch + 1, args.epoch))

        scheduler.step()
        for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            points, target   = data
            points           = points.data.numpy()
            points           = d_utils.random_point_dropout(points)
            points[:,:, 0:3] = d_utils.random_scale_point_cloud(points[:,:, 0:3])
            points[:,:, 0:3] = d_utils.shift_point_cloud(points[:,:, 0:3])
            points           = torch.Tensor(points)
            target           = target[:, 0]

            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            optimizer.zero_grad()

            classifier = model.train()
            pred, trans_feat = model(points)
            loss = criterion(pred, target.long(), trans_feat)
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.long().data).cpu().sum()
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
    NUM_POINT = 1024
    NUM_CLASS = 40

    # model parameters
    NORMAL        = True
    BATCH_SIZE    = 24
    LEARNING_RATE = 1e-3
    DECAY_RATE    = 1e-4
    OPTIMIZER     = "Adam"


if __name__ == "__main__":
    args = parse_args()
    cfg  = TrainConfig()
    print("Called with args:")
    print(args)
    cfg.display()

    # use GPU if available
    if torch.cuda.is_available(): 
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    # set the relative dir
    log_dir, checkpoints_dir = set_dir(args)

    # set the logger
    logger = set_logger(log_dir, args)

    # set dataloading
    logger.log_string("Load dataset model and optimizer ...")
    DataLoader = set_data_loader(logger, args, cfg)
    ModelList  = set_model(DataLoader, args, cfg)

    # start traing
    train(DataLoader, ModelList, logger, checkpoints_dir, args, cfg)
