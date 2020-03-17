"""Train the model"""

import os, json, shutil, argparse
import logging, datetime
from tqdm import *
from pathlib import Path
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import lib.utils.data_utils as d_utils
import lib.model.net as net
import lib.model.data_loader as data_loader
from lib.model.data_loader import ModelNetDataLoader, PartNormalDataset
from lib.utils.module_utils import *
from lib.utils import gpu_utils
from lib.config import Config

def parse_args():
    parser = argparse.ArgumentParser(
        description="Arguments for PointNet training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # data parameters
    parser.add_argument(
            "--gpu", dest="gpu",
            type=str, default=None,
            help="specify gpu device [exapmle: 1, 1,2(divide by ',')]"
        )
    # training parameters
    parser.add_argument(
            "--epoch", dest="epoch",
            type=int, default=None,
            help="number of epoch in training [exapmle: 200]"
        )
    parser.add_argument(
            "--step_size", dest="step_size",
            type=int,  default=None,
            help="Decay step for lr decay [example: every 20 epochs]"
        )
    parser.add_argument(
            "--learning_rate", dest="learning_rate",
            type=float, default=None,
            help="learning rate in training [example: 0.001]"
        )
    parser.add_argument(
            "--batch_size", dest="batch_size",
            type=int, default=None,
            help="batch size in training [example: 12]"
        )
    # train component
    parser.add_argument(
            "--model", dest="model",
            type=str, default=None,
            help="model name [pointnet pointnet2]"
        )
    parser.add_argument(
            "--scale", dest="scale",
            type=str, default=None,
            help="scale in PointNet2 [example: msg]"
        )
    parser.add_argument(
            "--task", dest="task",
            type=str, default=None,
            help="task in [example: cls, part_seg]"
        )
    # relative path
    parser.add_argument(
            "--log_dir", dest="log_dir",
            type=str, default=None,
            help="experiment root"
        )
    parser.add_argument(
            "--json_path", dest="json_path",
            type=str, default=None,
            help="use exist training configuration from json_path cover Config"
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
    experiment_dir = experiment_dir.joinpath(cfg.MODEL)
    experiment_dir.mkdir(exist_ok=True)
    if cfg.LOG_DIR is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(cfg.LOG_DIR)
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

    logger.log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    logger.log_string("The number of test data is: %d" % len(TEST_DATASET))
    trainDataLoader = torch.utils.data.DataLoader(
            TRAIN_DATASET, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKER
        )
    testDataLoader  = torch.utils.data.DataLoader(
            TEST_DATASET, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKER
        )
    return trainDataLoader, testDataLoader, weights

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
    num_class  = cfg.NUM_CLASS # useless in "part_seg" task model define
    num_part   = cfg.NUM_PART  # useless in "cls" and "sem_seg" task model define
    normal     = cfg.NORMAL
    num_gpu    = cfg.NUM_GPU
    task       = cfg.TASK
    scale      = cfg.SCALE
    parameters = DataLoader[-1]
    weights = parameters

    model      = getattr(net, cfg.MODEL)
    model      = model(num_class=num_class, num_part=num_part, normal_channel=normal, \
                       task=task, scale=scale).cuda(device_ids[0])
    criterion  = net.get_loss(task=task, weights=weights, model=cfg.MODEL).cuda(device_ids[0])

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
    assert torch.cuda.device_count() >= num_gpu, "Actual GPU is less than you call"
    model     = nn.DataParallel(model, device_ids=device_ids)
    criterion = nn.DataParallel(criterion, device_ids=device_ids)
    
    return [model, criterion, optimizer, start_epoch]


def test(model, loader, logger, cfg=None):
    device = model.output_device
    if cfg.TASK == "cls":
        mean_correct = []
        class_acc    = np.zeros((cfg.NUM_CLASS, 3))
        for batch_id, (points, target) in tqdm(enumerate(loader), total=len(loader), smoothing=0.9):
            target         = target[:, 0]
            points         = points.transpose(2, 1)
            points         = points.cuda(device)
            target         = target.cuda(device)
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
        return [instance_acc, class_acc]
    
    elif cfg.TASK == "part_seg":
        test_metrics        = {}
        total_correct       = 0
        total_seen          = 0
        total_seen_class    = [0 for _ in range(cfg.NUM_PART)]
        total_correct_class = [0 for _ in range(cfg.NUM_PART)]
        shape_ious          = {cat: [] for cat in cfg.SEG_CLASSES.keys()}

        for batch_id, (points, label, target) in tqdm(enumerate(loader), total=len(loader), smoothing=0.9):
            batch_size, npoint, _ = points.size()
            points      = points.transpose(2, 1)
            points      = points.float().cuda(device) # points: [B, 6, npoint]
            label       = label.long().cuda(device)   # label:  [B, 1]
            target      = target.long().cuda(device)  # target: [B, npoint]
            classifier  = model.eval()
            to_cat      = to_categorical(label, cfg.NUM_CLASS).to(device) # to_cat: [B, 1, nclass]
            pred, _     = classifier([points, to_cat]) # pred: [B, npoint, npart]
            pred        = pred.cpu().data.numpy()
            pred_logits = pred # pred_logits: [B, npoint, npart]
            target      = target.cpu().data.numpy()

            for i in range(batch_size):
                cat        = cfg.SEG_LABEL_TO_CAT[target[i, 0]]
                logits     = pred_logits[i, :, :] # logits: [npoint, npart]
                pred[i, :] = np.expand_dims(
                        np.argmax(logits[:, cfg.SEG_CLASSES[cat]], 1) + cfg.SEG_CLASSES[cat][0], axis=-1
                    )
                
            correct        = np.sum(pred == target)
            total_correct += correct
            total_seen    += (batch_size * cfg.NUM_POINT)

            for i in range(cfg.NUM_PART):
                total_seen_class[i]    += np.sum(target == i)
                total_correct_class[i] += np.sum((pred==i) & np.expand_dims((target==i), axis=-1))
            for i in range(batch_size):
                segp      = pred[i, :, 0] # segp: [npoint, npart]
                segl      = target[i, :]  # segl: [npoint]
                cat       = cfg.SEG_LABEL_TO_CAT[segl[0]]
                part_ious = [0.0 for _ in range(len(cfg.SEG_CLASSES[cat]))]
                for l in cfg.SEG_CLASSES[cat]:
                    if np.sum(segl == l) == 0 and \
                       np.sum(segp == l) == 0:  # part is not present, no prediction as well
                        part_ious[l - cfg.SEG_CLASSES[cat][0]] = 1.0
                    else:
                        part_ious[l - cfg.SEG_CLASSES[cat][0]] = \
                            np.sum(
                                    np.sum((segl == l) & (segp == l)) / \
                                    np.sum((segl == l) | (segp == l))
                                )
                                
                shape_ious[cat].append(np.mean(part_ious))

        all_shape_ious = []
        for cat in shape_ious.keys():
            for iou in shape_ious[cat]:
                all_shape_ious.append(iou)
            shape_ious[cat] = np.mean(shape_ious[cat])
        mean_shape_ious                    = np.mean(list(shape_ious.values()))
        test_metrics["accuracy"]           = total_correct / float(total_seen)
        test_metrics['class_avg_accuracy'] = np.mean(
                np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float).reshape((-1, 1, 1))
            )
        for cat in sorted(shape_ious.keys()):
            logger.log_string('eval mIoU of %s %f' % (cat + ' ' * (14 - len(cat)), shape_ious[cat]))
        test_metrics['class_avg_iou']    = mean_shape_ious
        test_metrics['instance_avg_iou'] = np.mean(all_shape_ious)

        return test_metrics


def train(DataLoader, ModelList, logger, checkpoints_dir, cfg):
    """Train the model on `num_steps` batches
    Args:
        DataLoader: contains trainloader and testloader
        ModelList:  contains model, criterion, optimizer and start_epoch
        logger:     logging context
    """
    # set model loading
    model, criterion, optimizer, start_epoch = ModelList
    device = model.output_device
    # set dataset loading
    trainDataLoader, testDataLoader, weights = DataLoader
    # set scheduler
    scheduler         = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.STEP_SIZE, gamma=cfg.DECAY_RATE)

    current_epoch         = 0
    global_step           = 0
    # label will be "acc" in "cls" task and "avg_iou" in "part_seg" task
    best_instance_label   = 0.0
    best_class_label      = 0.0
    mean_correct      = []

    # start training
    logger.log_string("Start training...")
    
    for epoch in range(start_epoch, cfg.EPOCH):
        # laerning rate step
        scheduler.step()
        # batch norm momentum step
        momentum = max(cfg.MOMENTUM_ORIGINAL * (cfg.MOMENTUM_DECCAY ** (epoch//cfg.STEP_SIZE)), \
                       cfg.MOMENTUM_CLIP)
        logger.log_string("Epoch {} ({}/{}):    BN momentum updated to: {}".format(
                current_epoch + 1, epoch + 1, cfg.EPOCH, momentum)
            )
        model = model.apply(lambda x: bn_momentum_adjust(x,momentum))
        with tqdm(total=len(trainDataLoader), smoothing=0.9, ncols=120) as t:
            for batch_id, data in enumerate(trainDataLoader):
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
    
                points = points.cuda(device)
                target = target.cuda(device)
                if cfg.TASK == "part_seg": label = label.cuda(device)
                optimizer.zero_grad()
    
                classifier = model.train()
                if cfg.TASK == "cls":
                    pred, trans_feat = classifier(points)
                elif cfg.TASK == "part_seg":
                    to_cat           = to_categorical(label, cfg.NUM_CLASS).to(device)
                    pred, trans_feat = classifier([points, to_cat])
                    pred             = pred.contiguous().view(-1, cfg.NUM_PART)

                target = target.view(-1, 1)[:, 0]
                loss   = torch.sum(criterion(pred, target.long(), trans_feat))
                t.set_postfix(loss='{:^7.6f}'.format(loss))
                t.update()
                pred_choice = pred.data.max(1)[1]
                if target.device != pred_choice.device:
                    pred_choice.to(target.device)
                
                correct = pred_choice.eq(target.long().data).cpu().sum()
                if cfg.TASK == "cls":
                    mean_correct.append(correct.item() / cfg.BATCH_SIZE)
                elif cfg.TASK == "part_seg":
                    mean_correct.append(correct.item() / (cfg.BATCH_SIZE*cfg.NUM_POINT))
    
                loss.backward()
                optimizer.step()
                global_step += 1
    
        # epoch summary
        train_instance_acc = np.mean(mean_correct)
        logger.log_string("Train Instance Accuracy: {}".format(train_instance_acc))

        with torch.no_grad():
            # start test
            test_output = test(model, testDataLoader, logger, cfg=cfg)
            if cfg.TASK == "cls":
                instance_acc, class_acc = test_output
                class_label    = class_acc
                instance_label = instance_acc

            elif cfg.TASK == "part_seg":
                test_metrics   = test_output
                class_label    = test_metrics['class_avg_iou']
                instance_label = test_metrics['instance_avg_iou']
                
            # change best label
            if class_label > best_class_label:
                best_class_label = class_label
            if instance_label > best_instance_label:
                best_instance_label = instance_label

            show_index     = cfg.SHOW_INDEX[cfg.TASK]
            logger.log_string("Test Instance {}: {}, Class {}: {}".format(
                        show_index, instance_label, show_index, class_label
                    )
                )
            logger.log_string("Best Instance {}: {}, Class {}: {}".format(
                        show_index, best_instance_label, show_index, best_class_label
                    )
                )

            if(instance_label >= best_instance_label):
                logger.log_string("Save model...")
                savepath = str(checkpoints_dir) + "/best_model.pth"
                logger.log_string('Saving at %s'% savepath)
                state = {
                    "epoch":                epoch+1,
                    "train_acc":            train_instance_acc,
                    "class_avg_label":      class_label,
                    "instance_avg_label":   instance_label,
                    "model_state_dict":     classifier.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }

            torch.save(state, savepath)
            current_epoch += 1
            
        
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
    NUM_GPU = 1


def update_cfg_by_args(args, cfg):
    try:
        with open(args.json_path, "r") as f:
            config = json.load(f)
        for key, value in config.items():
            cfg.update(key, value)
        device_ids  = [int(i) for i in cfg.GPU_LIST.split(",")]
        gpu_list   = [str(i) for i in device_ids]
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPU_LIST
    except:
        cfg.update("EPOCH",         args.epoch)
        cfg.update("MODEL",         args.model)
        cfg.update("LOG_DIR",       args.log_dir)
        cfg.update("STEP_SIZE",     args.step_size)
        cfg.update("BATCH_SIZE",    args.batch_size)
        cfg.update("LEARNING_RATE", args.learning_rate)
    
        # update gpu
        if torch.cuda.is_available(): # use GPU if available
            if args.gpu == None:
                device_ids = gpu_utils.supervise_gpu(nb_gpu=cfg.NUM_GPU)
                gpu_list   = [str(i) for i in device_ids]
            else:
                gpu_list    = args.gpu
                gpu_list    = [i for i in gpu_list.split(",")]
                device_ids  = [int(i) for i in gpu_list]

            cfg.GPU_LIST = ",".join(gpu_list)
            os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPU_LIST
        
    cfg.TASK  = args.task
    cfg.display()

    print("set CUDA_VISIBLE_DEVICES: {}\n".format(",".join(gpu_list)))
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
