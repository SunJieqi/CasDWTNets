import os
import sys
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

import torchvision.transforms as transforms

class Config:
    log = "./log"  # Path to save log
    checkpoint_path = "./checkpoints"  # Path to store model
    resume = "./checkpoints/latest.pth"
    evaluate = None  # 测试模型，evaluate为模型地址

    pretrained = False
    seed = 0
    num_classes = 10

    # type: STEP or COSINE or HTD
    type = "STEP"
    base_lr = 0.1
    # only for STEP
    lr_epochs = [60, 90]
    lr_mults = 0.1
    # for HTD and COSINE
    min_lr = 0.0
    # only for HTD
    lower_bound = -6.0
    upper_bound = 3.0


    epochs = 100
    batch_size = 128
    accumulation_steps = 1

    nesterov = True
    momentum = 0.9
    weight_decay = 0.0001
    num_workers = 4
    print_interval = 600
    apex = False
    mixup = False
    mixup_alpha = 0.4

    train_transform = transforms.Compose([
        transforms.Pad(4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    train_dataset_init = {
        "root": "cifar",
        "train": True,
        "download": True,
        "transform": train_transform
    }
    val_dataset_init = {
        "root": "cifar",
        "train": False,
        "download": True,
        "transform": val_transform
    }
