"""
     This .py for CIFAR
"""
import sys
import os
import argparse
import random
import time
import warnings

from model.shufflenet import shufflenet_v2_x0_5

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
warnings.filterwarnings('ignore')

# from cifar100res.apex import amp
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
# from thop import profile
# from thop import clever_format
from torch.utils.data import DataLoader
from config import Config
# from cifar100res.public.distillation.models.resnetforcifar import resnet50, resnet18
from torchvision.datasets import CIFAR10
from public.imagenet.utils import DataPrefetcher, get_logger, AverageMeter, accuracy
from utils import (
    Logger,
    adjust_learning_rate,
    count_parameters,
    # data_augmentation,
    get_current_lr,
    # get_data_loader,
    # load_checkpoint,
    mixup_criterion,
    mixup_data,
    # save_checkpoint,
)
from torchstat import stat


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
    parser.add_argument('--lr_type',
                        type=str,
                        default=Config.type,
                        help='learning rate type')
    parser.add_argument('--base_lr',
                        type=float,
                        default=Config.base_lr,
                        help='learning rate')

    parser.add_argument('--lr_epochs',
                        type=float,
                        default=Config.lr_epochs,
                        help='lr_epochs')
    parser.add_argument('--lr_mults',
                        type=float,
                        default=Config.lr_mults,
                        help='momentum')
    parser.add_argument('--min_lr',
                        type=float,
                        default=Config.min_lr,
                        help='momentum')
    parser.add_argument('--lower_bound',
                        type=float,
                        default=Config.lower_bound,
                        help='lower_bound')
    parser.add_argument('--upper_bound',
                        type=float,
                        default=Config.upper_bound,
                        help='upper_bound')
    parser.add_argument('--weight_decay',
                        type=float,
                        default=Config.weight_decay,
                        help='weight decay')
    parser.add_argument('--nesterov',
                        type=float,
                        default=Config.nesterov,
                        help='gamma')
    parser.add_argument('--momentum',
                        type=list,
                        default=Config.momentum,
                        help='momentum')
    parser.add_argument('--epochs',
                        type=int,
                        default=Config.epochs,
                        help='num of training epochs')
    parser.add_argument('--batch_size',
                        type=int,
                        default=Config.batch_size,
                        help='batch size')
    parser.add_argument('--accumulation_steps',
                        type=int,
                        default=Config.accumulation_steps,
                        help='gradient accumulation steps')
    parser.add_argument('--mixup',
                        type=bool,
                        default=Config.mixup,
                        help='data mixup')
    parser.add_argument('--mixup_alpha',
                        type=int,
                        default=Config.mixup_alpha,
                        help='data mixup')
    parser.add_argument('--pretrained',
                        type=bool,
                        default=Config.pretrained,
                        help='load pretrained model params or not')
    parser.add_argument('--num_classes',
                        type=int,
                        default=Config.num_classes,
                        help='model classification num')
    parser.add_argument('--num_workers',
                        type=int,
                        default=Config.num_workers,
                        help='number of worker to load data')
    parser.add_argument('--resume',
                        type=str,
                        default=Config.resume,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkpoints',
                        type=str,
                        default=Config.checkpoint_path,
                        help='path for saving trained models')
    parser.add_argument('--log',
                        type=str,
                        default=Config.log,
                        help='path to save log')
    parser.add_argument('--evaluate',
                        type=str,
                        default=Config.evaluate,
                        help='path for evaluate model')
    parser.add_argument('--seed', type=int, default=Config.seed, help='seed')
    parser.add_argument('--print_interval',
                        type=bool,
                        default=Config.print_interval,
                        help='print interval')
    parser.add_argument('--apex',
                        type=bool,
                        default=Config.apex,
                        help='use apex or not')

    return parser.parse_args()


def train(train_loader, model, criterion, optimizer, scheduler, epoch, logger,
          args):
    top1 = AverageMeter()
    top5 = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    iters = len(train_loader.dataset) // args.batch_size
    prefetcher = DataPrefetcher(train_loader)
    inputs, labels = prefetcher.next()
    iter_index = 1

    while inputs is not None:
        inputs, labels = inputs.cuda(), labels.cuda()
        if args.mixup:
            inputs, targets_a, targets_b, lam = mixup_data(
                inputs, labels, args.mixup_alpha
            )

            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        outputs = model(inputs)
        # loss = criterion(outputs, labels)
        loss = loss / args.accumulation_steps

        # if args.apex:
        #     with amp.scale_loss(loss, optimizer) as scaled_loss:
        #         scaled_loss.backward()
        # else:
        loss.backward()

        if iter_index % args.accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        # measure accuracy and record loss
        acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
        top1.update(acc1.item(), inputs.size(0))
        top5.update(acc5.item(), inputs.size(0))
        losses.update(loss.item(), inputs.size(0))

        inputs, labels = prefetcher.next()
        # 每30打印一次
        if iter_index % args.print_interval == 0:
            logger.info(
                f"train: epoch {epoch:0>3d}, iter [{iter_index:0>4d}, {iters:0>4d}], lr: {scheduler.get_lr()[0]:.6f}, top1 acc: {acc1.item():.2f}%, top5 acc: {acc5.item():.2f}%, loss_total: {loss.item():.2f}"
            )

        iter_index += 1

    scheduler.step()

    return top1.avg, top5.avg, losses.avg


def validate(val_loader, model, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for inputs, labels in val_loader:
            data_time.update(time.time() - end)
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            top1.update(acc1.item(), inputs.size(0))
            top5.update(acc5.item(), inputs.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

    throughput = 1.0 / (batch_time.avg / inputs.size(0))

    return top1.avg, top5.avg, throughput


def main(logger, args):
    if not torch.cuda.is_available():
        raise Exception("need gpu to train network!")

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True

    gpus = torch.cuda.device_count()
    logger.info(f'use {gpus} gpus')
    logger.info(f"args: {args}")

    cudnn.benchmark = True
    cudnn.enabled = True
    start_time = time.time()

    # dataset and dataloader
    logger.info('start loading data')

    train_dataset = CIFAR10(**Config.train_dataset_init)
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True)
    val_dataset = CIFAR10(**Config.val_dataset_init)
    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            pin_memory=True)
    logger.info('finish loading data')

    logger.info(f"creating model")

    model = shufflenet_v2_x0_5()

    print("params {} ".format(sum(x.numel() for x in model.parameters())))
    # stat(model, (3, 32, 32))
    for name, param in model.named_parameters():
        logger.info(f"{name},{param.requires_grad}")

    model = model.cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    # optimizer = torch.optim.SGD(model.parameters(),
    #                             args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.base_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=args.nesterov,
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.lr_epochs, gamma=1)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=args.gamma, last_epoch=-1)

    # if args.apex:
    #     model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    # model = nn.DataParallel(model)

    if args.evaluate:
        if not os.path.isfile(args.evaluate):
            raise Exception(
                f"{args.resume} is not a file, please check it again")
        logger.info('start only evaluating')
        logger.info(f"start resuming model from {args.evaluate}")
        checkpoint = torch.load(args.evaluate,
                                map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        acc1, acc5, throughput = validate(val_loader, model, args)
        logger.info(
            f"epoch {checkpoint['epoch']:0>3d}, top1 acc: {acc1:.2f}%, top5 acc: {acc5:.2f}%, throughput: {throughput:.2f}sample/s"
        )

        return

    best_acc = 0.0
    start_epoch = 1
    # resume training
    if os.path.exists(args.resume):
        logger.info(f"start resuming model from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
        start_epoch += checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logger.info(
            f"finish resuming model from {args.resume}, epoch {checkpoint['epoch']}, "
            f"loss: {checkpoint['loss']:3f}, best_acc: {checkpoint['best_acc']:.2f}%, lr: {checkpoint['lr']:.6f}, "
            f"top1_acc: {checkpoint['acc1']}%")

    if not os.path.exists(args.checkpoints):
        os.makedirs(args.checkpoints)

    logger.info('start training')
    for epoch in range(start_epoch, args.epochs + 1):
        acc1, acc5, losses = train(train_loader, model, criterion, optimizer,
                                   scheduler, epoch, logger, args)
        logger.info(
            f"train: epoch {epoch:0>3d}, top1 acc: {acc1:.2f}%, top5 acc: {acc5:.2f}%, losses: {losses:.2f}"
        )

        acc1, acc5, throughput = validate(val_loader, model, args)
        logger.info(
            f"val: epoch {epoch:0>3d}, top1 acc: {acc1:.2f}%, top5 acc: {acc5:.2f}%, throughput: {throughput:.2f}sample/s"
        )
        if acc1 > best_acc:
            # torch.save(
            # # model.module.state_dict(),
            #            os.path.join(args.checkpoints, "best.pth"))
            best_acc = acc1
        # remember best prec@1 and save checkpoint
        torch.save(
            {
                'epoch': epoch,
                'best_acc': best_acc,
                'acc1': acc1,
                'loss': losses,
                'lr': adjust_learning_rate(optimizer, epoch),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            },
            os.path.join(args.checkpoints, '3ddwtlevel2.pth'))

    logger.info(f"finish training, best acc: {best_acc:.2f}%")
    training_time = (time.time() - start_time) / 3600
    logger.info(
        f"finish training, total training time: {training_time:.2f} hours")


if __name__ == '__main__':
    args = parse_args()
    logger = get_logger(__name__, args.log)
    main(logger, args)
