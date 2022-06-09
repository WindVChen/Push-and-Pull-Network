import argparse
import math
import os
import shutil
import time

from pathlib import Path
import glob
import re

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F

import utils.PPNetBulider
import utils.customTransform

import wandb
import tqdm

import random
import numpy as np

accMin = 0

parser = argparse.ArgumentParser()
parser.add_argument('--data', metavar='DIR', default=r'G:\Dataset\FGSC-23',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 512), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--dim', default=2048, type=int,
                    help='feature dimension (default: 2048)')
parser.add_argument('--pred-dim', default=512, type=int,
                    help='hidden dimension of the predictor (default: 512)')
parser.add_argument('--num_classes', default=23, type=int,
                    help='number of subclasses (for FGSC-23, it should be set to 23)')
parser.add_argument('--proxy_per_cls', default=3, type=int,
                    help='number of proxies per subclass (for FGSC-23, the optimal is 3)')
parser.add_argument('--pretrained', action='store_true')
parser.add_argument('--baseline', action='store_true')
parser.add_argument('--test', action='store_true')
parser.add_argument('--test_model_path', default=None, type=str)
parser.add_argument('--fixed_seed', action='store_true')


def seed_torch(seed=0):
    print("Seed Fixed!")
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _init_fn(worker_id):
    np.random.seed(0)


def main():
    args = parser.parse_args()
    if args.fixed_seed:
        seed_torch()
    if not args.test:
        args.save_dir = increment_path("./model/exp", exist_ok=False)  # increment run
        print("Model saved in " + args.save_dir)
        if not os.path.exists(args.save_dir):
            os.mkdir(args.save_dir)
    wandb_run = wandb.init(config=args, project="YourProject",
                           name="YourName", id=None)
    main_worker(args)


def main_worker(args):
    model = utils.PPNetBulider.PPNet(args.arch, args.dim, args.pred_dim, args.pretrained, args.baseline,
                                     args.num_classes, args.proxy_per_cls)

    init_lr = args.lr

    model = model.cuda()

    print(model)

    similar_criterion = nn.CosineSimilarity(dim=1).cuda()
    classify_criterion = nn.CrossEntropyLoss().cuda()

    if not args.baseline:
        optim_params = [{'params': model.encoder.parameters(), 'fix_lr': False},
                        {'params': model.avg_push.parameters(), 'fix_lr': False, 'custom': True},
                        {'params': model.max_push.parameters(), 'fix_lr': False, 'custom': True},
                        {'params': model.pull.parameters(), 'fix_lr': False, 'custom': True},
                        {'params': model.prototype.parameters(), 'fix_lr': False, 'custom': True, 'big': True},
                        {'params': model.classify_fc.parameters(), 'fix_lr': False, 'custom': True}]
    else:
        optim_params = [{'params': model.encoder.parameters(), 'fix_lr': False, 'custom': False},
                        {'params': model.classify_fc.parameters(), 'fix_lr': False, 'custom': True}]

    optimizer = torch.optim.SGD(optim_params, init_lr, momentum=args.momentum, weight_decay=args.weight_decay)

    cudnn.benchmark = True if not args.fixed_seed else False

    traindir = os.path.join(args.data, 'train')
    validdir = os.path.join(args.data, 'valid')
    testdir = os.path.join(args.data, 'test')

    train_augmentation = utils.customTransform.customAugmentation()
    valid_augmentation = transforms.Compose([utils.customTransform.resize(), transforms.ToTensor()])
    test_augmentation = transforms.Compose([utils.customTransform.resize(), transforms.ToTensor()])

    train_dataset = datasets.ImageFolder(
        traindir,
        utils.PPNetBulider.TwoCropsTransform(train_augmentation))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True, worker_init_fn=_init_fn if args.fixed_seed else None)

    valid_dataset = datasets.ImageFolder(
        validdir,
        valid_augmentation)

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    test_dataset = datasets.ImageFolder(
        testdir,
        test_augmentation)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    if not args.test:
        for epoch in range(args.start_epoch, args.epochs):
            adjust_learning_rate(optimizer, init_lr, epoch, args)
            for param_group in optimizer.param_groups:
                if 'custom' in param_group and param_group['custom']:
                    if 'big' in param_group:
                        wandb.log({'lr/prototype': param_group['lr']})
                    else:
                        wandb.log({'lr/custom': param_group['lr']})
                else:
                    wandb.log({'lr/classifier': param_group['lr']})

            # train for one epoch
            train(train_loader, model, similar_criterion, classify_criterion, optimizer, epoch, args)

            valid(valid_loader, model, classify_criterion, args)

    test(test_loader, model, args)


def train(train_loader, model, similar_criterion, classify_criterion, optimizer, epoch, args):
    print("Starting Train**********")
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    if not args.baseline:
        avgPushlossLog = AverageMeter('avgPushLoss', ':.4f')
        maxPushlossLog = AverageMeter('maxPushLoss', ':.4f')
        pulllossLog = AverageMeter('pullLoss', ':.4f')
        lossClassifyLog = AverageMeter('classifyLoss', ':.4f')
        progress = ProgressMeter(
            len(train_loader),
            [batch_time, data_time, losses, avgPushlossLog, maxPushlossLog, pulllossLog, lossClassifyLog],
            prefix="Epoch: [{}]".format(epoch))
    else:
        progress = ProgressMeter(
            len(train_loader),
            [batch_time, data_time, losses],
            prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()

    allAcc = 0
    avgAcc = 0
    classesCal = [0 for x in range(args.num_classes)]
    correctCal = [0 for x in range(args.num_classes)]
    for step, (images, label) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        label = label.cuda()
        images[0] = images[0].cuda(non_blocking=True)
        images[1] = images[1].cuda(non_blocking=True)

        if not args.baseline:
            # compute output and loss
            t1, t2, m1, m2, a1, a2, m_d, a_d, pt = model(x1=images[0], x2=images[1], baseline=args.baseline)
            avgPushloss = -(similar_criterion(a1[0], a_d[1]).mean() + similar_criterion(a2[0], a_d[0]).mean()) * 0.5 \
                          + (similar_criterion(a1[1], a_d[1]).mean() + similar_criterion(a2[1], a_d[0]).mean()) * 0.5
            maxPushloss = -(similar_criterion(m1[0], m_d[1]).mean() + similar_criterion(m2[0], m_d[0]).mean()) * 0.5 \
                          + (similar_criterion(m1[1], m_d[1]).mean() + similar_criterion(m2[1], m_d[0]).mean()) * 0.5
            pullLoss = (lossPull(pt[0], pt[1], label, args) + lossPull(pt[0], pt[2], label, args)) * 0.5
            lossClassify = (classify_criterion(t1, label) + classify_criterion(t2, label)) * 0.5
            loss = lossClassify + avgPushloss + maxPushloss + pullLoss
        else:
            t1, _, _, _, _, _, _, _ = model(x1=images[0], x2=None, baseline=args.baseline)
            loss = classify_criterion(t1, label)

        _, predict = torch.max(t1, 1)
        allAcc += (predict == label).sum().item()
        for i, j in enumerate(label):
            classesCal[j.item()] += 1
            if (predict[i] == label[i]):
                correctCal[j.item()] += 1

        losses.update(loss.item(), images[0].size(0))
        if not args.baseline:
            avgPushlossLog.update(avgPushloss.item(), images[0].size(0))
            maxPushlossLog.update(maxPushloss.item(), images[0].size(0))
            pulllossLog.update(pullLoss.item(), images[0].size(0))
            lossClassifyLog.update(lossClassify.item(), images[0].size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if step % args.print_freq == 0:
            progress.display(step)
    for i in range(args.num_classes):
        avgAcc += correctCal[i] / classesCal[i]
    avgAcc /= args.num_classes
    if not args.baseline:
        wandb.log({"Loss/TrainLoss": losses.avg, "Loss/avgPushLoss": avgPushlossLog.avg,
                   "Loss/maxPushLoss": maxPushlossLog.avg, "Loss/pullLoss": pulllossLog.avg,
                   "Loss/classifyLoss": lossClassifyLog.avg, 'AvgAcc/TrainAcc': avgAcc,
                   'AllAcc/TrainAcc': allAcc / len(train_loader.dataset)})
    else:
        wandb.log({"Loss/TrainLoss": losses.avg, 'AvgAcc/TrainAcc': avgAcc,
                   'AllAcc/TrainAcc': allAcc / len(train_loader.dataset)})


def valid(valid_loader, model, classify_criterion, args):
    global accMin
    print("Starting Valid********")
    classesCal = [0 for x in range(args.num_classes)]
    correctCal = [0 for x in range(args.num_classes)]
    model.eval()
    accNum = 0
    accAll = 0
    lossAll = 0
    for image, label in tqdm.tqdm(valid_loader):
        image = image.cuda()
        label = label.cuda()
        if not args.baseline:
            with torch.no_grad():
                t1, t2, _, _, _, _, _, _, _ = model(image, image, args.baseline)
            loss = (classify_criterion(t1, label) + classify_criterion(t2, label)) * 0.5
        else:
            with torch.no_grad():
                t1, _, _, _, _, _, _, _ = model(image, None, args.baseline)
            loss = classify_criterion(t1, label)
        _, prediction = torch.max(t1, 1)
        accAll += (prediction == label).sum().item()
        for i, j in enumerate(label):
            classesCal[j.item()] += 1
            if (prediction[i] == label[i]):
                correctCal[j.item()] += 1
        lossAll += loss.item() * image.size(0)
    for i in range(args.num_classes):
        if (classesCal[i]):
            accNum += correctCal[i] / classesCal[i]
            print("class {} \t acc: {}".format(i, correctCal[i] / classesCal[i]))
    accNum /= args.num_classes
    print("lossAvg: {} \t avgAcc: {} \t allAcc: {}".format(lossAll / len(valid_loader.dataset), accNum,
                                                           accAll / len(valid_loader.dataset)))

    wandb.log({"Loss/ValLoss": lossAll / len(valid_loader.dataset), "AvgAcc/ValAcc": accNum,
               "AllAcc/ValAcc": accAll / len(valid_loader.dataset)})
    if (accNum > accMin):
        print("Save Model To {}".format(args.save_dir + "/bestModel.pth"))
        torch.save(model.state_dict(), args.save_dir + "/bestModel.pth")
        accMin = accNum
    else:
        torch.save(model.state_dict(), args.save_dir + "/lastModel.pth")


def test(test_loader, model, args):
    print("Starting test********")
    classesCal = [0 for x in range(args.num_classes)]
    correctCal = [0 for x in range(args.num_classes)]

    if args.test:
        state = torch.load(os.path.join(args.test_model_path))
        print("test:")
        model.load_state_dict(state)
        model.eval()
        accAll = 0
        accNum = 0
        for image, label in tqdm.tqdm(test_loader):
            image = image.cuda()
            label = label.cuda()
            if not args.baseline:
                with torch.no_grad():
                    t1, t2, _, _, _, _, _, _, _ = model(image, image, args.baseline)
            else:
                with torch.no_grad():
                    t1, _, _, _, _, _, _, _ = model(image, None, args.baseline)
            _, prediction = torch.max(t1, 1)
            accAll += (prediction == label).sum().item()
            for i, j in enumerate(label):
                classesCal[j.item()] += 1
                if (prediction[i] == label[i]):
                    correctCal[j.item()] += 1
        for i in range(args.num_classes):
            if (classesCal[i]):
                accNum += correctCal[i] / classesCal[i]
                print("class {} \t acc: {}".format(i, correctCal[i] / classesCal[i]))
        accNum /= args.num_classes
        print("avgAcc: {} \t allAcc: {}".format(accNum, accAll / len(test_loader.dataset)))
    else:
        best_state = torch.load(args.save_dir + "/bestModel.pth")
        last_state = torch.load(args.save_dir + "/lastModel.pth")

        print("best test:")
        model.load_state_dict(best_state)
        model.eval()
        accAll = 0
        accNum = 0
        for image, label in tqdm.tqdm(test_loader):
            image = image.cuda()
            label = label.cuda()
            if not args.baseline:
                with torch.no_grad():
                    t1, t2, _, _, _, _, _, _, _ = model(image, image, args.baseline)
            else:
                with torch.no_grad():
                    t1, _, _, _, _, _, _, _ = model(image, None, args.baseline)
            _, prediction = torch.max(t1, 1)
            accAll += (prediction == label).sum().item()
            for i, j in enumerate(label):
                classesCal[j.item()] += 1
                if (prediction[i] == label[i]):
                    correctCal[j.item()] += 1
        for i in range(args.num_classes):
            if (classesCal[i]):
                accNum += correctCal[i] / classesCal[i]
                print("class {} \t acc: {}".format(i, correctCal[i] / classesCal[i]))
        accNum /= args.num_classes
        print("avgAcc: {} \t allAcc: {}".format(accNum, accAll / len(test_loader.dataset)))

        print("last test:")
        model.load_state_dict(last_state)
        model.eval()
        accNum = 0
        accAll = 0
        classesCal = [0 for x in range(args.num_classes)]
        correctCal = [0 for x in range(args.num_classes)]
        for image, label in tqdm.tqdm(test_loader):
            image = image.cuda()
            label = label.cuda()
            if not args.baseline:
                with torch.no_grad():
                    t1, t2, _, _, _, _, _, _, _ = model(image, image, args.baseline)
            else:
                with torch.no_grad():
                    t1, _, _, _, _, _, _, _ = model(image, None, args.baseline)
            _, prediction = torch.max(t1, 1)
            accAll += (prediction == label).sum().item()
            for i, j in enumerate(label):
                classesCal[j.item()] += 1
                if (prediction[i] == label[i]):
                    correctCal[j.item()] += 1
        for i in range(args.num_classes):
            if (classesCal[i]):
                accNum += correctCal[i] / classesCal[i]
                print("class {} \t acc: {}".format(i, correctCal[i] / classesCal[i]))
        accNum /= args.num_classes
        print("avgAcc: {} \t allAcc: {}".format(accNum, accAll / len(test_loader.dataset)))


def lossPull(prototype, feature, label, args):
    sim_matrix = torch.mm(F.normalize(feature, dim=-1), F.normalize(prototype, dim=-1).t().contiguous())
    mask = torch.zeros_like(sim_matrix)
    for i in range(args.batch_size):
        for j in range(args.proxy_per_cls):
            mask[i][args.proxy_per_cls * label[i] + j] = 1

    unsim = sim_matrix.masked_select((torch.ones_like(mask) - mask).bool()).mean()
    sim = - sim_matrix.masked_select(mask.bool()).mean()

    prototype_sim_matrix = torch.mm(F.normalize(prototype, dim=-1), F.normalize(prototype, dim=-1).t().contiguous())
    prototype_mask = (torch.ones_like(prototype_sim_matrix) - torch.eye(args.proxy_per_cls * args.num_classes,
                                                                        device=prototype_sim_matrix.device)).bool()
    prototypeLoss = prototype_sim_matrix.masked_select(prototype_mask).mean()

    return unsim + sim + prototypeLoss


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    totalWarm = 10
    if (epoch < totalWarm):
        cur_lr = init_lr * ((epoch + 1) / totalWarm)
    else:
        cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * (epoch + 1 - totalWarm) / (args.epochs - totalWarm)))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr
        if 'custom' in param_group and param_group['custom']:
            if 'big' in param_group:
                param_group['lr'] *= 10
            else:
                param_group['lr'] *= 5


def increment_path(path, exist_ok=True, sep=''):
    # Increment path, i.e. runs/exp --> runs/exp{sep}0, runs/exp{sep}1 etc.
    path = Path(path)  # os-agnostic
    if not path.exists():
        return str(path)
    else:
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        return f"{path}{sep}{n}"  # update path


if __name__ == '__main__':
    main()
