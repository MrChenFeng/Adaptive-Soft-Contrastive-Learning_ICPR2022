import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from tqdm import tqdm

from dataloader import *
from model import *
from utils import *

"""### Set arguments"""

parser = argparse.ArgumentParser(description='Train ASCL.')

# training config:
parser.add_argument('-a', '--arch', default='resnet18')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float, metavar='LR', help='initial learning rate',
                    dest='lr')
parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100', 'stl10', 'tinyimagenet'], type=str,
                    help='train dataset')
parser.add_argument('--data_path', default='cifar10', type=str, help='train dataset')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs')
parser.add_argument('--eval_epochs', default=100, type=int, metavar='N', help='number of total epochs')
parser.add_argument('--warm_up', default=5, type=int, metavar='N', help='number of lr warmup epochs')
parser.add_argument('--cos', action='store_false', help='use cosine lr schedule for self-supervised train')
parser.add_argument('--eval_cos', action='store_true', help='use cosine lr schedule for linear eval')
parser.add_argument('--batch-size', default=256, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--wd', default=5e-4, type=float, metavar='W', help='weight decay')
parser.add_argument('--m', default=0.99, type=float, help='moco momentum of updating key encoder')
parser.add_argument('--dim', default=128, type=int, help='feature dimension')
parser.add_argument('--mem_size', default=4096, type=int, help='memorybank size')
parser.add_argument('--resume', action='store_true', help='resume from previous run')
parser.add_argument('--gpuid', default='0', type=str, help='gpuid')
parser.add_argument('--run_id', default='0', type=str, help='run_id')

# model configs:
parser.add_argument('--K', default=1, type=int, help='number of neighbors')
parser.add_argument('--t1', default=0.1, type=float, help='softmax temperature for student')
parser.add_argument('--t2', default=0.05, type=float, help='softmax temperature for teacher')
parser.add_argument('--aug', default='weak_augment', choices=['weak_augment', 'strong_augment'], type=str,
                    help='augmentation for teacher')
parser.add_argument('--model', default='moco', choices=['byol', 'moco'], type=str, help='model type')
parser.add_argument('--type', default='ascl', choices=['ascl', 'ahcl', 'hard', 'normal'], type=str,
                    help='neighbor relation type')

def selfsup_train(args, model, train_loader, memory_loader, test_loader):
    """### Start training"""
    # define optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9)
    epoch_start = 0
    # load model if resume training
    if args.resume:
        checkpoint = torch.load(f'{args.run_id}/{args.results_dir}' + '/ckpts.pth')
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch_start = checkpoint['epoch']
        print('Loaded from: {}'.format(args.resume))

    # dump args
    with open(f'{args.run_id}/{args.results_dir}' + '/args.json', 'w') as fid:
        json.dump(args.__dict__, fid, indent=2)

    train_logs = open(f'{args.run_id}/{args.results_dir}/train_logs.txt', 'w')

    # train for one epoch
    def train(net, data_loader, train_optimizer, epoch, args):
        net.train()
        total_loss = AverageMeter('loss')
        train_bar = tqdm(data_loader)
        for i, [[im_1, im_2], targets] in enumerate(train_bar):
            adjust_learning_rate(train_optimizer, args.warm_up, epoch, args.epochs, args.lr, i, data_loader.__len__())
            im_1, im_2, targets = im_1.cuda(non_blocking=True), im_2.cuda(non_blocking=True), targets.cuda(
                non_blocking=True)
            loss = net(im_1, im_2, targets)

            train_optimizer.zero_grad()
            loss.backward()
            train_optimizer.step()

            total_loss.update(loss, im_1.size(0))
            train_bar.set_description(
                'Train Epoch: [{}/{}], lr: {:.6f}, Loss: {:.4f}'.format(
                    epoch, args.epochs,
                    train_optimizer.param_groups[0]['lr'],
                    total_loss.avg))

        return total_loss.avg

    # test using a knn monitor
    def online_test(net, memory_data_loader, test_data_loader, epoch, args):
        net.eval()
        classes = args.num_classes
        total_top1, total_top5, total_num, feature_bank, target_bank = 0.0, 0.0, 0, [], []
        with torch.no_grad():
            # generate feature bank
            for data, target in tqdm(memory_data_loader, desc='Feature extracting'):
                feature = net(data.cuda(non_blocking=True), feat=True)
                feature = F.normalize(feature, dim=1)
                feature_bank.append(feature)
                target_bank.append(target.cuda(non_blocking=True))
            # [D, N]
            feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
            feature_labels = torch.cat(target_bank, dim=0).t().contiguous()
            # [N]
            test_bar = tqdm(test_data_loader)
            for data, target in test_bar:
                data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
                feature = net(data, feat=True)
                feature = F.normalize(feature, dim=1)

                pred_labels = knn_predict(feature, feature_bank, feature_labels, classes, 200, 0.1)

                total_num += data.size(0)
                total_top1 += (pred_labels[:, 0] == target).float().sum().item()
                test_bar.set_description(
                    'Test Epoch: [{}/{}] Acc@1:{:.2f}%'.format(epoch, args.epochs, total_top1 / total_num * 100))

        return total_top1 / total_num * 100

    # training loop
    best_acc = 0
    for epoch in range(epoch_start, args.epochs):
        train_loss = train(model, train_loader, optimizer, epoch, args)

        cur_acc = online_test(model.encoder_q, memory_loader, test_loader, epoch, args)
        if cur_acc > best_acc:
            best_acc = cur_acc
        # save statistics
        train_logs.write(
            f'Epoch [{epoch}/{args.epochs}]: Best accuracy: {best_acc:.4f}! Current accuracy: {cur_acc:.4f} Current loss: {train_loss}\n')
        train_logs.flush()
        # save model
        if epoch % 50 == 0 and epoch != 0:
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), },
                       f'{args.run_id}/{args.results_dir}' + '/ckpts.pth')
    torch.save({'epoch': args.epochs, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), },
               f'{args.run_id}/{args.results_dir}' + '/model_last.pth')
    return model


def linear_eval(args, encoder, train_loader, test_loader):
    """Define train/test"""
    model = LinearHead(encoder, dim_in=encoder.feat_dim, num_class=args.num_classes).cuda()
    eval_logs = open(f'{args.run_id}/{args.results_dir}/eval_logs.txt', 'w')

    if args.eval_cos:
        lr = 1
    else:
        lr = 10
    optimizer = torch.optim.SGD(model.fc.parameters(), lr=lr, momentum=0.9, weight_decay=0, nesterov=True)
    if args.eval_cos:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=args.eval_epochs * train_loader.__len__())
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 80], gamma=0.1)

    best_acc = 0
    best_acc5 = 0

    for epoch in range(args.eval_epochs):
        losses = AverageMeter('Loss', ':.4e')

        model.eval()
        for i, (image, label) in enumerate(train_loader):
            image = image.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)

            out = model(image)
            loss = F.cross_entropy(out, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())

            if args.eval_cos:
                scheduler.step()

        if not args.eval_cos:
            scheduler.step()

        model.eval()
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        with torch.no_grad():
            # end = time.time()
            for i, (image, label) in enumerate(test_loader):
                image = image.cuda(non_blocking=True)
                label = label.cuda(non_blocking=True)

                # compute output
                output = model(image)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, label, topk=(1, 5))
                top1.update(acc1[0], image.size(0))
                top5.update(acc5[0], image.size(0))

        sum1, cnt1, sum5, cnt5 = top1.sum, top1.count, top5.sum, top5.count

        top1_acc = sum1.float() / float(cnt1)
        top5_acc = sum5.float() / float(cnt5)

        best_acc = max(top1_acc, best_acc)
        best_acc5 = max(top5_acc, best_acc5)
        print(
            'Epoch:{} * Acc@1 {top1_acc:.3f} Acc@5 {top5_acc:.3f} Best_Acc@1 {best_acc:.3f} Best_Acc@5 {best_acc5:.3f}'.format(
                epoch, top1_acc=top1_acc,
                top5_acc=top5_acc,
                best_acc=best_acc,
                best_acc5=best_acc5))
        # save statistics
        eval_logs.write(
            'Epoch:{} * Acc@1 {top1_acc:.3f} Acc@5 {top5_acc:.3f} Best_Acc@1 {best_acc:.3f} Best_Acc@5 {best_acc5:.3f}'.format(
                epoch, top1_acc=top1_acc,
                top5_acc=top5_acc,
                best_acc=best_acc,
                best_acc5=best_acc5))
        eval_logs.flush()


def main():
    args = parser.parse_args()
    print(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid
    random.seed(1228)
    torch.manual_seed(1228)
    torch.cuda.manual_seed_all(1228)
    np.random.seed(1228)
    torch.backends.cudnn.benchmark = True

    ############################################### Init ####################################################
    # create model
    model = ASCL(dataset=args.dataset, model = args.model, dim=args.dim, mem_size=args.mem_size, m=args.m, T1=args.t1, T2=args.t2, arch=args.arch,  type=args.type,
                 nn_num=args.K).cuda()

    # logs
    args.results_dir = f'{args.dataset}_{args.model}_{args.type}_{args.epochs}_{args.aug}_{args.nn_num}_{args.t1}_{args.t2}'
    if not os.path.exists(args.run_id):
        os.mkdir(args.run_id)
    if not os.path.exists(f'{args.run_id}/{args.results_dir}'):
        os.mkdir(f'{args.run_id}/{args.results_dir}')

    # prepare dataset
    train_loader, memory_loader, linear_loader, test_loader = prepare_dataset(args)

    ############################################### Main ####################################################
    # self-supervised training
    model = selfsup_train(args, model, train_loader, memory_loader, test_loader)

    # linear eval loop
    linear_eval(args, model.encoder_q, linear_loader, test_loader)


if __name__ == '__main__':
    main()
