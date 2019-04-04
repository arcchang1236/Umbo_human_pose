import os
import argparse
import time
import matplotlib.pyplot as plt

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.datasets as datasets
import numpy as np

from config import cfg
from utils.logger import Logger
from utils.evaluation import accuracy, AverageMeter, final_preds
from utils.misc import save_model, adjust_learning_rate
from utils.osutils import mkdir_p, isfile, isdir, join
from utils.transforms import fliplr, flip_back
from networks import network 
from dataloader.mscocoMulti import MscocoMulti
from dataloader.panoptic import PanopticDataset
from tensorboardX import SummaryWriter 

def main(args):
    # create checkpoint dir
    if not isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # TensorBoardX
    writer = SummaryWriter('runs/exp-1')
    writer = SummaryWriter()
    
    # create model
    model = network.__dict__[cfg.model](cfg.output_shape, cfg.num_of_joints, pretrained=True)
    model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion1 = torch.nn.MSELoss().cuda() # for Global loss
    criterion2 = torch.nn.MSELoss(reduce=False).cuda() # for Refine loss
    criterion3 = torch.nn.MSELoss().cuda() # for Depth loss

    optimizer = torch.optim.Adam(model.parameters(),
                                lr = cfg.lr,
                                weight_decay=cfg.weight_decay)
    
    if args.resume:
        if isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            pretrained_dict = checkpoint['state_dict']
            model.load_state_dict(pretrained_dict)
            args.start_epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            logger = Logger(join(args.checkpoint, 'log.txt'), resume=True)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:        
        logger = Logger(join(args.checkpoint, 'log.txt'))
        logger.set_names(['Epoch', 'LR', 'Train Loss'])

    cudnn.benchmark = True
    print('    Total params: %.2fMB' % (sum(p.numel() for p in model.parameters())/(1024*1024)*4))

    train_loader = torch.utils.data.DataLoader(
        PanopticDataset(cfg),
        batch_size=cfg.batch_size*args.num_gpus, shuffle=True,
        num_workers=args.workers, pin_memory=True) 
    print("Iteration for each epoch: {}".format(len(train_loader)))

    for epoch in range(args.start_epoch, args.epochs):
        lr = adjust_learning_rate(optimizer, epoch, cfg.lr_dec_epoch, cfg.lr_gamma)
        print('\nEpoch: %d | LR: %.8f' % (epoch + 1, lr)) 

        # train for one epoch
        train_loss = train(train_loader, model, [criterion1, criterion2, criterion3], optimizer, epoch, writer)
        print('train_loss: ',train_loss)
        writer.add_scalar('train_loss_epoch', train_loss, epoch)

        # append logger file
        logger.append([epoch + 1, lr, train_loss])

        save_model({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }, checkpoint=args.checkpoint)

    logger.close()



def train(train_loader, model, criterions, optimizer, epoch, writer):
    # prepare for refine loss
    def ohkm(loss, top_k):
        ohkm_loss = 0.
        for i in range(loss.size()[0]):
            sub_loss = loss[i]
            topk_val, topk_idx = torch.topk(sub_loss, k=top_k, dim=0, sorted=False)
            tmp_loss = torch.gather(sub_loss, 0, topk_idx)
            ohkm_loss += torch.sum(tmp_loss) / top_k
        ohkm_loss /= loss.size()[0]
        return ohkm_loss
    criterion1, criterion2, criterion3 = criterions

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    #for i, (inputs, targets_2d, valid, meta) in enumerate(train_loader): 
    for i, (inputs, joints_plane, joints_world, targets_2d, targets_3d) in enumerate(train_loader): 
        # print(inputs.shape, targets_2d[0].shape, valid[0].shape, meta.shape)
        # print(len(joints_plane), len(targets_2d))
        # print(inputs.shape, targets_2d[0].shape, targets_3d.shape)

        input_var = torch.autograd.Variable(inputs.cuda())
        target15, target11, target9, target7 = targets_2d
        refine_target_var = torch.autograd.Variable(target7.cuda(async=True))
        depth_target_var = torch.autograd.Variable(targets_3d.cuda(async=True))
        #valid_var = torch.autograd.Variable(valid.cuda(async=True))

        #print(targets[0].shape, valid[0].shape, len(valid))

        # compute output
        global_outputs, refine_output, depth_output = model(input_var)
        score_map = refine_output.data.cpu()

        loss = 0.
        global_loss_record = 0.
        refine_loss_record = 0.
        depth_loss_record = 0.
        # compute global loss and refine loss
        for global_output, label in zip(global_outputs, targets_2d):
            num_points = global_output.size()[1]
            #global_label = label * (valid > 1.1).type(torch.FloatTensor).view(-1, num_points, 1, 1)
            global_label = label
            #print(global_output.size(), num_points, np.shape(global_label), np.shape(label))
            global_loss = criterion1(global_output, torch.autograd.Variable(global_label.cuda(async=True))) / 2.0
            loss += global_loss
            global_loss_record += global_loss.data.item()
        #print(refine_output.shape, refine_target_var.shape)
        refine_loss = criterion2(refine_output, refine_target_var)
        refine_loss = refine_loss.mean(dim=3).mean(dim=2)
        #refine_loss *= (valid_var > 0.1).type(torch.cuda.FloatTensor)
        refine_loss = ohkm(refine_loss, 8)
        loss += refine_loss
        refine_loss_record = refine_loss.data.item()

        # compute depth loss
        depth_loss = criterion3(depth_output, depth_target_var)
        loss += depth_loss
        depth_loss_record = depth_loss.data.item()
        
        # record loss
        losses.update(loss.data.item(), inputs.size(0))

        # compute gradient and do Optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if(i%500==0 and i!=0):
            print('Epoch: {} [{}] | Ltotal: {:.8f}, Lg: {:.8f}, Lr: {:.8f}, Ld: {:.8f}, Lavg: {:.8f}'
                .format(epoch, i, loss.data.item(), global_loss_record, 
                    refine_loss_record, depth_loss_record, losses.avg)) 
            writer.add_scalar('train_loss', losses.avg, i*epoch)

            # Visualize
            TA = targets_3d[0][0].data.squeeze().cpu().numpy().astype(np.float32)
            TB = target7[0][0].data.squeeze().cpu().numpy().astype(np.float32)
            TA = np.transpose(TA)
            TB = np.transpose(TB)
            plt.imsave('heatmap/gt/3D_{}_{}.png'.format(epoch, i), TA, cmap="viridis")
            plt.imsave('heatmap/gt/2D_{}_{}.png'.format(epoch, i), TB, cmap="viridis")

            TC = refine_output[0][0].data.squeeze().cpu().numpy().astype(np.float32)
            TC = np.transpose(TC)
            plt.imsave('heatmap/2d/{}_{}.png'.format(epoch, i), TC, cmap="viridis")
            TD = depth_output[0][0].data.squeeze().cpu().numpy().astype(np.float32)
            TD = np.transpose(TD)
            plt.imsave('heatmap/3d/{}_{}.png'.format(epoch, i), TD, cmap="viridis")

    return losses.avg



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CPN Training')
    parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                        help='number of data loading workers (default: 12)')
    parser.add_argument('-g', '--num_gpus', default=1, type=int, metavar='N',
                        help='number of GPU to use (default: 1)')    
    parser.add_argument('--epochs', default=32, type=int, metavar='N',
                        help='number of total epochs to run (default: 32)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint')


    main(parser.parse_args())
