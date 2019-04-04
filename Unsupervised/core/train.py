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
from networks.PoseExpNet import *
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

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model2 = PoseExpNet(nb_ref_imgs=args.sequence_length - 1, output_exp=args.mask_loss_weight > 0).to(device)

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
        train_loss = train(train_loader, model, model2, [criterion1, criterion2, criterion3], optimizer, epoch, writer)
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



def train(train_loader, model, model2, criterions, optimizer, epoch, writer):
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
    model2.train()

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

        tgt_img = tgt_img.to(device)
        ref_imgs = [img.to(device) for img in ref_imgs]

        # compute output
        global_outputs, refine_output, depth_output = model(input_var)
        score_map = refine_output.data.cpu()

        explainability_mask, pose = model2(tgt_img, ref_imgs)

        loss = reconstructionLoss(tgt_img, ref_imgs, intrinsics, depth_output, explainability_mask, pose)
        
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

def reconstructionLoss(tgt_img, ref_imgs, intrinsics, epth, explainability_mask, pose,
                        rotation_mode='euler', padding_mode='zeros'):
    def one_scale(depth, explainability_mask):
        assert(explainability_mask is None or depth.size()[2:] == explainability_mask.size()[2:])
        assert(pose.size(1) == len(ref_imgs))

        reconstruction_loss = 0
        b, _, h, w = depth.size()
        downscale = tgt_img.size(2)/h

        tgt_img_scaled = F.interpolate(tgt_img, (h, w), mode='area')
        ref_imgs_scaled = [F.interpolate(ref_img, (h, w), mode='area') for ref_img in ref_imgs]
        intrinsics_scaled = torch.cat((intrinsics[:, 0:2]/downscale, intrinsics[:, 2:]), dim=1)

        warped_imgs = []
        diff_maps = []

        for i, ref_img in enumerate(ref_imgs_scaled):
            current_pose = pose[:, i]

            ref_img_warped, valid_points = inverse_warp(ref_img, depth[:,0], current_pose,
                                                        intrinsics_scaled,
                                                        rotation_mode, padding_mode)
            diff = (tgt_img_scaled - ref_img_warped) * valid_points.unsqueeze(1).float()

            if explainability_mask is not None:
                diff = diff * explainability_mask[:,i:i+1].expand_as(diff)

            reconstruction_loss += diff.abs().mean()
            assert((reconstruction_loss == reconstruction_loss).item() == 1)

            warped_imgs.append(ref_img_warped[0])
            diff_maps.append(diff[0])

        return reconstruction_loss, warped_imgs, diff_maps

    warped_results, diff_results = [], []
    if type(explainability_mask) not in [tuple, list]:
        explainability_mask = [explainability_mask]
    if type(depth) not in [list, tuple]:
        depth = [depth]

    total_loss = 0
    for d, mask in zip(depth, explainability_mask):
        loss, warped, diff = one_scale(d, mask)
        total_loss += loss
        warped_results.append(warped)
        diff_results.append(diff)
    return loss, warped_results, diff_results

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
