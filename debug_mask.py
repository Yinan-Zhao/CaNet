from torch.utils import data
import torch.optim as optim
import torch.backends.cudnn as cudnn
import os.path as osp
from utils import *
import time
import torch.nn.functional as F
import tqdm
import random
import argparse
from dataset_mask_memory_train import Dataset as Dataset_train
from dataset_mask_memory_val import Dataset as Dataset_val
import os
import torch
from one_shot_network_mask import Res_Deeplab
import torch.nn as nn
import numpy as np



parser = argparse.ArgumentParser()


parser.add_argument('-lr',
                    type=float,
                    help='learning rate',
                    default=0.00025)

parser.add_argument('-prob',
                    type=float,
                    help='dropout rate of history mask',
                    default=0.7)


parser.add_argument('-bs',
                    type=int,
                    help='batchsize',
                    default=4)

parser.add_argument('-bs_val',
                    type=int,
                    help='batchsize for val',
                    default=1)



parser.add_argument('-fold',
                    type=int,
                    help='fold',
                    default=1)



parser.add_argument('-gpu',
                    type=str,
                    help='gpu id to use',
                    default='0')

parser.add_argument('-checkpoint_dir',
                    type=str,
                    help='checkpoint directory',
                    default='checkpoint')

parser.add_argument('-p_scalar',
                    type=float,
                    default=100.)

parser.add_argument('-iter_time',
                    type=int,
                    default=1)

parser.add_argument('--aspp',
                    action='store_true',
                    help="use aspp module")

parser.add_argument('--is_debug',
                    action='store_true',
                    help="use aspp module")

parser.add_argument('--normalize_key',
                    action='store_true',
                    help="use aspp module")

options = parser.parse_args()


data_dir = '/home/yz9244/CaNet/data/pascal/VOCdevkit/VOC2012/'


#set gpus
gpu_list = [int(x) for x in options.gpu.split(',')]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = options.gpu

torch.backends.cudnn.benchmark = True


IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]
num_class = 2
num_epoch = 200
learning_rate = options.lr  # 0.000025#0.00025
input_size = (321, 321)
batch_size = options.bs
weight_decay = 0.0005
momentum = 0.9
power = 0.9

cudnn.enabled = True

# Create network.
model = Res_Deeplab(num_classes=num_class, aspp=options.aspp, p_scalar=options.p_scalar, normalize_key=options.normalize_key)
#load resnet-50 preatrained parameter
model = load_resnet50_param(model, stop_layer='layer4')
model.cuda()
#model=nn.DataParallel(model,[2,3])

# disable the  gradients of not optomized layers
turn_off(model)



checkpoint = 'ckpt/%s/fo=%d/model/best.pth' % (options.checkpoint_dir, 1)
model.load_state_dict(torch.load(checkpoint, map_location=lambda storage, loc: storage), strict=False)


# loading data
# valset
# this only a quick val dataset where all images are 321*321.
valset = Dataset_val(data_dir=data_dir, fold=options.fold, input_size=input_size, normalize_mean=IMG_MEAN,
                 normalize_std=IMG_STD)
valloader = data.DataLoader(valset, batch_size=options.bs_val, shuffle=False, num_workers=4,
                            drop_last=False)


loss_list = []#track training loss
iou_list = []#track validaiton iou

model.cuda()

# ======================evaluate now==================
with torch.no_grad():
    print ('----Evaluation----')
    model = model.eval()

    valset.history_mask_list=[None] * 1000
    best_iou = 0
    count = 0
    for eva_iter in range(options.iter_time):
        count = 0
        all_inter, all_union, all_predict = [0] * 5, [0] * 5, [0] * 5
        for i_iter, batch in enumerate(valloader):

            query_rgb, query_mask, support_rgb, support_mask, history_mask, sample_class, index, query_name, support_name = batch

            query_rgb = (query_rgb).cuda(0)
            support_rgb = (support_rgb).cuda(0)
            support_mask = (support_mask).cuda(0)
            query_mask = (query_mask).cuda(0).long()  # change formation for crossentropy use

            query_mask = query_mask[:, 0, :, :]  # remove the second dim,change formation for crossentropy use
            history_mask = (history_mask).cuda(0)

            if options.is_debug:
                pred, qread, qk_b, mk_b, mv_b, p = model(query_rgb, support_rgb, support_mask, is_debug=True)

                np.save('debug/qread-%04d-%s-%s.npy'%(count, query_name[0], support_name[0]), qread.detach().cpu().float().numpy())
                np.save('debug/p-%04d-%s-%s.npy'%(count, query_name[0], support_name[0]), p.detach().cpu().float().numpy())
            else:
                pred = model(query_rgb, support_rgb, support_mask)

            pred_softmax = F.softmax(pred, dim=1).data.cpu()

            # update history mask
            for j in range(support_mask.shape[0]):
                sub_index = index[j]
                valset.history_mask_list[sub_index] = pred_softmax[j]

                pred = nn.functional.interpolate(pred, size=input_size, mode='bilinear',
                                                 align_corners=True)  #upsample  # upsample
            _, pred_label = torch.max(pred, 1)
            inter_list, union_list, _, num_predict_list = get_iou_v1(query_mask, pred_label)
            for j in range(query_mask.shape[0]):#batch size
                all_inter[sample_class[j] - (options.fold * 5 + 1)] += inter_list[j]
                all_union[sample_class[j] - (options.fold * 5 + 1)] += union_list[j]

            count += 1

        IOU = [0] * 5

        for j in range(5):
            IOU[j] = all_inter[j] / all_union[j]

        mean_iou = np.mean(IOU)
        print('IOU:%.4f' % (mean_iou))











