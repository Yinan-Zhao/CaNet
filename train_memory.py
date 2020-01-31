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
from one_shot_network import Res_Deeplab
import torch.nn as nn
import numpy as np
from config import cfg
import random
from models import ModelBuilder, SegmentationAttentionSeparateModule


data_dir = '/home/yz9244/CaNet/data/pascal/VOCdevkit/VOC2012/'
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]
num_class = 2
num_epoch = 200
input_size = (328, 328)
weight_decay = 0.0005
momentum = 0.9
power = 0.9

cudnn.enabled = True

def data_preprocess(sample_batched, cfg, val=False):
    query_rgb, query_mask, support_rgb, support_mask, history_mask, sample_class, index= batch
    feed_dict = {}
    feed_dict['img_data'] = query_rgb.cuda()
    if not val:
        query_mask = nn.functional.interpolate(query_mask, size=(input_size[0]//cfg.DATASET.segm_downsampling_rate,
            input_size[1]//cfg.DATASET.segm_downsampling_rate), mode='nearest')
    feed_dict['seg_label'] = query_mask[:,0,:,:].long().cuda() 


    feed_dict['img_refs_rgb'] = torch.unsqueeze(support_rgb, 2)
    feed_dict['img_refs_rgb'] = feed_dict['img_refs_rgb'].cuda()

    feed_dict['img_refs_mask'] = torch.unsqueeze(support_mask, 2)
    feed_dict['img_refs_mask'] = feed_dict['img_refs_mask'].cuda()

    return feed_dict

def checkpoint(nets, cfg, iter_idx):
    print('Saving checkpoints...')
    (net_enc_query, net_enc_memory, net_att_query, net_att_memory, net_decoder, crit) = nets

    dict_enc_query = net_enc_query.state_dict()
    dict_enc_memory = net_enc_memory.state_dict()
    dict_att_query = net_att_query.state_dict()
    dict_att_memory = net_att_memory.state_dict()
    dict_decoder = net_decoder.state_dict()

    torch.save(
        dict_enc_query,
        '{}/enc_query_iter_{}.pth'.format(cfg.DIR, iter_idx))
    torch.save(
        dict_enc_memory,
        '{}/enc_memory_iter_{}.pth'.format(cfg.DIR, iter_idx))
    torch.save(
        dict_att_query,
        '{}/att_query_iter_{}.pth'.format(cfg.DIR, iter_idx))
    torch.save(
        dict_att_memory,
        '{}/att_memory_iter_{}.pth'.format(cfg.DIR, iter_idx))
    torch.save(
        dict_decoder,
        '{}/decoder_iter_{}.pth'.format(cfg.DIR, iter_idx))

def group_weight(module):
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.conv._ConvNd):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.batchnorm._BatchNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)

    assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
    groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
    return groups

def create_optimizers(nets, cfg):
    (net_enc_query, net_enc_memory, net_att_query, net_att_memory, net_decoder, crit) = nets
    optimizer_enc_query = torch.optim.SGD(
        group_weight(net_enc_query),
        lr=cfg.TRAIN.lr_encoder,
        momentum=cfg.TRAIN.beta1,
        weight_decay=cfg.TRAIN.weight_decay)
    optimizer_enc_memory = torch.optim.SGD(
        group_weight(net_enc_memory),
        lr=cfg.TRAIN.lr_encoder,
        momentum=cfg.TRAIN.beta1,
        weight_decay=cfg.TRAIN.weight_decay)
    optimizer_att_query = torch.optim.SGD(
        group_weight(net_att_query),
        lr=cfg.TRAIN.lr_encoder,
        momentum=cfg.TRAIN.beta1,
        weight_decay=cfg.TRAIN.weight_decay)
    optimizer_att_memory = torch.optim.SGD(
        group_weight(net_att_memory),
        lr=cfg.TRAIN.lr_encoder,
        momentum=cfg.TRAIN.beta1,
        weight_decay=cfg.TRAIN.weight_decay)
    optimizer_decoder = torch.optim.SGD(
        group_weight(net_decoder),
        lr=cfg.TRAIN.lr_decoder,
        momentum=cfg.TRAIN.beta1,
        weight_decay=cfg.TRAIN.weight_decay)
    return (optimizer_enc_query, optimizer_enc_memory, optimizer_att_query, optimizer_att_memory, optimizer_decoder)

def adjust_learning_rate(optimizers, cur_iter, total_iter, cfg):
    scale_running_lr = ((1. - float(cur_iter) / total_iter) ** cfg.TRAIN.lr_pow)
    cfg.TRAIN.running_lr_encoder = cfg.TRAIN.lr_encoder * scale_running_lr
    cfg.TRAIN.running_lr_decoder = cfg.TRAIN.lr_decoder * scale_running_lr

    (optimizer_enc_query, optimizer_enc_memory, optimizer_att_query, optimizer_att_memory, optimizer_decoder) = optimizers
    for param_group in optimizer_enc_query.param_groups:
        param_group['lr'] = cfg.TRAIN.running_lr_encoder
    for param_group in optimizer_enc_memory.param_groups:
        param_group['lr'] = cfg.TRAIN.running_lr_encoder
    for param_group in optimizer_att_query.param_groups:
        param_group['lr'] = cfg.TRAIN.running_lr_encoder
    for param_group in optimizer_att_memory.param_groups:
        param_group['lr'] = cfg.TRAIN.running_lr_encoder
    for param_group in optimizer_decoder.param_groups:
        param_group['lr'] = cfg.TRAIN.running_lr_decoder


parser = argparse.ArgumentParser()
parser.add_argument('-bs',
                    type=int,
                    help='batchsize',
                    default=4)
parser.add_argument('-bs_val',
                    type=int,
                    help='batchsize for val',
                    default=64)
parser.add_argument('-fold',
                    type=int,
                    help='fold',
                    default=1)
parser.add_argument('-gpu',
                    type=str,
                    help='gpu id to use',
                    default='0')
parser.add_argument("--cfg",
                    default="config/ade20k-resnet50dilated-ppm_deepsup.yaml",
                    metavar="FILE",
                    help="path to config file",
                    type=str)
parser.add_argument("--memory_enc_pretrained",
                    action='store_true',
                    help="use a pretrained memory encoder")


options = parser.parse_args()
args = options

cfg.merge_from_file(args.cfg)
cfg.memory_enc_pretrained = args.memory_enc_pretrained

checkpoint_dir = cfg.DIR
if not os.path.isdir(cfg.DIR):
    os.makedirs(cfg.DIR)

with open(os.path.join(cfg.DIR, 'config.yaml'), 'w') as f:
    f.write("{}".format(cfg))

# Start from checkpoint
'''if cfg.TRAIN.start_iter > 0:
    cfg.MODEL.weights_enc_query = os.path.join(
        cfg.DIR, 'enc_query_iter_{}.pth'.format(cfg.TRAIN.start_iter))
    cfg.MODEL.weights_enc_memory = os.path.join(
        cfg.DIR, 'enc_memory_iter_{}.pth'.format(cfg.TRAIN.start_iter))
    cfg.MODEL.weights_att_query = os.path.join(
        cfg.DIR, 'att_query_iter_{}.pth'.format(cfg.TRAIN.start_iter))
    cfg.MODEL.weights_att_memory = os.path.join(
        cfg.DIR, 'att_memory_iter_{}.pth'.format(cfg.TRAIN.start_iter))
    cfg.MODEL.weights_decoder = os.path.join(
        cfg.DIR, 'decoder_iter_{}.pth'.format(cfg.TRAIN.start_iter))
    assert os.path.exists(cfg.MODEL.weights_enc_query) and os.path.exists(cfg.MODEL.weights_enc_memory) and \
        os.path.exists(cfg.MODEL.weights_att_query) and os.path.exists(cfg.MODEL.weights_att_memory) and \
        os.path.exists(cfg.MODEL.weights_decoder), "checkpoint does not exitst!"'''


cfg.TRAIN.running_lr_encoder = cfg.TRAIN.lr_encoder
cfg.TRAIN.running_lr_decoder = cfg.TRAIN.lr_decoder

random.seed(cfg.TRAIN.seed)
torch.manual_seed(cfg.TRAIN.seed)


#set gpus
gpu_list = [int(x) for x in options.gpu.split(',')]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = options.gpu
torch.backends.cudnn.benchmark = True


net_enc_query = ModelBuilder.build_encoder(
    arch=cfg.MODEL.arch_encoder.lower(),
    fc_dim=cfg.MODEL.fc_dim,
    weights=cfg.MODEL.weights_enc_query)
if cfg.MODEL.memory_encoder_arch:
    net_enc_memory = ModelBuilder.build_encoder_memory_separate(
        arch=cfg.MODEL.memory_encoder_arch.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        weights=cfg.MODEL.weights_enc_memory,
        num_class=cfg.TASK.n_ways+1,
        RGB_mask_combine_val=cfg.DATASET.RGB_mask_combine_val,
        segm_downsampling_rate=cfg.DATASET.segm_downsampling_rate)
else:
    if cfg.MODEL.memory_encoder_noBN:
        net_enc_memory = ModelBuilder.build_encoder_memory_separate(
            arch=cfg.MODEL.arch_encoder.lower()+'_nobn',
            fc_dim=cfg.MODEL.fc_dim,
            weights=cfg.MODEL.weights_enc_memory,
            num_class=cfg.TASK.n_ways+1,
            RGB_mask_combine_val=cfg.DATASET.RGB_mask_combine_val,
            segm_downsampling_rate=cfg.DATASET.segm_downsampling_rate)
    else:
        net_enc_memory = ModelBuilder.build_encoder_memory_separate(
            arch=cfg.MODEL.arch_encoder.lower(),
            fc_dim=cfg.MODEL.fc_dim,
            weights=cfg.MODEL.weights_enc_memory,
            num_class=cfg.TASK.n_ways+1,
            RGB_mask_combine_val=cfg.DATASET.RGB_mask_combine_val,
            segm_downsampling_rate=cfg.DATASET.segm_downsampling_rate,
            pretrained=cfg.memory_enc_pretrained)
net_att_query = ModelBuilder.build_att_query(
    arch=cfg.MODEL.arch_attention,
    fc_dim=cfg.MODEL.fc_dim,
    weights=cfg.MODEL.weights_att_query)
net_att_memory = ModelBuilder.build_att_memory(
    arch=cfg.MODEL.arch_attention,
    fc_dim=cfg.MODEL.fc_dim,
    att_fc_dim=cfg.MODEL.att_fc_dim,
    weights=cfg.MODEL.weights_att_memory)
net_decoder = ModelBuilder.build_decoder(
    arch=cfg.MODEL.arch_decoder.lower(),
    fc_dim=cfg.MODEL.fc_dim,
    num_class=cfg.TASK.n_ways+1,
    weights=cfg.MODEL.weights_decoder)

crit = nn.NLLLoss(ignore_index=255)

if cfg.MODEL.arch_decoder.endswith('deepsup'):
    segmentation_module = SegmentationAttentionSeparateModule(
        net_enc_query, net_enc_memory, net_att_query, net_att_memory, net_decoder, crit, cfg.TRAIN.deep_sup_scale, zero_memory=cfg.MODEL.zero_memory, random_memory_bias=cfg.MODEL.random_memory_bias, random_memory_nobias=cfg.MODEL.random_memory_nobias, random_scale=cfg.MODEL.random_scale, zero_qval=cfg.MODEL.zero_qval, qval_qread_BN=cfg.MODEL.qval_qread_BN, normalize_key=cfg.MODEL.normalize_key, p_scalar=cfg.MODEL.p_scalar, memory_feature_aggregation=cfg.MODEL.memory_feature_aggregation, memory_noLabel=cfg.MODEL.memory_noLabel, mask_feat_downsample_rate=cfg.MODEL.mask_feat_downsample_rate, att_mat_downsample_rate=cfg.MODEL.att_mat_downsample_rate, att_voting=cfg.MODEL.att_voting, mask_foreground=cfg.MODEL.mask_foreground)
else:
    segmentation_module = SegmentationAttentionSeparateModule(
        net_enc_query, net_enc_memory, net_att_query, net_att_memory, net_decoder, crit, zero_memory=cfg.MODEL.zero_memory, random_memory_bias=cfg.MODEL.random_memory_bias, random_memory_nobias=cfg.MODEL.random_memory_nobias, random_scale=cfg.MODEL.random_scale, zero_qval=cfg.MODEL.zero_qval, qval_qread_BN=cfg.MODEL.qval_qread_BN, normalize_key=cfg.MODEL.normalize_key, p_scalar=cfg.MODEL.p_scalar, memory_feature_aggregation=cfg.MODEL.memory_feature_aggregation, memory_noLabel=cfg.MODEL.memory_noLabel, mask_feat_downsample_rate=cfg.MODEL.mask_feat_downsample_rate, att_mat_downsample_rate=cfg.MODEL.att_mat_downsample_rate, att_voting=cfg.MODEL.att_voting, mask_foreground=cfg.MODEL.mask_foreground)


segmentation_module.cuda()

# Set up optimizers
nets = (net_enc_query, net_enc_memory, net_att_query, net_att_memory, net_decoder, crit)
optimizers = create_optimizers(nets, cfg)


# trainset
dataset = Dataset_train(data_dir=data_dir, fold=options.fold, input_size=input_size, normalize_mean=IMG_MEAN,
                  normalize_std=IMG_STD)
trainloader = data.DataLoader(dataset, batch_size=options.bs, shuffle=True, num_workers=4)

# valset
# this only a quick val dataset where all images are 321*321.
valset = Dataset_val(data_dir=data_dir, fold=options.fold, input_size=input_size, normalize_mean=IMG_MEAN,
                 normalize_std=IMG_STD)
valloader = data.DataLoader(valset, batch_size=options.bs_val, shuffle=False, num_workers=4,
                            drop_last=False)

save_pred_every = len(trainloader)


loss_list = []#track training loss
iou_list = []#track validaiton iou
highest_iou = 0

tempory_loss = 0  # accumulated loss
best_epoch=0
for epoch in range(0,num_epoch):
    segmentation_module.train(not cfg.TRAIN.fix_bn)
    begin_time = time.time()
    tqdm_gen = tqdm.tqdm(trainloader)

    for i_iter, batch in enumerate(tqdm_gen):

        feed_dict = data_preprocess(batch, cfg, False)
        _, _, _, _, _,sample_class,index= batch
        segmentation_module.zero_grad()

        # adjust learning rate
        cur_iter = i_iter + (epoch - 1) * len(trainloader)
        total_iter = num_epoch*len(trainloader)
        adjust_learning_rate(optimizers, cur_iter, total_iter, cfg)
        # forward pass
        loss, acc = segmentation_module(feed_dict)

        loss.backward()
        for optimizer in optimizers:
            optimizer.step()

        tqdm_gen.set_description('e:%d loss = %.4f-:%.4f' % (
        epoch, loss.item(),highest_iou))


        #save training loss
        tempory_loss += loss.item()
        if i_iter % save_pred_every == 0 and i_iter != 0:

            loss_list.append(tempory_loss / save_pred_every)
            #plot_loss(checkpoint_dir, loss_list, save_pred_every)
            np.savetxt(os.path.join(checkpoint_dir, 'loss_history.txt'), np.array(loss_list))
            tempory_loss = 0

    # ======================evaluate now==================
    with torch.no_grad():
        print ('----Evaluation----')
        segmentation_module.eval()

        valset.history_mask_list=[None] * 1000
        best_iou = 0
        for eva_iter in range(options.iter_time):
            all_inter, all_union, all_predict = [0] * 5, [0] * 5, [0] * 5
            for i_iter, batch in enumerate(valloader):
                feed_dict = data_preprocess(batch, cfg, True)
                _, query_mask, _, _, _, sample_class, index = batch
                pred_softmax = segmentation_module(feed_dict, segSize=input_size)
                pred = pred_softmax.cpu()
                

                _, pred_label = torch.max(pred, 1)
                inter_list, union_list, _, num_predict_list = get_iou_v1(query_mask, pred_label)
                for j in range(query_mask.shape[0]):#batch size
                    all_inter[sample_class[j] - (options.fold * 5 + 1)] += inter_list[j]
                    all_union[sample_class[j] - (options.fold * 5 + 1)] += union_list[j]


            IOU = [0] * 5

            for j in range(5):
                IOU[j] = all_inter[j] / all_union[j]

            mean_iou = np.mean(IOU)
            print('IOU:%.4f' % (mean_iou))
            if mean_iou > best_iou:
                best_iou = mean_iou
            else:
                break

        iou_list.append(best_iou)
        #plot_iou(checkpoint_dir, iou_list)
        np.savetxt(os.path.join(checkpoint_dir, 'iou_history.txt'), np.array(iou_list))
        if best_iou>highest_iou:
            highest_iou = best_iou
            segmentation_module.eval()
            checkpoint(nets, cfg, 'best')
            segmentation_module.train(not cfg.TRAIN.fix_bn)
            best_epoch = epoch
            print('A better model is saved')

        print('IOU for this epoch: %.4f' % (best_iou))


        segmentation_module.train(not cfg.TRAIN.fix_bn)
        segmentation_module.cuda()



    epoch_time = time.time() - begin_time
    print('best epoch:%d ,iout:%.4f' % (best_epoch, highest_iou))
    print('This epoch taks:', epoch_time, 'second')
    print('still need hour:%.4f' % ((num_epoch - epoch) * epoch_time / 3600))



