from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()
_C.DIR = "ckpt/ade20k-resnet50dilated-ppm_deepsup"

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.name = "VOC"
_C.DATASET.data_dir = "./data/pascal/VOCdevkit/VOC2012/"
_C.DATASET.data_split = "trainaug"
_C.DATASET.input_size = (256, 256)


_C.DATASET.root_dataset = "./data/"
_C.DATASET.list_train = "./data/training.odgt"
_C.DATASET.list_val = "./data/validation.odgt"
# multiscale train/test, size of short edge (int or tuple)
_C.DATASET.imgSizes = (300, 375, 450, 525, 600)
# maximum input image size of long edge
_C.DATASET.imgMaxSize = 1000
# maxmimum downsampling rate of the network
_C.DATASET.padding_constant = 8
# downsampling rate of the segmentation label
_C.DATASET.segm_downsampling_rate = 8
# randomly horizontally flip images when train/test
_C.DATASET.random_flip = True
_C.DATASET.ref_path = "./data/ref_training.txt"
_C.DATASET.ref_val_path = "./data/ref_real_val.txt"
_C.DATASET.ref_start = 0
_C.DATASET.ref_end = 1
_C.DATASET.ref_val_start = 0
_C.DATASET.ref_val_end = 1
_C.DATASET.random_pick = 1
_C.DATASET.no_align = False
_C.DATASET.zero_input_rgb = False
_C.DATASET.zero_input_seg = False
_C.DATASET.random_input_seg = False
_C.DATASET.RGB_mask_combine_val = False

# -----------------------------------------------------------------------------
# Task
# -----------------------------------------------------------------------------
_C.TASK = CN()
_C.TASK.n_ways = 1
_C.TASK.n_shots = 1
_C.TASK.n_queries = 1
_C.TASK.fold_idx = 0

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# architecture of net_encoder
_C.MODEL.arch_encoder = "resnet50dilated"
# architecture of net_attention
_C.MODEL.arch_attention = "attention"
# architecture of net_decoder
_C.MODEL.arch_decoder = "ppm_deepsup"
# weights to finetune net_encoder
_C.MODEL.weights_encoder = ""
# weights to finetune net_decoder
_C.MODEL.weights_decoder = ""
_C.MODEL.weights_enc_query = ""
_C.MODEL.weights_enc_memory = ""
_C.MODEL.weights_att_query = ""
_C.MODEL.weights_att_memory = ""
# number of feature channels between encoder and decoder
_C.MODEL.fc_dim = 2048
_C.MODEL.att_fc_dim = 512
_C.MODEL.mask_feat_downsample_rate = 1
_C.MODEL.att_mat_downsample_rate = 1
_C.MODEL.zero_memory = False
_C.MODEL.zero_qval = False
_C.MODEL.random_memory_bias = False
_C.MODEL.random_memory_nobias = False
_C.MODEL.random_scale = 1.0
_C.MODEL.memory_encoder_noBN = False
_C.MODEL.memory_encoder_arch = ""
_C.MODEL.qval_qread_BN = False
_C.MODEL.normalize_key = False
_C.MODEL.p_scalar = 40.0
_C.MODEL.memory_feature_aggregation = False
_C.MODEL.memory_noLabel = False
_C.MODEL.att_voting = False
_C.MODEL.mask_foreground = False

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.n_iters = 30000
_C.TRAIN.n_batch = 2


_C.TRAIN.batch_size_per_gpu = 2
# epoch to start training. useful if continue from a checkpoint
_C.TRAIN.start_iter = 0
# iterations of each epoch (irrelevant to batch size)
_C.TRAIN.epoch_iters = 5000

_C.TRAIN.optim = "SGD"
_C.TRAIN.lr_encoder = 0.02
_C.TRAIN.lr_decoder = 0.02
# power in poly to drop LR
_C.TRAIN.lr_pow = 0.9
# momentum for sgd, beta1 for adam
_C.TRAIN.beta1 = 0.9
# weights regularizer
_C.TRAIN.weight_decay = 1e-4
# the weighting of deep supervision loss
_C.TRAIN.deep_sup_scale = 0.4
# fix bn params, only under finetuning
_C.TRAIN.fix_bn = False
# number of data loading workers
_C.TRAIN.workers = 16

# frequency to display
_C.TRAIN.disp_iter = 20
# manual seed
_C.TRAIN.seed = 304
_C.TRAIN.save_freq = 10000
_C.TRAIN.permute_labels = False

# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------
_C.VAL = CN()
# currently only supports 1
_C.VAL.n_batch = 1
_C.VAL.n_iters = 1000
# output visualization during validation
_C.VAL.visualize = False
# the checkpoint to evaluate on
_C.VAL.checkpoint = "iter_50000.pth"
_C.VAL.permute_labels = False
_C.VAL.n_runs = 5
_C.VAL.seed = 321

# -----------------------------------------------------------------------------
# Testing
# -----------------------------------------------------------------------------
_C.TEST = CN()
# currently only supports 1
_C.TEST.batch_size = 1
# the checkpoint to test on
_C.TEST.checkpoint = "epoch_20.pth"
# folder to output visualization results
_C.TEST.result = "./"
