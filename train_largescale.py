import torch
print(torch.__version__)


import os, argparse, sklearn

import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tensorboardX import SummaryWriter

from supervised_config import get_config
from image_iter import FaceDataset

from util.utils import separate_irse_bn_paras, separate_resnet_bn_paras, separate_mobilefacenet_bn_paras
from util.utils import get_val_data, perform_val, get_time, buffer_val, AverageMeter, train_accuracy

import time
# # from vit_pytorch_my import ViT_face
# # from vit_pytorch_my import ViTs_face
# from vit_pytorch_my.vit_face import ViT_face,ViT_face_landmark,ViT_face_landmark_patch8,ViT_face_landmark_largepatch,ViT_face_landmark_astoken,ViT_face_landmark_patch8_global,ViT_face_landmark_patch8_overlap,ViT_face_landmark_patch8_landmark_cla
# from vit_pytorch_my.vit_face import ViTs_face_overlap
# from vit_pytorch_my import iresnet,ada_iresnet
# from vit_pytorch_my.vits_face import ViTs_face
# from vit_pytorch_my.vit_myland import ViT_stn_land
# from vit_pytorch_my.swim_transformer import SwinTransformer
# from vit_pytorch_my.ViL import MsViT
# # from vit_pytorch_my.TNT import TNT,_create_tnt
# from vit_pytorch_my.reduce_swim_t import reduce_SwinTransformer


from face_pre_pro.ViT_face import face_landmark_4simmin_glo_loc, ViT_face_landmark_patch8
from IPython import embed
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
import pdb
from tqdm import tqdm
# from pthflops import count_ops
from ptflops import get_model_complexity_info
import numpy as np
#dis
import torch.distributed as dist
import torch.utils.data.distributed
from torch.nn.parallel import DistributedDataParallel as DDP
from util.mixup_my import Mixup
from timm.loss import SoftTargetCrossEntropy
import platform
def need_save(acc, highest_acc):
    do_save = False
    save_cnt = 0
    if acc[0] > 0.98:
        do_save = True
    for i, accuracy in enumerate(acc):
        if accuracy > highest_acc[i]:
            highest_acc[i] = accuracy
            do_save = True
        if i > 0 and accuracy >= highest_acc[i]-0.002:
            save_cnt += 1
    if save_cnt >= len(acc)*3/4 and acc[0]>0.99:
        do_save = True
    print("highest_acc:", highest_acc)
    return do_save
def schedule_lr(opt):
        for params in opt.param_groups:                 
            params['lr'] /= 3
def schedule_lr2(opt):
        for params in opt.param_groups:                 
            params['lr'] /= 2 


def param_groups_lrd_old(model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=.6):
    """
    Parameter groups for layer-wise lr decay
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    param_group_names = {}
    param_groups = {}
    # pdb.set_trace()
    # num_layers = len(model.blocks) + 1
    num_layers = len(model.transformer.layers) + 1
    

    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # no decay: all 1D parameters and model specific ones
        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay
            
        layer_id = get_layer_id_for_vit(n, num_layers)
        group_name = "layer_%d_%s" % (layer_id, g_decay)

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]

            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)
    # pdb.set_trace()
    # print("parameter groups: \n%s" % json.dumps(param_group_names, indent=2))

    return list(param_groups.values())

def param_groups_lrd(model, weight_decay=0.05, no_weight_decay_list=[],low_weight_decay_list=[], layer_decay=.6):
    """
    Parameter groups for layer-wise lr decay
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    param_group_names = {}
    param_groups = {}
    # pdb.set_trace()
    # num_layers = len(model.blocks) + 1
    num_layers = len(model.transformer.layers) + 1
    

    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # no decay: all 1D parameters and model specific ones
        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.
        elif n.startswith('stn'):
            g_decay = "low_decay"
            this_decay = 5e-2
        else:
            g_decay = "decay"
            this_decay = weight_decay
            
        layer_id = get_layer_id_for_vit(n, num_layers)
        group_name = "layer_%d_%s" % (layer_id, g_decay)

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]

            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)
    # pdb.set_trace()
    # print("parameter groups: \n%s" % json.dumps(param_group_names, indent=2))

    return list(param_groups.values())
def get_layer_id_for_vit(name, num_layers):
    """
    Assign a parameter with its layer id
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
    """
    
    if name in ['cls_token', 'pos_embedding']:
        return 0
    elif name.startswith('patch_to_embedding'):
        return 0
    elif name.startswith('stn'):
        # pdb.set_trace()
        return 0#0,num_layers
    elif name.startswith('output_layer'):
        # pdb.set_trace()
        return 0#0
    elif name.startswith('global_token'):
        # pdb.set_trace()
        return 0
    elif name.startswith('transformer'):#layers
        return int(name.split('.')[2]) + 1
    
    else:
        return num_layers
def load_part_checkpoint_landmark(path,model,pretrain_name=['stn','output'],freeze=True):
    # pdb.set_trace()
    pretrained_dict =  torch.load(path, map_location='cpu')
    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    # pretrained_dict=list(pretrained_dict.keys())
    back_remove=list(pretrained_dict.keys())
    for keys in back_remove:
        if 'dummy_orthogonal_classifier' in keys:
            # pdb.set_trace()
            continue
        pretrained_dict[keys.replace('module.','')]=pretrained_dict.pop(keys)

    # pdb.set_trace()
    # for name_space in pretrain_name:
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if pretrain_name[0] in k or pretrain_name[1] in k}
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if pretrain_name[0] in k or pretrain_name[1] in k}
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict) 
    # 3. load the new state dict
    model.load_state_dict(model_dict,strict=True)
    # model.encoder.output_layer.load_state_dict(pretrained_dict,strict=True)
    model_dict = model.state_dict()
    #freeze stn and output layer
    # pdb.set_trace()
    if freeze:
        for name, param in model.named_parameters():
            # if not param.requires_grad:
            if pretrain_name[0] in name or pretrain_name[1] in name:
                # pdb.set_trace()
                param.requires_grad = False

def load_part_checkpoint_landmark_fromdino(path,model,pretrain_name=['stn','output'],freeze=True):
    # pdb.set_trace()
    pretrained_dict =  torch.load(path, map_location='cpu')['teacher']
    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    # pretrained_dict=list(pretrained_dict.keys())
    back_remove=list(pretrained_dict.keys())
    for keys in back_remove:
        if 'dummy_orthogonal_classifier' in keys:
            # pdb.set_trace()
            continue
        pretrained_dict[keys.replace('module.','')]=pretrained_dict.pop(keys)
    back_remove=list(pretrained_dict.keys())
    for keys in back_remove:
        if 'dummy_orthogonal_classifier' in keys:
            # pdb.set_trace()
            continue
        pretrained_dict[keys.replace('backbone.','')]=pretrained_dict.pop(keys)

    # pdb.set_trace()
    # for name_space in pretrain_name:
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if pretrain_name[0] in k or pretrain_name[1] in k}
    # print(pretrained_dict.keys())
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if pretrain_name[0] in k or pretrain_name[1] in k}
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict) 
    # 3. load the new state dict
    # pdb.set_trace()
    model.load_state_dict(model_dict,strict=True)
    # model.encoder.output_layer.load_state_dict(pretrained_dict,strict=True)
    model_dict = model.state_dict()
    #freeze stn and output layer
    # pdb.set_trace()
    if freeze:
        for name, param in model.named_parameters():
            # if not param.requires_grad:
            if pretrain_name[0] in name or pretrain_name[1] in name:
                # pdb.set_trace()
                param.requires_grad = False

def load_part_checkpoint_landmark_fromsimmim(path,model,pretrain_name=['stn','output']):
    # pdb.set_trace()

    # best_model_dict = torch.load(BACKBONE_RESUME_ROOT,map_location=torch.device('cpu'))['model']
    # #remove 'backbone' from dino
    # back_remove=list(best_model_dict.keys())
    # for keys in back_remove:
    #     if 'dummy_orthogonal_classifier' in keys:
    #         # pdb.set_trace()
    #         continue
    #     best_model_dict[keys.replace('encoder.','')]=best_model_dict.pop(keys)

    # pdb.set_trace()
    pretrained_dict =  torch.load(path, map_location='cpu')['model']
    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    # pretrained_dict=list(pretrained_dict.keys())
    back_remove=list(pretrained_dict.keys())
    for keys in back_remove:
        if 'dummy_orthogonal_classifier' in keys:
            # pdb.set_trace()
            continue
        pretrained_dict[keys.replace('encoder.','')]=pretrained_dict.pop(keys)

    # pdb.set_trace()
    # for name_space in pretrain_name:
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if pretrain_name[0] in k or pretrain_name[1] in k}
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if pretrain_name[0] in k or pretrain_name[1] in k}
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict) 
    # 3. load the new state dict
    model.load_state_dict(model_dict,strict=True)
    # model.encoder.output_layer.load_state_dict(pretrained_dict,strict=True)
    model_dict = model.state_dict()
    #freeze stn and output layer
    for name, param in model.named_parameters():
        # if not param.requires_grad:
        if pretrain_name[0] in name or pretrain_name[1] in name:
            # pdb.set_trace()
            param.requires_grad = False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument("-w", "--workers_id", help="gpu ids or cpu", default='5', type=str)#3,4,7
    parser.add_argument("-e", "--epochs", help="training epochs", default=34, type=int)#125#35
    parser.add_argument("-b", "--batch_size", help="batch_size", default=200, type=int)#480,350#275#66 #76  80-84 176 na:160 pandora:400  tiny:240  #320
    #A100: 200 no landmark; A100: 200 land; 3090: 110 landmark 260-154    ,154*2*3=230*2*2=308*3*1,260
    # parser.add_argument("-d", "--data_mode", help="use which database, [casia, vgg, ms1m, retina, ms1mr]",default='retina', type=str)
    parser.add_argument("-n", "--net", 
            help="which network, ['VIT','VITs','VIT_land','VIT_land_8','ViT_stn_land','ViT_land_largepatch','ViL','Swim','Swim']",
                        default='VIT_land_8', type=str)
    
    parser.add_argument("-head", "--head", help="head type, ['Softmax', 'ArcFace', 'CosFace', 'SFaceLoss','CosFace_my']", default='CosFace', type=str)
    parser.add_argument("-t", "--target", help="verification targets", default='lfw,cfp_fp,agedb_30', type=str)#lfw,talfw,calfw,cplfw,cfp_fp,agedb_30
    parser.add_argument("-r", "--resume", help="resume model", default='/face_rec/ssl_results/webface_noshuffle_nopertur_vit_rerun_nocoor_ViTB/checkpoint0040.pth', type=str)
    
    
    parser.add_argument( "--pretrain_path", help="pretrain_path is the supervised model from stage 1", default='webface_196land_sp/Backbone_VIT_land_8_Epoch_34_Batch_327225_Time_2022-05-05-10-34_checkpoint.pth', type=str)
    parser.add_argument( "--model_dir", help="model_dir is the self-supervised model from stage 2(LAFS)", default='webface_ssl/checkpoint0040.pth', type=str)
    parser.add_argument( "--dataset_path", help="dataset_path is the path to the rec file (MS1M, WebFace4m)", default='webface_ssl/checkpoint0040.pth', type=str)
    parser.add_argument( "--eval_root", help="eval_root is the path to the evaluation files, using validation files from MS1MV3 ", default='webface_ssl/checkpoint0040.pth', type=str)

    
    
    parser.add_argument('--outdir', help="output dir", default='./results/3gpu_B_augall_again1', type=str)

    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-2,#5e-2
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',#2.9e-4,7.5e-5 1e-3#sgd 5e-4  #5e-5 layerwise 1e-4?
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',#5,7
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                "ViT-L_32", "ViT-H_14", "R50-ViT-B_16","ViT-B_32_small",'ViT-B_8'],
                    default="ViT-B_16", 
                    help="Which variant to use.")
    #mixup args
    parser.add_argument('--mixup', type=float, default=0.2,#0.8
                    help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=0.1,#1.0
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
    parser.add_argument('--smoothing', type=float, default=0, help='Label smoothing (default: 0.1)')
    #dis
    # parser.add_argument('--world-size', default=-1, type=int,
    #                 help='number of nodes for distributed training')
    # parser.add_argument('--rank', default=-1, type=int,
    #                     help='node rank for distributed training')
    # parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
    #                     help='url used to set up distributed training')
    # parser.add_argument('--dist-backend', default='nccl', type=str,
    #                     help='distributed backend')
    parser.add_argument('--fp16', default=True, type=bool,
                    help='mix precision')
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
    args = parser.parse_args()

    
    # args.outdir='/results/ms1mv3_largescale'

    
    try:
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
        dist_url = "tcp://{}:{}".format(os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"])
    except KeyError:
        world_size = 1
        rank = 0
        dist_url = "tcp://127.0.0.1:12584"
    print(rank,dist_url)
    args.rank=rank
    cfg = get_config(args)

    # #pretrain_path is the supervised model from stage 1
    # pretrain_path='/data/home/acw569/precheck/webface_196land_sp/Backbone_VIT_land_8_Epoch_34_Batch_327225_Time_2022-05-05-10-34_checkpoint.pth'
    # #model_dir is the self-supervised pretrained model path
    mobi_pretrain=args.pretrain_path
    
    webface=True
    with_land=True


    # disable them if you want to train without landmark supervision
    pre_land=False
    keep_land=False

    ###############

    SEED = cfg['SEED'] # random seed for reproduce results
    # torch.manual_seed(SEED)
    #dis
    print('world_size='+str(world_size))
    print('rank='+str(rank))
    print('loc_rank=')
    print(args.local_rank)
    dist.init_process_group(backend='nccl', init_method=dist_url, rank=rank, world_size=world_size)
    local_rank = args.local_rank
    # local_rank=3
    # pdb.set_trace()
    torch.cuda.set_device(local_rank)#local_rank
    # device_local=args.local_rank
    DATA_ROOT = cfg['DATA_ROOT'] # the parent root where your train/val/test data are stored
    EVAL_PATH = cfg['EVAL_PATH']
    


    
    WORK_PATH = cfg['WORK_PATH'] # the root to buffer your checkpoints and to log your train/val status
    BACKBONE_RESUME_ROOT = cfg['BACKBONE_RESUME_ROOT'] # the root to resume training from a saved checkpoint
    
    
        
    BACKBONE_NAME = cfg['BACKBONE_NAME']
    HEAD_NAME = cfg['HEAD_NAME'] # support:  ['Softmax', 'ArcFace', 'CosFace', 'SFaceLoss']

    INPUT_SIZE = cfg['INPUT_SIZE']
    EMBEDDING_SIZE = cfg['EMBEDDING_SIZE'] # feature dimension
    BATCH_SIZE = cfg['BATCH_SIZE']
    # pdb.set_trace()
    args.lr=cfg['acc_step']/480.0*args.lr*np.sqrt(world_size*BATCH_SIZE/336.0)*336



    # args.lr=BATCH_SIZE*cfg['acc_step']/480.0*args.lr*world_size


    # min_lr=1e-6**cfg['acc_step']/450.0*args.lr*world_size
    NUM_EPOCH = cfg['NUM_EPOCH']

    DEVICE = cfg['DEVICE']
    MULTI_GPU = cfg['MULTI_GPU'] # flag to use multiple GPUs
    GPU_ID = cfg['GPU_ID'] # specify your GPU ids
    print('GPU_ID', GPU_ID)
    TARGET = cfg['TARGET']
    print("=" * 60)
    print("Overall Configurations:")
    print(cfg)
    with open(os.path.join(WORK_PATH, 'config.txt'), 'w') as f:
        f.write(str(cfg))
    print("=" * 60)
    # pdb.set_trace()
    # dist.init_process_group(backend='nccl')
    # torch.cuda.set_device(1)#device_local
    # pdb.set_trace()
    GPU_ID=str(rank)
    # GPU_ID=str(3)
    print('GPU_ID', GPU_ID)
    writer = SummaryWriter(WORK_PATH) # writer for buffering intermedium results
    torch.backends.cudnn.benchmark = True



    # dataset = FaceDataset(os.path.join(args.dataset_path, 'train.rec'), rand_mirror=True,random_resizecrop=True,rand_au=True,config_str='rand-m2-mstd0.5-inc1') #MS1MV3
    dataset = FaceDataset(os.path.join(args.dataset_path, 'train.rec'), rand_mirror=True,random_resizecrop=True,rand_au=True,config_str='rand-m1-mstd0.5-inc1') # WebFace

    # dataset = FaceDataset(os.path.join(arg.dataset_path, 'train.rec'), rand_mirror=True,random_resizecrop=True,rand_au=False)
    #ms1m
    
    with open(os.path.join(DATA_ROOT, 'property'), 'r') as f:
        NUM_CLASS, h, w = [int(i) for i in f.read().split(',')] #VGG 8631
    NUM_CLASS=205990# webaface identities:205990; 
    # NUM_CLASS=93431# ms1m identities
    patch_size=8
    num_patches=196
    h, w=112,112


    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, shuffle=True)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,  num_workers=8, pin_memory=True,drop_last=True,sampler=train_sampler)#shuffle=True,

    print("Number of Training Classes: {}".format(NUM_CLASS))
    args.mixup_fn = None
    mixup_active = args.mixup > 0 #or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        args.mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=NUM_CLASS)#args.nb_classes)

    

    # pdb.set_trace()
    #======= model & loss & optimizer =======#
    #below is for iresnet---------
    # 'VIT_land_8':iresnet.iresnet100(dropout=0.1, fp16=True,GPU_ID=None, num_features=512,NUM_CLASS = NUM_CLASS)
    # 'VIT_land_8':ada_iresnet.build_model(model_name='ir_101')
    # -----------------------
    #
    BACKBONE_DICT = {
        'VIT_land_8': ViT_face_landmark_patch8(
                        loss_type = HEAD_NAME,
                        GPU_ID = None,
                        num_class = NUM_CLASS,
                        num_patches=num_patches,
                        image_size=112,
                        patch_size=patch_size,#8
                        dim=768,#512
                         depth=12,#20
                         heads=11,#8
                         mlp_dim=2048,
                         dropout=0.1,
                         emb_dropout=0.1,
                        with_land=with_land
                    )
                    
                    
                     }
    BACKBONE = BACKBONE_DICT[BACKBONE_NAME]
    
    # pdb.set_trace()
    landmarkcnn=None
    if pre_land==True:

        from einops import rearrange, repeat
        landmarkcnn=face_landmark_4simmin_glo_loc(loss_type = 'CosFace',
                            GPU_ID = None,
                            num_class = 30000,
                            num_patches=num_patches,
                            image_size=112,
                            patch_size=patch_size,#8
                            dim=512,#512
                            depth=12,#20
                            heads=11,#8
                            mlp_dim=2560,
                            dropout=0.1,
                            emb_dropout=0.1)
        landmarkcnn=landmarkcnn.cuda()
        load_part_checkpoint_landmark(path=args.pretrain_path,model=landmarkcnn,pretrain_name=['stn','output'])   
        landmarkcnn.eval()
        # if knowledge_dis:
        transf_cit = torch.nn.MSELoss()
    
    EMBEDDING_SIZE=768#int(embed_dim * 2 ** (self.num_layers - 1)) 384,768,640
    # print('# generator parameters:', sum(param.numel() for param in BACKBONE.parameters()))
    # pdb.set_trace()
    print("=" * 60)
    macs, params = get_model_complexity_info(BACKBONE, (3, 112, 112), as_strings=True,
                                           print_per_layer_stat=False, verbose=False)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    # inp = torch.rand(1,3,112,112)#.to(DEVICE)
    # count_ops(BACKBONE, inp)
    print("=" * 60)
    # print(BACKBONE)
    print("{} Backbone Generated".format(BACKBONE_NAME))
    print("=" * 60)
    # pdb.set_trace()
    if args.mixup_fn is not None:
        LOSS=SoftTargetCrossEntropy()#.cuda(args.gpu)
    else:
        LOSS = nn.CrossEntropyLoss()
    
    params_stn=[param for name, param in BACKBONE.named_parameters() if 'stn' in name]
    params_out=[param for name, param in BACKBONE.named_parameters() if 'output_layer' in name]
    # self.params_stn=[x for x in params_vir if 'stn' in x.keys()]
    # self.params_out=[x for x in params_vir if 'output_layer' in x.keys()]
    # pdb.set_trace()
    params_trans=[[name, param] for name, param in BACKBONE.named_parameters() if 'stn' not in name]
    params_trans=[param for name, param in params_trans if 'output_layer' not in name]

    # OPTIMIZER = optim.AdamW([
    #             {'params': params_trans},
    #             {'params': params_stn+params_out, 'weight_decay': 5e-2}#5e-2
    #         ], lr = args.lr,weight_decay = 1e-1)  # stage 1 setting
    

    #12 layers
    param_groups=param_groups_lrd(BACKBONE, 1e-1,
    no_weight_decay_list=[],
    layer_decay=0.58)
    
    OPTIMIZER = optim.AdamW(param_groups,
                            lr = args.lr,
                            # weight_decay = 1e-1,#0.05,5e-4#5e-2
                            # momentum = conf.momentum
                        )

    print("=" * 60)
    print(OPTIMIZER)
    print("Optimizer Generated")
    print("=" * 60)
    


    # load weight from the self-supervised pretrained checkpoint
    best_model_dict = torch.load(args.model_dir,map_location=torch.device('cpu'))['teacher']#['teacher'],['model']
    #remove 'backbone' from dino
    back_remove=list(best_model_dict.keys())
    for keys in back_remove:
        if 'dummy_orthogonal_classifier' in keys:
            # pdb.set_trace()
            continue
        best_model_dict[keys.replace('encoder.','').replace('backbone.','').replace('module.','')]=best_model_dict.pop(keys)
        # best_model_dict[keys.replace('backbone.','')]=best_model_dict.pop(keys)
        # best_model_dict[keys.replace('module.','')]=best_model_dict.pop(keys)
    # for keys in back_remove:
    #     if 'dummy_orthogonal_classifier' in keys:
    #         # pdb.set_trace()
    #         continue
    #     # best_model_dict[keys.replace('encoder.','')]=best_model_dict.pop(keys)
    #     best_model_dict[keys.replace('backbone.','')]=best_model_dict.pop(keys)
    #     # best_model_dict[keys.replace('module.','')]=best_model_dict.pop(keys)
    # pdb.set_trace()
    BACKBONE.load_state_dict(best_model_dict,strict=False)# fix this
    #load landmark part
    if with_land:
        # load_part_checkpoint_landmark_fromdino(path=mobi_pretrain,model=BACKBONE,pretrain_name=['stn','random'],freeze=False)    
        load_part_checkpoint_landmark(path=mobi_pretrain,model=BACKBONE,pretrain_name=['stn','output'],freeze=False)    
    # #simmin load
    # best_model_dict = torch.load(BACKBONE_RESUME_ROOT,map_location=torch.device('cpu'))['model']
    # #remove 'backbone' from dino
    # back_remove=list(best_model_dict.keys())
    # for keys in back_remove:
    #     if 'dummy_orthogonal_classifier' in keys:
    #         # pdb.set_trace()
    #         continue
    #     best_model_dict[keys.replace('encoder.','')]=best_model_dict.pop(keys)
    # # pdb.set_trace()
    # BACKBONE.load_state_dict(best_model_dict,strict=False)
    

    BACKBONE=BACKBONE.to(local_rank)
    BACKBONE = torch.nn.parallel.DistributedDataParallel(
        module=BACKBONE, broadcast_buffers=True, device_ids=[local_rank],output_device=local_rank,find_unused_parameters=True)
    # pdb.set_trace()
    # BACKBONE_RESUME_ROOT='/data/scratch/acw569/checkpoint/sp_check/ms1m_ViTB_landfromms1m_mobleDINO_ssl_5mixup_nolosss_5e4_fine_nonormandkickout_varyingloss
    # BACKBONE_RESUME_ROOT='/data/scratch/acw569/checkpoint/sp_check/ms1m_land_ms1mdata_VITB_22_realnoaug_standwd_1e4lr/Backbone_VIT_land_8_Epoch_22_Batch_214149_Time_2023-05-31-21-30_checkpoint.pth'
    if BACKBONE_RESUME_ROOT:
        print("=" * 60)
        print(BACKBONE_RESUME_ROOT)
        if os.path.isfile(BACKBONE_RESUME_ROOT):
            print("Loading Backbone Checkpoint '{}'".format(BACKBONE_RESUME_ROOT))
            # BACKBONE.load_state_dict(torch.load(BACKBONE_RESUME_ROOT),strict=True)
            # BACKBONE = torch.nn.DataParallel(BACKBONE)
            # # # BACKBONE.load_state_dict(torch.load(BACKBONE_RESUME_ROOT,map_location=torch.device('cpu'))['teacher'],strict=True)
            # pdb.set_trace()
            #simclr
            # BACKBONE.load_state_dict(torch.load(BACKBONE_RESUME_ROOT,map_location=torch.device('cpu'))['state_dict_glo'],strict=False)

            
        else:
            print("No Checkpoint Found at '{}' . Please Have a Check or Continue to Train from Scratch".format(BACKBONE_RESUME_ROOT))
        print("=" * 60)
    # pdb.set_trace()
    # if MULTI_GPU:
    #     # multi-GPU setting
    #     BACKBONE = nn.DataParallel(BACKBONE, device_ids = GPU_ID)
    #     BACKBONE = BACKBONE.to(DEVICE)
    # else:
    #     # single-GPU setting
    #     BACKBONE = BACKBONE.to(DEVICE)
    #dis
    # BACKBONE=BACKBONE.to(local_rank)
    # BACKBONE = torch.nn.parallel.DistributedDataParallel(
    #     module=BACKBONE, broadcast_buffers=True, device_ids=[local_rank],output_device=local_rank)
    # BACKBONE = DDP(BACKBONE, device_ids=[local_rank], output_device=local_rank)
    # BACKBONE = DistributedDataParallel(BACKBONE, device_ids=[args.local_rank])


    
    #======= train & validation & save checkpoint =======#
    # if 'Alienware' not in platform.node():
    vers = get_val_data(EVAL_PATH, TARGET)
    # vers = get_val_data(EVAL_PATH, TARGET)
    highest_acc = [0.0 for t in TARGET]
    DISP_FREQ = 2000 # frequency to display training loss & acc
    VER_FREQ = len(dataset)//(world_size*BATCH_SIZE*2)#4000

    batch = 0  # batch index

    losses = AverageMeter()
    top1 = AverageMeter()
    start_epoch=0
    # pdb.set_trace()
    from warmup_scheduler import GradualWarmupScheduler
    max_steps=(args.epochs-args.warmup_epochs-start_epoch)*len(dataset)//(cfg['acc_step']*BATCH_SIZE*world_size)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(OPTIMIZER,T_max= max_steps, eta_min=1e-6)#optimizer_stn,optimizer
    if args.warmup_epochs>0:

        scheduler = GradualWarmupScheduler(OPTIMIZER, multiplier=1, total_epoch=args.warmup_epochs*len(dataset)//(cfg['acc_step']*BATCH_SIZE*world_size), after_scheduler=scheduler)
    BACKBONE.train()  # set to training mode
    eval_step=0
    epoch=0

    if args.fp16:
        scaler=torch.cuda.amp.GradScaler()
    # if rank==0:
    #     # for params in OPTIMIZER.param_groups:
    #     #     lr = params['lr']
    #     #     break
    #     # print("Learning rate %f"%lr)
    #     # print("Perform Evaluation on", TARGET, ", and Save Checkpoints...")
    #     # acc = []
    #     for ver in vers:
    #         name, data_set, issame = ver
    #         if rank==0:
    #             if name=='1234':#'lfw',agebd#agedb_30,cfp_fp
    #                 visualize=True
    #             else:
    #                 visualize=False
    #         else:
    #             visualize=False
    #         # pdb.set_trace()
    #         # accuracy, std, xnorm, best_threshold, roc_curve = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, data_set, issame,epoch=epoch,step=batch,logpath=args.outdir,visualize=visualize)
    #         # buffer_val(writer, name, accuracy, std, xnorm, best_threshold, roc_curve, batch + 1)
    #         accuracy, std, xnorm, best_threshold, roc_curve = perform_val(MULTI_GPU, local_rank, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, data_set, issame,epoch=epoch,step=batch,logpath=args.outdir,visualize=visualize,pre_land=pre_land,keep_land=keep_land,landmarkcnn=landmarkcnn)
    #         buffer_val(writer, name, accuracy, std, xnorm, best_threshold, roc_curve, batch + 1)
    #         print('[%s][%d]XNorm: %1.5f' % (name, batch+1, xnorm))
    #         print('[%s][%d]Accuracy-Flip: %1.5f+-%1.5f' % (name, batch+1, accuracy, std))
    #         print('[%s][%d]Best-Threshold: %1.5f' % (name, batch+1, best_threshold))
    #         # acc.append(accuracy)
    #     BACKBONE.train()
    #     sys.exit()
    
    for epoch in range(NUM_EPOCH): # start training process
        # if epoch==25:
        #     dataset = FaceDataset(os.path.join(DATA_ROOT, 'train.rec'), rand_mirror=True,random_resizecrop=True,rand_au=True,config_str='rand-m1-mstd0.5-inc1')
        #     trainloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=6, pin_memory=True,drop_last=True)
        #     # schedule_lr(OPTIMIZER)
        # if epoch==30:
        #     dataset = FaceDataset(os.path.join(DATA_ROOT, 'train.rec'), rand_mirror=True,random_resizecrop=True,rand_au=False)
        #     trainloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=6, pin_memory=True,drop_last=True)
        # if epoch==20:
        #     schedule_lr(OPTIMIZER)
        # if epoch==25:
        #     schedule_lr2(OPTIMIZER)
        # lr_scheduler.step(epoch)
        trainloader.sampler.set_epoch(epoch)
        last_time = time.time()
        # if rank==0:
        #     pdb.set_trace()
        for inputs, labels in tqdm(iter(trainloader),total=len(dataset)//BATCH_SIZE//world_size):
            # if rank==0:
            #     print('batch='+str(batch)+',max='+str(len(dataset)//BATCH_SIZE//world_size))
            # pdb.set_trace()
            # compute output
            inputs = inputs.cuda()/255.0*2-1 #to(DEVICE)
            labels = labels.cuda().long()#to(DEVICE).long()
            labels_ori=labels.clone()
            # pdb.set_trace()
            if args.mixup_fn is not None:
                if len(inputs)%2!=0:
                    inputs=inputs[:-1]#.float()
                    labels=labels[:-1]
                    labels_ori=labels_ori[:-1]
                    print('drop one')
                    # continue
                # pdb.set_trace()
                inputs, labels = args.mixup_fn(inputs.float(), labels,device=local_rank)
            if args.fp16:
                with torch.cuda.amp.autocast():
                    # outputs, emb,landmark_cls = BACKBONE(inputs.float(), labels)
                    # pdb.set_trace()
                    if pre_land==True:
                        land_label,img_reconstructed=landmarkcnn(inputs.float())#div 255/2
                        # land_label,img_reconstructed=landmarkcnn(images[0])
                        #reconstructed image to embedding
                        if not keep_land:
                            inputs = rearrange(img_reconstructed, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = landmarkcnn.patch_size, p2 = landmarkcnn.patch_size)
                        # else:
                        #     inputs=img_reconstructed
                    outputs, pred_land = BACKBONE(inputs.float(), labels)
                    # pdb.set_trace()
                    # nonnan_index=~torch.isnan(outputs)
                    # loss = LOSS(outputs[nonnan_index], labels[nonnan_index])
                    # outputs=torch.nan_to_num(outputs, nan=1e-6)
                    loss = LOSS(outputs, labels)# sphereface, adaface please comment this
                    # pdb.set_trace()
                    if pre_land and keep_land:
                        # label_land,rec_img=landmarkcnn(img)
                        # pdb.set_trace()
                        loss_land=transf_cit(land_label/111.0,pred_land/111.0)
                        if 13>=epoch>7:
                            land_loss_control=100
                        elif 20>=epoch>13:
                            land_loss_control=1
                        elif 27>=epoch>20:
                            land_loss_control=0.11
                        elif epoch>27:
                            land_loss_control=0
                        else:
                            land_loss_control=1000
                        loss=loss+land_loss_control*loss_land
                    #
                    # loss=outputs
                    # pdb.set_trace()
                    # landmark_cls_loss=LOSS(landmark_cls,labels)
                    # loss=loss+landmark_cls_loss
                    if cfg['acc_step'] > 1:
                            loss = loss / cfg['acc_step']
            else:
                outputs, emb = BACKBONE(inputs.float(), labels)
                # pdb.set_trace()
                loss = LOSS(outputs, labels)
                #landmark cls loss
                
                if cfg['acc_step'] > 1:
                        loss = loss / cfg['acc_step']
            
            #print("outputs", outputs, outputs.data)
            # measure accuracy and record loss
            # prec1= train_accuracy(outputs.data, labels_ori, topk = (1,))
            # if torch.isnan(loss):
            #     print('nan occurs, skip for this batch')
            #     continue

            losses.update(loss.data.item(), inputs.size(0))
            # top1.update(prec1.data.item(), inputs.size(0))


            # compute gradient and do SGD step
            
            if args.fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if (batch+1)% cfg['acc_step'] ==0:
                # pdb.set_trace()
                # print(BACKBONE.output_layer.weight.grad)
                # writer.add_scalar('mean_grad_outlayer', BACKBONE.output_layer.weight.grad.mean(), step)
                if args.fp16:
                    #resnet
                    # scaler.unscale_(OPTIMIZER)
                    if torch.isnan(loss):
                        torch.nn.utils.clip_grad_norm_(BACKBONE.parameters(), 5)
                    scaler.step(OPTIMIZER)
                    scaler.update()
                    OPTIMIZER.zero_grad()
                else:
                    # if torch.isnan(loss):
                    # torch.nn.utils.clip_grad_norm_(BACKBONE.parameters(), max_norm=10,norm_type=2)
                    OPTIMIZER.step()
                    OPTIMIZER.zero_grad()
                    # OPTIMIZER_stn.step()
                    # OPTIMIZER_stn.zero_grad()
                    # with amp.scale_loss(loss, OPTIMIZER) as scaled_loss:
                    #     scaled_loss.backward()
                scheduler.step()
                # scheduler_warmup.step()
                eval_step+=1
            
            # dispaly training loss & acc every DISP_FREQ (buffer for visualization)
            if rank == 0:
                if ((batch -1) % DISP_FREQ == 0) and batch != 0:
                    epoch_loss = losses.avg
                    epoch_acc = top1.avg
                    writer.add_scalar("Training/Training_Loss", epoch_loss, batch + 1)
                    # writer.add_scalar("Training/landmark_cls_loss", landmark_cls_loss.item(), batch + 1)
                    writer.add_scalar("Training/Training_Accuracy", epoch_acc, batch + 1)

                    batch_time = time.time() - last_time
                    last_time = time.time()

                    print('Epoch {} Batch {}\t'
                        'Speed: {speed:.2f} samples/s\t'
                        'Training Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Training Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                        epoch + 1, batch + 1, speed=inputs.size(0) * DISP_FREQ / float(batch_time),
                        loss=losses, top1=top1))
                    # if loss_land is not None:
                    if keep_land:
                        print(loss_land)
                    # print('landmark_cls_loss:'+str(landmark_cls_loss.item()))
                    #print("=" * 60)
                    losses = AverageMeter()
                    top1 = AverageMeter()
                    for params in OPTIMIZER.param_groups:
                        lr = params['lr']
                        break
                    writer.add_scalar("LR", lr, batch + 1)
                # pdb.set_trace()
                if ((eval_step -2) % (VER_FREQ//cfg['acc_step']) == 1) and (batch+1)% cfg['acc_step'] ==0:# and batch != 0: #perform validation & save checkpoints (buffer for visualization)
                    for params in OPTIMIZER.param_groups:
                        lr = params['lr']
                        break
                    print("Learning rate %f"%lr)
                    print("Perform Evaluation on", TARGET, ", and Save Checkpoints...")
                    acc = []
                    if 'Alienware' in platform.node():
                        #args.outdir='./results/49_12vit_lr5e4_randaam_2_droppath1_margin35_lr3e3'
                        vers = get_val_data(EVAL_PATH, TARGET)
                    for ver in vers:
                        name, data_set, issame = ver
                        if rank==0:
                            if name=='cfpfp1':#'lfw',agebd#agedb_30,cfp_fp
                                visualize=True
                            else:
                                visualize=False
                        else:
                            visualize=False
                        # pdb.set_trace()
                        accuracy, std, xnorm, best_threshold, roc_curve = perform_val(MULTI_GPU, local_rank, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, data_set, issame,epoch=epoch,step=batch,logpath=args.outdir,visualize=visualize,pre_land=pre_land,keep_land=keep_land,landmarkcnn=landmarkcnn)
                        buffer_val(writer, name, accuracy, std, xnorm, best_threshold, roc_curve, batch + 1)
                        print('[%s][%d]XNorm: %1.5f' % (name, batch+1, xnorm))
                        print('[%s][%d]Accuracy-Flip: %1.5f+-%1.5f' % (name, batch+1, accuracy, std))
                        print('[%s][%d]Best-Threshold: %1.5f' % (name, batch+1, best_threshold))
                        acc.append(accuracy)
                    if 'Alienware' in platform.node():
                        #args.outdir='./results/49_12vit_lr5e4_randaam_2_droppath1_margin35_lr3e3'
                        del vers
                    # save checkpoints per epoch
                    if need_save(acc, highest_acc) and rank==0:
                        if MULTI_GPU:
                            torch.save(BACKBONE.module.state_dict(), os.path.join(WORK_PATH, "Backbone_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(BACKBONE_NAME, epoch + 1, batch + 1, get_time())))
                        else:
                            torch.save(BACKBONE.state_dict(), os.path.join(WORK_PATH, "Backbone_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(BACKBONE_NAME, epoch + 1, batch + 1, get_time())))
                    BACKBONE.train()  # set to training mode
                    # BACKBONE=BACKBONE.to(local_rank)

            batch += 1 # batch index


