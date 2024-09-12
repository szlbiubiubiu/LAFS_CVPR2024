import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn

from torch.nn import Parameter
from IPython import embed
import numpy as np

import pdb
from timm.models.layers import DropPath
# import face_alignment
# from vit_pytorch_my import FAN_network
import torchvision.models as models
import math
import os
import logging
from torch.nn.functional import normalize, linear
import torch.distributed as dist

from torch.autograd import Variable
MIN_NUM_PATCHES=15

from face_pre_pro.mobilenet import MobileNetV3_backbone
from timm.models.layers import trunc_normal_
class CosFace(nn.Module):
    r"""Implement of CosFace (https://arxiv.org/pdf/1801.09414.pdf):
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        device_id: the ID of GPU where the model will be trained by model parallel.
                       if device_id=None, it will be trained on CPU without model parallel.
        s: norm of input feature
        m: margin
        cos(theta)-m
    """

    def __init__(self, in_features, out_features, device_id, s=64.0, m=0.4):#0.35
        super(CosFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device_id = device_id
        self.s = s
        self.m = m
        print("self.device_id", self.device_id)
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------

        if self.device_id == None:
            cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        else:
            x = input
            sub_weights = torch.chunk(self.weight, len(self.device_id), dim=0)
            temp_x = x.cuda(self.device_id[0])
            weight = sub_weights[0].cuda(self.device_id[0])
            cosine = F.linear(F.normalize(temp_x), F.normalize(weight))
            for i in range(1, len(self.device_id)):
                temp_x = x.cuda(self.device_id[i])
                weight = sub_weights[i].cuda(self.device_id[i])
                cosine = torch.cat((cosine, F.linear(F.normalize(temp_x), F.normalize(weight)).cuda(self.device_id[0])),
                                   dim=1)
        phi = cosine - self.m
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size())
        # pdb.set_trace()
        if len(label.shape)>1:
            if self.device_id == None:
                one_hot=label.cuda()
            else:
                one_hot=label.cuda(self.device_id[0])
        else:
            if self.device_id != None:
                one_hot = one_hot.cuda(self.device_id[0])
            # one_hot = one_hot.cuda() if cosine.is_cuda else one_hot

                one_hot.scatter_(1, label.cuda(self.device_id[0]).view(-1, 1).long(), 1)
            else:
                in_device=label.device
                one_hot=one_hot.to(in_device)
                one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + (
                    (1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features = ' + str(self.in_features) \
               + ', out_features = ' + str(self.out_features) \
               + ', s = ' + str(self.s) \
               + ', m = ' + str(self.m) + ')'



class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x
class Residual_droppath(nn.Module):
    def __init__(self, fn,drop_path_rate=0.1):
        super().__init__()
        self.fn = fn
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
    def forward(self, x, **kwargs):
        return self.drop_path(self.fn(x, **kwargs)) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.,last_dim=None):
        super().__init__()
        if last_dim==None:
            last_dim=dim
        
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, last_dim),
            nn.Dropout(dropout)
        )
        # else:
            
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        # self.to_qkv = nn.Linear(dim, np.int64(inner_dim/2) , bias = False)
        # self.to_out = nn.Sequential(
        #     nn.Linear(np.int64(inner_dim/6), dim),
        #     nn.Dropout(dropout)
        # )
        self.attention_score = 0

    def forward(self, x, mask = None):
        # pdb.set_trace()
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max
        #embed()
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)
        # pdb.set_trace()
        self.attention_score=attn.detach()
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)

        return out

class Transformer(nn.Module):
    # def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
    #     super().__init__()
    #     self.layers = nn.ModuleList([])
    #     for _ in range(depth):
    #         self.layers.append(nn.ModuleList([
    #             Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
    #             Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
    #         ]))
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout,last_dim=None):
        super().__init__()
        self.layers = nn.ModuleList([])
        if last_dim==None:
            last_dim=dim
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual_droppath(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual_droppath(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
        # pdb.set_trace()
        # self.layers.append(nn.ModuleList([
        #         Residual_droppath(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
        #         Residual_droppath(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout,last_dim=last_dim)))
        #     ]))
    def forward(self, x, mask = None):
        for attn, ff in self.layers:
            x = attn(x, mask = mask)
            #embed()
            x = ff(x)
        return x

def bn_init(bn):
    bn.weight.data.fill_(1)
    bn.bias.data.zero_()
def create_e_matrix(n):
    end = torch.zeros((n*n,n))
    for i in range(n):
        end[i * n:(i + 1) * n, i] = 1
    start = torch.zeros(n, n)
    for i in range(n):
        start[i, i] = 1
    start = start.repeat(n,1)
    return start,end

class GNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(GNN, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        # GNN Matrix: E x N
        # Start Matrix Item:  define the source node of one edge
        # End Matrix Item:  define the target node of one edge
        # Algorithm details in Residual Gated Graph Convnets: arXiv preprint arXiv:1711.07553
        # or Benchmarking Graph Neural Networks: arXiv preprint arXiv:2003.00982v3

        start, end = create_e_matrix(self.num_classes)
        self.start = Variable(start, requires_grad=False)
        self.end = Variable(end, requires_grad=False)

        dim_in = self.in_channels
        dim_out = self.in_channels

        self.U1 = nn.Linear(dim_in, dim_out, bias=False)
        self.V1 = nn.Linear(dim_in, dim_out, bias=False)
        self.A1 = nn.Linear(dim_in, dim_out, bias=False)
        self.B1 = nn.Linear(dim_in, dim_out, bias=False)
        self.E1 = nn.Linear(dim_in, dim_out, bias=False)

        # self.U2 = nn.Linear(dim_in, dim_out, bias=False)
        # self.V2 = nn.Linear(dim_in, dim_out, bias=False)
        # self.A2 = nn.Linear(dim_in, dim_out, bias=False)
        # self.B2 = nn.Linear(dim_in, dim_out, bias=False)
        # self.E2 = nn.Linear(dim_in, dim_out, bias=False)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(2)
        self.bnv1 = nn.BatchNorm1d(num_classes)
        self.bne1 = nn.BatchNorm1d(num_classes*num_classes)

        # self.bnv2 = nn.BatchNorm1d(num_classes)
        # self.bne2 = nn.BatchNorm1d(num_classes * num_classes)

        self.act = nn.ReLU()

        self.init_weights_linear(dim_in, 1)

    def init_weights_linear(self, dim_in, gain):
        # conv1
        scale = gain * np.sqrt(2.0 / dim_in)
        self.U1.weight.data.normal_(0, scale)
        self.V1.weight.data.normal_(0, scale)
        self.A1.weight.data.normal_(0, scale)
        self.B1.weight.data.normal_(0, scale)
        self.E1.weight.data.normal_(0, scale)

        # self.U2.weight.data.normal_(0, scale)
        # self.V2.weight.data.normal_(0, scale)
        # self.A2.weight.data.normal_(0, scale)
        # self.B2.weight.data.normal_(0, scale)
        # self.E2.weight.data.normal_(0, scale)

        bn_init(self.bnv1)
        bn_init(self.bne1)
        # bn_init(self.bnv2)
        # bn_init(self.bne2)

    def forward(self, x, edge):
        # device
        dev = x.get_device()
        if dev >= 0:
            start = self.start.to(dev)
            end = self.end.to(dev)

        # GNN Layer 1:
        res = x
        Vix = self.A1(x)  # V x d_out
        Vjx = self.B1(x)  # V x d_out
        e = self.E1(edge)  # E x d_out
        edge = edge + self.act(self.bne1(torch.einsum('ev, bvc -> bec', (end, Vix)) + torch.einsum('ev, bvc -> bec',(start, Vjx)) + e))  # E x d_out

        e = self.sigmoid(edge)
        b, _, c = e.shape
        e = e.view(b,self.num_classes, self.num_classes, c)
        e = self.softmax(e)
        e = e.view(b, -1, c)

        Ujx = self.V1(x)  # V x H_out
        Ujx = torch.einsum('ev, bvc -> bec', (start, Ujx))  # E x H_out
        Uix = self.U1(x)  # V x H_out
        x = Uix + torch.einsum('ve, bec -> bvc', (end.t(), e * Ujx)) / self.num_classes  # V x H_out
        x = self.act(res + self.bnv1(x))
        res = x

        # # GNN Layer 2:
        # Vix = self.A2(x)  # V x d_out
        # Vjx = self.B2(x)  # V x d_out
        # e = self.E2(edge)  # E x d_out
        # edge = edge + self.act(self.bne2(torch.einsum('ev, bvc -> bec', (end, Vix)) + torch.einsum('ev, bvc -> bec', (start, Vjx)) + e))  # E x d_out

        # e = self.sigmoid(edge)
        # b, _, c = e.shape
        # e = e.view(b, self.num_classes, self.num_classes, c)
        # e = self.softmax(e)
        # e = e.view(b, -1, c)

        # Ujx = self.V2(x)  # V x H_out
        # Ujx = torch.einsum('ev, bvc -> bec', (start, Ujx))  # E x H_out
        # Uix = self.U2(x)  # V x H_out
        # x = Uix + torch.einsum('ve, bec -> bvc', (end.t(), e * Ujx)) / self.num_classes  # V x H_out
        # x = self.act(res + self.bnv2(x))
        return x, edge

class ViT_face_landmark_patch8_global(nn.Module):
    def __init__(self, *, loss_type, GPU_ID, num_class, image_size, patch_size, dim, depth, heads, mlp_dim, pool = 'cls', num_patches=None, channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        # assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        if num_patches==None:
            num_patches = (image_size // patch_size) ** 2
        # num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        assert num_patches > MIN_NUM_PATCHES, f'your number of patches ({num_patches}) is way too small for attention to be effective (at least 16). Try decreasing your patch size'
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        # pdb.set_trace()
        self.patch_size = patch_size

        # self.row_num=int(np.sqrt(num_patches)/2)#49
        self.row_num=int(np.sqrt(num_patches))#196
        self.stn=MobileNetV3_backbone(mode='large')
        # # # pdb.set_trace()
        # # # self.stn= ViT_face_stn_patch8(
        # # #                  loss_type = 'None',
        # # #                  GPU_ID = GPU_ID,
        # # #                  num_class = num_class,
        # # #                  image_size=112,
        # # #                  patch_size=8,#8
        # # #                  dim=96,#512
        # # #                  depth=12,#20
        # # #                  heads=3,#8
        # # #                  mlp_dim=1024,
        # # #                  dropout=0.1,
        # # #                  emb_dropout=0.1
        # # #              )
        # # #resnet
        # self.stn=models.resnet50()
        # self.stn.fc=nn.Sequential()
        # # # hybrid_dimension=50
        # # # drop_ratio=0.9

        self.output_layer = nn.Sequential(
            nn.Dropout(p=0.5),    # refer to paper section 6
            nn.Linear(160, self.row_num*self.row_num*2),#2048
        )
        self.global_token = nn.Sequential(
            nn.Dropout(p=0.5),    # refer to paper section 6
            nn.Linear(160, dim),# 
        )
        # self.output_layer = nn.Linear(96, int(self.row_num*self.row_num*2))#49*2,6        mobilenet 96 irse:128
        # self.patch_shape=torch.tensor([2*patch_size,2*patch_size])#49
        self.patch_shape=torch.tensor([patch_size,patch_size])#196
        self.theta=0

        # self.drop_2d=torch.nn.Dropout2d(p=0.1)

        self.dim=dim
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()
        self.sigmoid=nn.Sigmoid()
        self.mlp_head = nn.Sequential(
            # nn.Dropout(p=0.1),
            # nn.Linear(dim,512),
            nn.LayerNorm(dim),#nn.Identity()
        )
        self.loss_type = loss_type
        self.GPU_ID = GPU_ID
        # pdb.set_trace()
        # self.edge_gen=nn.Linear(160, self.row_num*self.row_num*2**2)
        # self.edge_gen=GEM(in_channels=dim,num_classes=num_class)
        # self.gnn=GNN(in_channels=dim,num_classes=145,neighbor_num=15)
        if self.loss_type == 'None':
            print("no loss for vit_face")
        else:
            if self.loss_type == 'Softmax':
                self.loss = Softmax(in_features=dim, out_features=num_class, device_id=self.GPU_ID)
            elif self.loss_type == 'CosFace':
                self.loss = CosFace(in_features=dim, out_features=num_class, device_id=self.GPU_ID,m=0.4)
            elif self.loss_type == 'ArcFace':
                self.loss = ArcFace(in_features=dim, out_features=num_class, device_id=self.GPU_ID)
            elif self.loss_type == 'SFace':
                self.loss = SFaceLoss(in_features=dim, out_features=num_class, device_id=self.GPU_ID)

    def forward(self, x, image_noaug=None,label= None , mask = None,visualize=False,save_token=False,Random_prob=False,ran_sample=False,glo_diff=False):
        p = self.patch_size
        # x=x/255.0*2-1  #no mean
        # img,fdsa=self.stn(img)
        # pdb.set_trace()
        # x_patch=x.clone()
        # x=x.detach()
        # pdb.set_trace()
        if image_noaug is not None:
            x_aug=x
            x=image_noaug
            
        theta0=self.stn(x)#.forward(x)            #with original stn
        # pdb.set_trace()

        theta0 = theta0.mean(dim=(-2, -1))#average pooling
        theta=self.output_layer(theta0)
        # pdb.set_trace()
        # edge=self.edge_gen(theta0)
        # GCN_out=
        glo_token=self.global_token(theta0).view(-1,1,self.dim)
        
        # #stn
        # # x,asdf=self.stn(x)
        # theta = theta.view(-1, 2, 3)
        # grid = F.affine_grid(theta, x.size())
        # x = F.grid_sample(x, grid)
        #landmark
        # pdb.set_trace()
        #min max scale
        t_max=torch.max(theta,1)[0]#.repeat(1,49*2)
        t_max=torch.unsqueeze(t_max,dim=1).repeat(1,self.row_num*self.row_num*2)
        t_min=torch.min(theta,1)[0]#.repeat(1,49*2)
        t_min=torch.unsqueeze(t_min,dim=1).repeat(1,self.row_num*self.row_num*2)
        theta=(theta-t_min)/(t_max-t_min)*111

        




        # #sigmoid scale
        # theta=self.sigmoid(theta)*111*2-111*0.5
        # theta=(self.sig(theta)*2-0.5)*(self.image_size-1)
        # theta=self.sig(theta)*111
        # theta=theta.round()
        # theta=theta.type(torch.int32)
        # pdb.set_trace()
        theta=theta.view(-1,self.row_num*self.row_num,2)
        self.theta=theta
        theta=theta.detach()
        hbs=np.int64(theta.shape[0]/2)
        # pdb.set_trace()
        if  glo_diff:
            theta0=theta[:hbs]
            theta=theta[hbs:]#.unsqueece(0)
            # theta0=torch.unsqueeze(theta0, 0)
            # theta=torch.unsqueeze(theta, 0)
        if Random_prob:
            # pdb.set_trace()
            # if not glo_diff:
            prob=torch.randn(theta.shape)*2#*12# 10 pixel 
            theta=theta+prob.cuda()
            # else:
            #     prob=torch.randn(theta[1].shape)*3#*12# 10 pixel 
            #     theta[1]=theta[1]+prob.cuda()
            # if not return_prob:
            b,c,fea=theta.shape
            if ran_sample:
                extract_id=torch.randint(0,c,(b,36,1)).cuda()
                keep_num=36
            else:
                extract_id=torch.randint(0,c,(b,self.row_num*self.row_num,1)).cuda()
                keep_num=self.row_num*self.row_num
            extract_id=extract_id.repeat(1,1,2)
            #extract landmarks
            # for i in range(b):
            # extract_id=extract_id.view(592,25,1)
            out_theta=torch.gather(theta, 1, extract_id)
            
            # theta[0][extract_id[0,:,0]][:,1]==out_theta[0][:,1]
            theta=out_theta
            # keep_num=25
            # self.theta=theta#.detach()
            # pdb.set_trace()

            if keep_num is not None:
                num_land=keep_num
            else:
                num_land=theta.shape[-2]#int(self.row_num*self.row_num)
        else:
            num_land=theta.shape[-2]
        #.detach()
        # pdb.set_trace()
        if  glo_diff:
            theta=torch.cat([theta0,theta])
            # theta=theta[1]
        if image_noaug is not None:
            x=extract_patches_pytorch_gridsample(x_aug,theta[:,:num_land],patch_shape=self.patch_shape,num_landm=num_land)
        else:
            x=extract_patches_pytorch_gridsample(x,theta[:,:num_land],patch_shape=self.patch_shape,num_landm=num_land)
        # pdb.set_trace()
        # x=x.detach()
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape
        # pdb.set_trace()
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        # x = torch.cat((cls_tokens, x), dim=1)
        x = torch.cat((glo_token, x), dim=1)
        # x = torch.cat((x,glo_token), dim=1)
        # x=self.drop_2d(x)
        x += self.pos_embedding[:, :(n + 1)]
        # x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x, mask)
        # pdb.set_trace()
        # x=self.gnn(x)
        if save_token==True:
            tokens=x[:,1:]
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        # x = x[:,1:,:].mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        emb = self.mlp_head(x)
        
        
        if save_token:
            return emb,tokens,self.theta
        if label is not None:
            x = self.loss(emb, label)
            return x, self.theta
            # return x, emb
        else:
            if visualize==True:
                return emb,self.theta
            else:
                return emb
            # return emb

class ViT_face_landmark_patch8(nn.Module):
    def __init__(self, *, loss_type, GPU_ID, num_class, image_size, patch_size, dim, depth, heads, mlp_dim, pool = 'cls',num_patches=None, channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.,fp16=True,with_land=False,use_standcoord=False,
    Random_prob=False,shuffle=False):
        super().__init__()
        # assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        if num_patches==None:
            num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        assert num_patches > MIN_NUM_PATCHES, f'your number of patches ({num_patches}) is way too small for attention to be effective (at least 16). Try decreasing your patch size'
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        # # pdb.set_trace()
        self.patch_size = patch_size
        self.fp16=fp16
        self.num_patches=num_patches
        self.row_num=int(np.sqrt(num_patches)/2)#49
        self.row_num=int(np.sqrt(num_patches))#196
        self.with_land=with_land
        if with_land:
            self.stn=MobileNetV3_backbone(mode='large')
            # # # pdb.set_trace()
            # # # self.stn= ViT_face_stn_patch8(
            # # #                  loss_type = 'None',
            # # #                  GPU_ID = GPU_ID,
            # # #                  num_class = num_class,
            # # #                  image_size=112,
            # # #                  patch_size=8,#8
            # # #                  dim=96,#512
            # # #                  depth=12,#20
            # # #                  heads=3,#8
            # # #                  mlp_dim=1024,
            # # #                  dropout=0.1,
            # # #                  emb_dropout=0.1
            # # #              )
            # # #resnet
            # # self.stn=models.resnet50()
            # # self.stn.fc=nn.Sequential()
            # # # hybrid_dimension=50
            # # # drop_ratio=0.9

            self.output_layer = nn.Sequential(
                nn.Dropout(p=0.5),    # refer to paper section 6
                nn.Linear(160, self.row_num*self.row_num*2),#2048
            )
            
            # self.output_layer = nn.Linear(96, int(self.row_num*self.row_num*2))#49*2,6        mobilenet 96 irse:128
            # self.patch_shape=torch.tensor([2*patch_size,2*patch_size])#49
        self.patch_shape=torch.tensor([patch_size,patch_size])#196
        self.theta=0

        self.drop_2d=torch.nn.Dropout2d(p=0.1)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            # nn.Dropout(p=0.1),
            # nn.Linear(dim,512),
            nn.LayerNorm(dim),#nn.Identity()
        )
        self.loss_type = loss_type
        self.GPU_ID = GPU_ID
        self.sigmoid = nn.Sigmoid()
        self.use_standcoord=use_standcoord
        self.Random_prob=Random_prob
        self.shuffle=shuffle
        if self.use_standcoord:
            range_coor=torch.arange(0,14)*8+4#[0,14]-->[4,12,20,108]
            x,y=torch.meshgrid(range_coor,range_coor)
            self.standard_coord=torch.stack((x,y),2).view(1,-1,2)
        
        if self.loss_type == 'None':
            print("no loss for vit_face")
        else:
            if self.loss_type == 'Softmax':
                self.loss = Softmax(in_features=dim, out_features=num_class, device_id=self.GPU_ID)
            elif self.loss_type == 'CosFace':
                
                # # # pdb.set_trace()
                # from vit_pytorch_my.partial_fc import PartialFCAdamW
                
                # self.loss = PartialFCAdamW(
                #     CosFace_arcimplement(), dim, num_class, 
                #     1.0, self.fp16)
                # # module_partial_fc.train().cuda()
                

                self.loss = CosFace(in_features=dim, out_features=num_class, device_id=self.GPU_ID,m=0.4)
            elif self.loss_type == 'ArcFace':
                self.loss = ArcFace(in_features=dim, out_features=num_class, device_id=self.GPU_ID)
            elif self.loss_type == 'SFace':
                self.loss = SFaceLoss(in_features=dim, out_features=num_class, device_id=self.GPU_ID)

    def forward(self, x, label= None , mask = None,visualize=False,save_token=False,opt=None,keep_num=None,glo_diff=False):
        p = self.patch_size

        #get the image shape
        b,imgshape=x.shape[0],x.shape[-2]
        if self.num_patches==144 and imgshape==112:
            keep_num=self.num_patches
        else:
            keep_num=(imgshape//p)**2
        # x=x/255.0*2-1  #no mean
        # img,fdsa=self.stn(img)
        # pdb.set_trace()
        # x_patch=x.clone()
        # x=x.detach()
        

        if keep_num is not None:
            num_land=keep_num
        else:
            num_land=int(self.row_num*self.row_num)
        if self.with_land:
            theta=self.stn(x)#.forward(x)            #with original stn
            # pdb.set_trace()

            theta = theta.mean(dim=(-2, -1))#average pooling   for cnn
            theta=self.output_layer(theta)
            
            # #stn
            # # x,asdf=self.stn(x)
            # theta = theta.view(-1, 2, 3)
            # grid = F.affine_grid(theta, x.size())
            # x = F.grid_sample(x, grid)
            #landmark
            # pdb.set_trace()
            #min max scale
            t_max=torch.max(theta,1)[0]#.repeat(1,49*2)
            t_max=torch.unsqueeze(t_max,dim=1).repeat(1,self.row_num*self.row_num*2)
            t_min=torch.min(theta,1)[0]#.repeat(1,49*2)
            t_min=torch.unsqueeze(t_min,dim=1).repeat(1,self.row_num*self.row_num*2)
            theta=(theta-t_min)/(t_max-t_min)*111

            # #sigmoid scale
            # theta=(self.sig(theta)*2-0.5)*(self.image_size-1)
            # theta=self.sigmoid(theta)*111*2-111*0.5
            # theta=theta.round()
            # theta=theta.type(torch.int32)
            theta=theta.view(-1,self.row_num*self.row_num,2)
            self.theta=theta#.detach()
            # pdb.set_trace()

            
            # pdb.set_trace()
            x=extract_patches_pytorch_gridsample(x,theta[:,:num_land],patch_shape=self.patch_shape,num_landm=num_land)
            # pdb.set_trace()
        
        # else:
        #     x=x#.transpose(1,2)
        # pdb.set_trace()
        if self.use_standcoord:
            # if self.use_standcoord:
            range_coor=torch.arange(0,np.sqrt(num_land))*8+4#[0,14]-->[4,12,20,108]
            coor_x,coor_y=torch.meshgrid(range_coor,range_coor)
            standard_coord=torch.stack((coor_x,coor_y),2).view(1,-1,2)
            # pdb.set_trace()
            theta=repeat(standard_coord, '() n d -> b n d', b = b).cuda()#[196,2-->b,196,2]
            b,c,fea=theta.shape
            if self.Random_prob:
                # pdb.set_trace()
                prob=torch.randn(theta.shape)*3#*12# 10 pixel 
                theta=theta+prob.cuda()
                # if not return_prob:
                
                # if ran_sample:
                #     extract_id=torch.randint(0,c,(b,36,1)).cuda()
                #     keep_num=36
                # else:
            if self.shuffle:
                extract_id=torch.randint(0,c,(b,c,1)).cuda()
                # keep_num=self.row_num*self.row_num
                extract_id=extract_id.repeat(1,1,2)
                #extract landmarks
                # for i in range(b):
                # extract_id=extract_id.view(592,25,1)
                out_theta=torch.gather(theta, 1, extract_id)
                theta= out_theta
            # stand_coord=self.standard_coord.repeat()
            # pdb.set_trace()
            # from PIL import Image
            x=extract_patches_pytorch_gridsample(x,theta[:,:num_land],patch_shape=self.patch_shape,num_landm=num_land)
            x=x.permute(0,1,3,2)
            # # x=torch.permute(x,(0,3,1,2))
            # x=x.permute(0,2,3,1)
            # x_gen=x_gen.permute(0,3,2,1)
            
            # # x_gen=torch.permute(x_gen,(0,3,1,2))
            # img = Image.fromarray(((x.cpu().detach().numpy()[0]*0.5+0.5)*255).astype(np.uint8))
            # img.save("faces.png")
            # img = Image.fromarray(((x_gen.cpu().detach().numpy()[0]*0.5+0.5)*255).astype(np.uint8))
            # # img = Image.fromarray(x_gen.cpu().detach().numpy()[0]*0.5+0.5)
            # img.save("faces_gen.png")
        if len(x.shape)==4:
            x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape
        # pdb.set_trace()
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        # x=self.drop_2d(x)
        x = self.dropout(x)
        x = self.transformer(x, mask)
        if save_token==True:
            tokens=x[:,1:]
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        emb = self.mlp_head(x)
        
        
        if save_token:
            return emb,tokens,self.theta
        if label is not None:
            if opt is not None:
                # pdb.set_trace()
                x = self.loss(emb, label,opt)
                return x, emb
            else:
                x = self.loss(emb, label)
                return x, self.theta
        # elif 

        else:
            if visualize==True:
                return emb,theta#self.theta
            else:
                return emb
            # return emb

class ViT_face_landmark_patch8_4simmin(nn.Module):
    def __init__(self, *, loss_type, GPU_ID, num_class, image_size, patch_size, dim, depth, heads, mlp_dim, pool = 'cls',num_patches=None, channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.,fp16=True):
        super().__init__()
        # assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        if num_patches==None:
            num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        assert num_patches > MIN_NUM_PATCHES, f'your number of patches ({num_patches}) is way too small for attention to be effective (at least 16). Try decreasing your patch size'
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        # # pdb.set_trace()
        self.patch_size = patch_size
        self.fp16=fp16
        self.num_patches=num_patches
        self.row_num=int(np.sqrt(num_patches)/2)#49
        self.row_num=int(np.sqrt(num_patches))#196
        # self.stn=MobileNetV3_backbone(mode='large')
        self.dim=dim
        # # # pdb.set_trace()
        # # # self.stn= ViT_face_stn_patch8(
        # # #                  loss_type = 'None',
        # # #                  GPU_ID = GPU_ID,
        # # #                  num_class = num_class,
        # # #                  image_size=112,
        # # #                  patch_size=8,#8
        # # #                  dim=96,#512
        # # #                  depth=12,#20
        # # #                  heads=3,#8
        # # #                  mlp_dim=1024,
        # # #                  dropout=0.1,
        # # #                  emb_dropout=0.1
        # # #              )
        # # #resnet
        # # self.stn=models.resnet50()
        # # self.stn.fc=nn.Sequential()
        # # # hybrid_dimension=50
        # # # drop_ratio=0.9

        # self.output_layer = nn.Sequential(
        #     nn.Dropout(p=0.5),    # refer to paper section 6
        #     nn.Linear(160, self.row_num*self.row_num*2),#2048
        # )
        
        # # self.output_layer = nn.Linear(96, int(self.row_num*self.row_num*2))#49*2,6        mobilenet 96 irse:128
        # # self.patch_shape=torch.tensor([2*patch_size,2*patch_size])#49
        # self.patch_shape=torch.tensor([patch_size,patch_size])#196
        self.theta=0

        self.drop_2d=torch.nn.Dropout2d(p=0.1)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            # nn.Dropout(p=0.1),
            # nn.Linear(dim,512),
            nn.LayerNorm(dim),#nn.Identity()
        )
        self.loss_type = loss_type
        self.GPU_ID = GPU_ID
        self.sigmoid = nn.Sigmoid()
        self.num_features=dim
        self.in_chans=channels


        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        self._trunc_normal_(self.mask_token, std=.02)
        # if self.loss_type == 'None':
        #     print("no loss for vit_face")
        # else:
        #     if self.loss_type == 'Softmax':
        #         self.loss = Softmax(in_features=dim, out_features=num_class, device_id=self.GPU_ID)
        #     elif self.loss_type == 'CosFace':
                
        #         # # # pdb.set_trace()
        #         # from vit_pytorch_my.partial_fc import PartialFCAdamW
                
        #         # self.loss = PartialFCAdamW(
        #         #     CosFace_arcimplement(), dim, num_class, 
        #         #     1.0, self.fp16)
        #         # # module_partial_fc.train().cuda()
                

        #         self.loss = CosFace(in_features=dim, out_features=num_class, device_id=self.GPU_ID,m=0.4)
        #     elif self.loss_type == 'ArcFace':
        #         self.loss = ArcFace(in_features=dim, out_features=num_class, device_id=self.GPU_ID)
        #     elif self.loss_type == 'SFace':
        #         self.loss = SFaceLoss(in_features=dim, out_features=num_class, device_id=self.GPU_ID)
    def _trunc_normal_(self, tensor, mean=0., std=1.):
        trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)
    def forward(self, x, mask_sim=None,label= None , mask = None,visualize=False,save_token=False,opt=None,keep_num=None):
        p = self.patch_size

        #get the image shape
        imgshape=x.shape[-2]
        if self.num_patches==144 and imgshape==112:
            keep_num=self.num_patches
        else:
            keep_num=(imgshape//p)**2
        # # x=x/255.0*2-1  #no mean
        # # img,fdsa=self.stn(img)
        # # pdb.set_trace()
        # # x_patch=x.clone()
        # # x=x.detach()
        # theta=self.stn(x)#.forward(x)            #with original stn
        # # pdb.set_trace()

        # theta = theta.mean(dim=(-2, -1))#average pooling   for cnn
        # theta=self.output_layer(theta)
        
        # # #stn
        # # # x,asdf=self.stn(x)
        # # theta = theta.view(-1, 2, 3)
        # # grid = F.affine_grid(theta, x.size())
        # # x = F.grid_sample(x, grid)
        # #landmark
        # # pdb.set_trace()
        # # #min max scale
        # # t_max=torch.max(theta,1)[0]#.repeat(1,49*2)
        # # t_max=torch.unsqueeze(t_max,dim=1).repeat(1,self.row_num*self.row_num*2)
        # # t_min=torch.min(theta,1)[0]#.repeat(1,49*2)
        # # t_min=torch.unsqueeze(t_min,dim=1).repeat(1,self.row_num*self.row_num*2)
        # # theta=(theta-t_min)/(t_max-t_min)*111

        # # #sigmoid scale
        # # theta=(self.sig(theta)*2-0.5)*(self.image_size-1)
        # theta=self.sigmoid(theta)*111*2-111*0.5
        # # theta=theta.round()
        # # theta=theta.type(torch.int32)
        # theta=theta.view(-1,self.row_num*self.row_num,2)
        # self.theta=theta#.detach()
        # # pdb.set_trace()

        # if keep_num is not None:
        #     num_land=keep_num
        # else:
        #     num_land=int(self.row_num*self.row_num)
        # # pdb.set_trace()
        # x=extract_patches_pytorch_gridsample(x,theta[:,:num_land],patch_shape=self.patch_shape,num_landm=num_land)
        # pdb.set_trace()
        land_x=x.detach()
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)
        x = self.patch_to_embedding(x)
        # pdb.set_trace()
        if mask_sim is not None:
            B, L, _ = x.shape
            mask_token = self.mask_token.expand(B, L, -1)
            w = mask_sim.flatten(1).unsqueeze(-1).type_as(mask_token)
            x = x * (1 - w) + mask_token * w




        b, n, _ = x.shape
        # pdb.set_trace()
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        # x=self.drop_2d(x)
        x = self.dropout(x)
        x = self.transformer(x, mask)
        # if save_token==True:
        #     tokens=x[:,1:]
        # x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        emb = self.mlp_head(x)[:,1:]#extract visual tokens for reconstruction
        
        
        if save_token:
            return emb,tokens,self.theta
        if label is not None:
            if opt is not None:
                # pdb.set_trace()
                x = self.loss(emb, label,opt)
                return x, emb
            else:
                x = self.loss(emb, label)
                return x, emb
        # elif 

        else:
            if visualize==True:
                return emb,theta#self.theta
            else:
                B, L, C = emb.shape
                H = W = int(L ** 0.5)
                emb = emb.permute(0, 2, 1).reshape(B, C, H, W)
                return emb,land_x
            # return emb

class ViT_face_landmark_patch8_4simmin_glo_loc(nn.Module):
    def __init__(self, *, loss_type, GPU_ID, num_class, image_size, patch_size, dim, depth, heads, mlp_dim, pool = 'cls',num_patches=None, channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.,fp16=True):
        super().__init__()
        # assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        if num_patches==None:
            num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        assert num_patches > MIN_NUM_PATCHES, f'your number of patches ({num_patches}) is way too small for attention to be effective (at least 16). Try decreasing your patch size'
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        # # pdb.set_trace()
        self.patch_size = patch_size
        self.fp16=fp16
        self.num_patches=num_patches
        self.row_num=int(np.sqrt(num_patches)/2)#49
        self.row_num=int(np.sqrt(num_patches))#196
        self.stn=MobileNetV3_backbone(mode='large')
        self.dim=dim
        # # # self.stn= ViT_face_stn_patch8(
        # # #                  loss_type = 'None',
        # # #                  GPU_ID = GPU_ID,
        # # #                  num_class = num_class,
        # # #                  image_size=112,
        # # #                  patch_size=8,#8
        # # #                  dim=96,#512
        # # #                  depth=12,#20
        # # #                  heads=3,#8
        # # #                  mlp_dim=1024,
        # # #                  dropout=0.1,
        # # #                  emb_dropout=0.1
        # # #              )
        # # #resnet
        # # self.stn=models.resnet50()
        # # self.stn.fc=nn.Sequential()
        # # # hybrid_dimension=50
        # # # drop_ratio=0.9

        self.output_layer = nn.Sequential(
            nn.Dropout(p=0.5),    # refer to paper section 6
            nn.Linear(160, self.row_num*self.row_num*2),#2048
        )
        self.global_token = nn.Sequential(
            nn.Dropout(p=0.5),    # refer to paper section 6
            nn.Linear(160, dim),# 
        )
        # self.output_layer = nn.Linear(96, int(self.row_num*self.row_num*2))#49*2,6        mobilenet 96 irse:128
        # self.patch_shape=torch.tensor([2*patch_size,2*patch_size])#49
        self.patch_shape=torch.tensor([patch_size,patch_size])#196
        self.theta=0

        self.drop_2d=torch.nn.Dropout2d(p=0.1)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            # nn.Dropout(p=0.1),
            # nn.Linear(dim,512),
            nn.LayerNorm(dim),#nn.Identity()
        )
        self.loss_type = loss_type
        self.GPU_ID = GPU_ID
        self.sigmoid = nn.Sigmoid()
        self.num_features=dim
        self.in_chans=channels


        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        self._trunc_normal_(self.mask_token, std=.02)
        # if self.loss_type == 'None':
        #     print("no loss for vit_face")
        # else:
        #     if self.loss_type == 'Softmax':
        #         self.loss = Softmax(in_features=dim, out_features=num_class, device_id=self.GPU_ID)
        #     elif self.loss_type == 'CosFace':
                
        #         # # # pdb.set_trace()
        #         # from vit_pytorch_my.partial_fc import PartialFCAdamW
                
        #         # self.loss = PartialFCAdamW(
        #         #     CosFace_arcimplement(), dim, num_class, 
        #         #     1.0, self.fp16)
        #         # # module_partial_fc.train().cuda()
                

        #         self.loss = CosFace(in_features=dim, out_features=num_class, device_id=self.GPU_ID,m=0.4)
        #     elif self.loss_type == 'ArcFace':
        #         self.loss = ArcFace(in_features=dim, out_features=num_class, device_id=self.GPU_ID)
        #     elif self.loss_type == 'SFace':
        #         self.loss = SFaceLoss(in_features=dim, out_features=num_class, device_id=self.GPU_ID)
    def _trunc_normal_(self, tensor, mean=0., std=1.):
        trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)
    
    def forward(self, x, mask_sim=None,label= None , mask = None,visualize=False,save_token=False,opt=None,keep_num=None,knowledge_dis=False):
        p = self.patch_size

        #get the image shape
        imgshape=x.shape[-2]
        if self.num_patches==144 and imgshape==112:
            keep_num=self.num_patches
        elif self.num_patches==196 and imgshape==112:
            keep_num=self.num_patches
        else:
            keep_num=(imgshape//p)**2
        # x=x/255.0*2-1  #no mean
        # img,fdsa=self.stn(img)
        # pdb.set_trace()
        # x_patch=x.clone()
        # x=x.detach()
        theta=self.stn(x)#.forward(x)            #with original stn
        # pdb.set_trace()

        theta0 = theta.mean(dim=(-2, -1))#average pooling   for cnn
        theta=self.output_layer(theta0)
        glo_token=self.global_token(theta0).view(-1,1,self.dim)
        
        # #stn
        # # x,asdf=self.stn(x)
        # theta = theta.view(-1, 2, 3)
        # grid = F.affine_grid(theta, x.size())
        # x = F.grid_sample(x, grid)
        #landmark
        # pdb.set_trace()
        #min max scale
        t_max=torch.max(theta,1)[0]#.repeat(1,49*2)
        t_max=torch.unsqueeze(t_max,dim=1).repeat(1,self.row_num*self.row_num*2)
        t_min=torch.min(theta,1)[0]#.repeat(1,49*2)
        t_min=torch.unsqueeze(t_min,dim=1).repeat(1,self.row_num*self.row_num*2)
        theta=(theta-t_min)/(t_max-t_min)*111

        # #sigmoid scale
        # theta=(self.sig(theta)*2-0.5)*(self.image_size-1)
        # theta=self.sigmoid(theta)*111*2-111*0.5
        # theta=theta.round()
        # theta=theta.type(torch.int32)
        theta=theta.view(-1,self.row_num*self.row_num,2)
        self.theta=theta#.detach()
        # pdb.set_trace()

        if keep_num is not None:
            num_land=keep_num
        else:
            num_land=int(self.row_num*self.row_num)
        # pdb.set_trace()
        x=extract_patches_pytorch_gridsample(x,theta[:,:num_land],patch_shape=self.patch_shape,num_landm=num_land)
        # pdb.set_trace()
        
        land_x=x.detach()
        x=x.detach()
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)
        x = self.patch_to_embedding(x)
        # pdb.set_trace()
        if mask_sim is not None:
            B, L, _ = x.shape
            mask_token = self.mask_token.expand(B, L, -1)
            w = mask_sim.flatten(1).unsqueeze(-1).type_as(mask_token)
            x = x * (1 - w) + mask_token * w




        b, n, _ = x.shape
        # pdb.set_trace()
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        # x = torch.cat((cls_tokens, x), dim=1)
        x = torch.cat((glo_token, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        # x=self.drop_2d(x)
        x = self.dropout(x)
        x = self.transformer(x, mask)
        # if save_token==True:
        #     tokens=x[:,1:]
        # x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        emb = self.mlp_head(x)[:,1:]#extract visual tokens for reconstruction
        glo_out=self.mlp_head(x)[:,0]
        # pdb.set_trace()
        if save_token:
            return emb,tokens,self.theta
        if label is not None:
            if opt is not None:
                # pdb.set_trace()
                x = self.loss(emb, label,opt)
                return x, emb
            else:
                x = self.loss(emb, label)
                return x, emb,
        # elif 

        else:
            if visualize==True:
                return emb,theta#self.theta
            else:
                
                B, L, C = emb.shape
                H = W = int(L ** 0.5)
                emb = emb.permute(0, 2, 1).reshape(B, C, H, W)
                if knowledge_dis:
                    return emb,land_x,glo_out,theta
                else:
                    return emb,land_x,glo_out
            # return emb
class mobile_dino(nn.Module):
    def __init__(self, embd_dim=128):
        super().__init__()
        self.stn=MobileNetV3_backbone(mode='large')
        self.output_layer = nn.Sequential(
            nn.Dropout(p=0.5),    # refer to paper section 6
            nn.Linear(160, embd_dim),#2048
        )
    def forward(self,x):
        theta=self.stn(x)#.forward(x)            #with original stn
        # pdb.set_trace()

        theta0 = theta.mean(dim=(-2, -1))#average pooling   for cnn
        theta=self.output_layer(theta0)
        return theta
class face_landmark_4simmin_glo_loc(nn.Module):
    def __init__(self, *, loss_type, GPU_ID, num_class, image_size, patch_size, dim, depth, heads, mlp_dim, pool = 'cls',num_patches=None, channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.,fp16=True):
        super().__init__()
        # assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        if num_patches==None:
            num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        assert num_patches > MIN_NUM_PATCHES, f'your number of patches ({num_patches}) is way too small for attention to be effective (at least 16). Try decreasing your patch size'
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        # # pdb.set_trace()
        self.patch_size = patch_size
        self.fp16=fp16
        self.num_patches=num_patches
        self.row_num=int(np.sqrt(num_patches)/2)#49
        self.row_num=int(np.sqrt(num_patches))#196
        self.stn=MobileNetV3_backbone(mode='large')
        self.dim=dim
        # # # self.stn= ViT_face_stn_patch8(
        # # #                  loss_type = 'None',
        # # #                  GPU_ID = GPU_ID,
        # # #                  num_class = num_class,
        # # #                  image_size=112,
        # # #                  patch_size=8,#8
        # # #                  dim=96,#512
        # # #                  depth=12,#20
        # # #                  heads=3,#8
        # # #                  mlp_dim=1024,
        # # #                  dropout=0.1,
        # # #                  emb_dropout=0.1
        # # #              )
        # # #resnet
        # # self.stn=models.resnet50()
        # # self.stn.fc=nn.Sequential()
        # # # hybrid_dimension=50
        # # # drop_ratio=0.9

        self.output_layer = nn.Sequential(
            nn.Dropout(p=0.5),    # refer to paper section 6
            nn.Linear(160, self.row_num*self.row_num*2),#2048
        )
        self.global_token = nn.Sequential(
            nn.Dropout(p=0.5),    # refer to paper section 6
            nn.Linear(160, dim),# 
        )
        # self.output_layer = nn.Linear(96, int(self.row_num*self.row_num*2))#49*2,6        mobilenet 96 irse:128
        # self.patch_shape=torch.tensor([2*patch_size,2*patch_size])#49
        self.patch_shape=torch.tensor([patch_size,patch_size])#196
        self.theta=0

        self.drop_2d=torch.nn.Dropout2d(p=0.1)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        # self.pool = pool
        # self.to_latent = nn.Identity()

        # self.mlp_head = nn.Sequential(
        #     # nn.Dropout(p=0.1),
        #     # nn.Linear(dim,512),
        #     nn.LayerNorm(dim),#nn.Identity()
        # )
        self.loss_type = loss_type
        self.GPU_ID = GPU_ID
        self.sigmoid = nn.Sigmoid()
        self.num_features=dim
        self.in_chans=channels


        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        self._trunc_normal_(self.mask_token, std=.02)
        # if self.loss_type == 'None':
        #     print("no loss for vit_face")
        # else:
        #     if self.loss_type == 'Softmax':
        #         self.loss = Softmax(in_features=dim, out_features=num_class, device_id=self.GPU_ID)
        #     elif self.loss_type == 'CosFace':
                
        #         # # # pdb.set_trace()
        #         # from vit_pytorch_my.partial_fc import PartialFCAdamW
                
        #         # self.loss = PartialFCAdamW(
        #         #     CosFace_arcimplement(), dim, num_class, 
        #         #     1.0, self.fp16)
        #         # # module_partial_fc.train().cuda()
                

        #         self.loss = CosFace(in_features=dim, out_features=num_class, device_id=self.GPU_ID,m=0.4)
        #     elif self.loss_type == 'ArcFace':
        #         self.loss = ArcFace(in_features=dim, out_features=num_class, device_id=self.GPU_ID)
        #     elif self.loss_type == 'SFace':
        #         self.loss = SFaceLoss(in_features=dim, out_features=num_class, device_id=self.GPU_ID)
    def _trunc_normal_(self, tensor, mean=0., std=1.):
        trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)
    def forward(self, x, x_Aug=None,keep_num=None,patch_shape=torch.tensor([10,10]),Random_prob=False,return_prob=False,ran_sample=False,random_coor=False,return_land=False):
        p = self.patch_size
        if not random_coor:
            #get the image shape
            imgshape=x.shape[-2]
            if self.num_patches==144 and imgshape==112:
                keep_num=self.num_patches
            elif self.num_patches==196 and imgshape==112:
                keep_num=self.num_patches
            else:
                keep_num=(imgshape//p)**2
            # x=x/255.0*2-1  #no mean
            # img,fdsa=self.stn(img)
            # pdb.set_trace()
            # x_patch=x.clone()
            # x=x.detach()
            theta=self.stn(x)#.forward(x)            #with original stn
            # pdb.set_trace()

            theta0 = theta.mean(dim=(-2, -1))#average pooling   for cnn
            theta=self.output_layer(theta0)
            glo_token=self.global_token(theta0).view(-1,1,self.dim)
            
            # #stn
            # # x,asdf=self.stn(x)
            # theta = theta.view(-1, 2, 3)
            # grid = F.affine_grid(theta, x.size())
            # x = F.grid_sample(x, grid)
            #landmark
            # pdb.set_trace()
            #min max scale
            t_max=torch.max(theta,1)[0]#.repeat(1,49*2)
            t_max=torch.unsqueeze(t_max,dim=1).repeat(1,self.row_num*self.row_num*2)
            t_min=torch.min(theta,1)[0]#.repeat(1,49*2)
            t_min=torch.unsqueeze(t_min,dim=1).repeat(1,self.row_num*self.row_num*2)
            theta=(theta-t_min)/(t_max-t_min)*111

            # #sigmoid scale
            # theta=(self.sig(theta)*2-0.5)*(self.image_size-1)
            # theta=self.sigmoid(theta)*111*2-111*0.5
            # theta=theta.round()
            # theta=theta.type(torch.int32)
            theta=theta.view(-1,self.row_num*self.row_num,2)
            if Random_prob:
                # pdb.set_trace()
                prob=torch.randn(theta.shape)*5#*12# 10 pixel 
                theta=theta+prob.cuda()
                if not return_prob:
                    b,c,fea=theta.shape
                    if ran_sample:
                        extract_id=torch.randint(0,c,(b,36,1)).cuda()
                        keep_num=36
                    else:
                        extract_id=torch.randint(0,c,(b,self.row_num*self.row_num,1)).cuda()
                        keep_num=self.row_num*self.row_num
                    extract_id=extract_id.repeat(1,1,2)
                    #extract landmarks
                    # for i in range(b):
                    # extract_id=extract_id.view(592,25,1)
                    out_theta=torch.gather(theta, 1, extract_id)
                    
                    # theta[0][extract_id[0,:,0]][:,1]==out_theta[0][:,1]
                    theta=out_theta
                # keep_num=25
            self.theta=theta#.detach()
            # pdb.set_trace()

            if keep_num is not None:
                num_land=keep_num
            else:
                num_land=theta.shape[-2]#int(self.row_num*self.row_num)
            # pdb.set_trace()
            # if return_land_only:
            #     return theta,0
            # else:
        else:
            # coor_max=theta.max()#get max value
            # pdb.set_trace()
            b=x.shape[0]
            if not ran_sample:
                num_land=self.row_num*self.row_num
            else:
                num_land=25
            new_theta=torch.rand(b,num_land,2)*111.0#.cuda()
            theta=new_theta.cuda()
        # pdb.set_trace()
        if return_land:
            return theta,x
        else:
            if x_Aug is None:
                x=extract_patches_pytorch_gridsample(x,theta[:,:num_land],patch_shape=self.patch_shape,num_landm=num_land)
            else:
                x=extract_patches_pytorch_gridsample(x_Aug,theta[:,:num_land],patch_shape=self.patch_shape,num_landm=num_land)
            return theta,x


class ViTs_face_overlap_4simmim(nn.Module):
    def __init__(self, *, loss_type, GPU_ID, num_class, image_size, patch_size, ac_patch_size,
                         pad, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        # assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * ac_patch_size ** 2
        assert num_patches > MIN_NUM_PATCHES, f'your number of patches ({num_patches}) is way too small for attention to be effective (at least 16). Try decreasing your patch size'
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        # pdb.set_trace()
        self.patch_size = patch_size
        self.soft_split = nn.Unfold(kernel_size=(ac_patch_size, ac_patch_size), stride=(self.patch_size, self.patch_size), padding=(pad, pad))


        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
        )
        self.loss_type = loss_type
        self.pred=None
        self.GPU_ID = GPU_ID
        self.fc=None
        self.num_features=dim
        self.in_chans=channels
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        self._trunc_normal_(self.mask_token, std=.02)
        # if self.loss_type == 'None':
        #     print("no loss for vit_face")
        # else:
        #     if self.loss_type == 'Softmax':
        #         self.loss = Softmax(in_features=dim, out_features=num_class, device_id=self.GPU_ID)
        #     elif self.loss_type == 'CosFace':
        #         self.loss = CosFace(in_features=dim, out_features=num_class, device_id=self.GPU_ID)
        #     elif self.loss_type == 'ArcFace':
        #         self.loss = ArcFace(in_features=dim, out_features=num_class, device_id=self.GPU_ID)
        #     elif self.loss_type == 'SFace':
        #         self.loss = SFaceLoss(in_features=dim, out_features=num_class, device_id=self.GPU_ID)
    def _trunc_normal_(self, tensor, mean=0., std=1.):
        trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)

    def forward(self, img, mask_sim=None,label= None , mask = None,visualize=False,save_token=False,opt=None,keep_num=None,knowledge_dis=False):
        p = self.patch_size
        # pdb.set_trace()
        land_x=img.detach()
        x = self.soft_split(img).transpose(1, 2)
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape


        if mask_sim is not None:
            B, L, _ = x.shape
            mask_token = self.mask_token.expand(B, L, -1)
            w = mask_sim.flatten(1).unsqueeze(-1).type_as(mask_token)
            x = x * (1 - w) + mask_token * w

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]


        x = self.dropout(x)
        x = self.transformer(x, mask)

        # x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        emb = self.mlp_head(x)[:,1:]#extract visual tokens for reconstruction
        glo_out=self.mlp_head(x)[:,0]
        # emb = self.mlp_head(x)
        # pdb.set_trace()


        if label is not None:
            x = self.loss(emb, label)
            return x, emb
        # else:
        else:
            if visualize==True:
                return emb,theta#self.theta
            else:
                B, L, C = emb.shape
                H = W = int(L ** 0.5)
                emb = emb.permute(0, 2, 1).reshape(B, C, H, W)
                return emb,land_x,glo_out

class ViTs_face_overlap(nn.Module):
    def __init__(self, *, loss_type, GPU_ID, num_class, image_size, patch_size, ac_patch_size,
                         pad, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        # assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * ac_patch_size ** 2
        assert num_patches > MIN_NUM_PATCHES, f'your number of patches ({num_patches}) is way too small for attention to be effective (at least 16). Try decreasing your patch size'
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        # pdb.set_trace()
        self.patch_size = patch_size
        self.soft_split = nn.Unfold(kernel_size=(ac_patch_size, ac_patch_size), stride=(self.patch_size, self.patch_size), padding=(pad, pad))


        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            # nn.LayerNorm(dim),
            nn.BatchNorm1d(dim)
        )
        
        self.loss_type = loss_type
        self.pred=None
        self.GPU_ID = GPU_ID
        self.fc=None
        # if self.loss_type == 'None':
        #     print("no loss for vit_face")
        # else:
        #     if self.loss_type == 'Softmax':
        #         self.loss = Softmax(in_features=dim, out_features=num_class, device_id=self.GPU_ID)
        #     elif self.loss_type == 'CosFace':
        #         self.loss = CosFace(in_features=dim, out_features=num_class, device_id=self.GPU_ID)
        #     elif self.loss_type == 'ArcFace':
        #         self.loss = ArcFace(in_features=dim, out_features=num_class, device_id=self.GPU_ID)
        #     elif self.loss_type == 'SFace':
        #         self.loss = SFaceLoss(in_features=dim, out_features=num_class, device_id=self.GPU_ID)
    def forward(self, x, return_before_head=False, patch_drop=0.,for_fea=False):
        # pdb.set_trace()
        if for_fea==True:
            return self.forward_features(x,patch_drop=patch_drop)
        if not isinstance(x, list):
            x = [x]
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]),
            return_counts=True,
        )[1], 0)
        start_idx = 0
        for end_idx in idx_crops:
            _h = self.forward_features(torch.cat(x[start_idx:end_idx]), patch_drop=patch_drop)
            _z = self.forward_head(_h)
            if start_idx == 0:
                h, z = _h, _z
            else:
                h, z = torch.cat((h, _h)), torch.cat((z, _z))
            patch_drop = 0.
            start_idx = end_idx

        if return_before_head:
            return h, z
        return z
    def forward_head(self, x):
        if self.pred is not None:
            return self.pred(x)
        return x
    def forward_features(self, img, label= None , mask = None,patch_drop=None):
        p = self.patch_size
        # pdb.set_trace()
        if len(img.shape)==4:
            x = self.soft_split(img).transpose(1, 2)
        else:
            x=img#.transpose(1,2)
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]

        if patch_drop > 0:
            patch_keep = 1. - patch_drop
            T_H = int(np.floor((x.shape[1]-1)*patch_keep))
            perm = 1 + torch.randperm(x.shape[1]-1)[:T_H]  # keep class token
            idx = torch.cat([torch.zeros(1, dtype=perm.dtype, device=perm.device), perm])
            x = x[:, idx, :]
        x = self.dropout(x)
        x = self.transformer(x, mask)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        emb = self.mlp_head(x)
        # pdb.set_trace()
        if self.fc is not None:
            x = self.fc(x)

        if label is not None:
            x = self.loss(emb, label)
            return x, emb
        else:
            return emb

def extract_patches_pytorch_gridsample(imgs, landmarks, patch_shape,num_landm=49):#numpy
    """ Extracts patches from an image.
    Args:
        imgs: a numpy array of dimensions [batch_size, width, height, channels]
        landmarks: a numpy array of dimensions [num_patches, 2]
        patch_shape: (width, height)
    Returns:
        a numpy array [num_patches, width, height, channels]
    """
    # pdb.set_trace()
    device=landmarks.device
    # imgs=imgs.to(device)
    # patch_shape = np.array(patch_shape)
    # patch_shape = np.array(patch_shape)
    # patch_half_shape = torch.require(torch.round(patch_shape / 2), dtype=int)
    img_shape=imgs.shape[2]
    # pdb.set_trace()
    list_patches = []
    patch_half_shape=patch_shape/2
    start = -patch_half_shape
    end = patch_half_shape
    # sampling_grid = torch.meshgrid[start[0]:end[0], start[1]:end[1]]
    sampling_grid = torch.meshgrid(torch.arange(start[0],end[0]),torch.arange(start[1],end[1]))#         start[0]:end[0], start[1]:end[1]]
    sampling_grid=torch.stack(sampling_grid,dim=0).to(device)#.cuda()
    # sampling_grid = sampling_grid.swapaxes(0, 2).swapaxes(0, 1)
    sampling_grid=torch.transpose(torch.transpose(sampling_grid,0,2),0,1)
    for i in range(num_landm):
        
        land=landmarks[:,i,:]

        patch_grid = (sampling_grid[None, :, :, :] + land[:, None, None, :])/(img_shape*0.5)-1
        sing_land_patch= F.grid_sample(imgs, patch_grid,align_corners=False)
        list_patches.append(sing_land_patch)
    # pdb.set_trace()
    list_patches=torch.stack(list_patches,dim=2)#.shape
    B, c, patches_num,w,h = list_patches.shape
    row=int(np.sqrt(patches_num))
    list_patches=list_patches.reshape(B,c,row,row,w,h)
    list_patches=list_patches.permute(0,1,2,4,3,5)
    list_patches=list_patches.reshape(B,c,w*int(np.sqrt(patches_num)),h*int(np.sqrt(patches_num)))
                      #.astype('int32')
    return list_patches#list_patches.cuda()
    # return list_patches   
