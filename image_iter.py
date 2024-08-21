#!/usr/bin/env python
# encoding: utf-8


import torchvision.transforms as transforms
import torch.utils.data as data
import numpy as np
# import cv2
import os
import torch

import mxnet as mx
from mxnet import ndarray as nd
from mxnet import io
from mxnet import recordio
import logging
import numbers
import random
logger = logging.getLogger()
from timm.data import create_transform
from timm.data.auto_augment import rand_augment_transform
from util import rand_aa_face
from ada_data_loader import Augmenter
# from RandAugment import RandAugment
from IPython import embed
import pdb
from PIL import Image
import torchvision.datasets as datasets
import torch.nn.functional as F
import json
import os
import cv2
import csv
from PIL import ImageFilter, ImageOps
import io
import pdb

import math

torch.pi = math.pi
class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img
        img=img.numpy()
        img=np.transpose(img,(1,2,0))
        img = Image.fromarray(img)
        img=img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )
        img=np.transpose(np.array(img),(2,0,1))
        return torch.tensor(img)

#scale 0.9-1
#ra-[0,pi/6]
def Affine_transform_s_a_t(imgs, landmarks_0=[0.45,0.55],scales_0=[0.9,1],angles_0=[0,torch.pi/6], patch_shape=[112,112],num_landm=1):#numpy
    """ Extracts patches from an image.
    landmarks:0-112
    scale:0-1
    angles:0-90
    """
    
    imgs=torch.unsqueeze(imgs,dim=0)
    #
    scales=torch.zeros(1,1,2)
    angles=torch.zeros(1,1,1)
    landmarks=torch.zeros(1,1,2)
    #generate random number
    scales[:,:,0]=random.uniform(scales_0[0], scales_0[1])
    scales[:,:,1]=random.uniform(scales_0[0], scales_0[1])
    angles[:,:,0]=random.uniform(angles_0[0], angles_0[1])
    # angles[:,:,1]=random.uniform(angles_0[0], angles_0[1])
    
    landmarks[:,:,0]=random.uniform(landmarks_0[0], landmarks_0[1])
    landmarks[:,:,1]=random.uniform(landmarks_0[0], landmarks_0[1])
    landmarks=landmarks*111.0
    # device=landmarks.device
    dtype_landmarks=landmarks.dtype
    imgs=imgs.type(dtype_landmarks)
    # landmarks=landmarks.type(dtype_img)
    # scales=scales.type(dtype_img)
    # imgs=imgs.to(device)
    # patch_shape = np.array(patch_shape)
    # patch_shape = np.array(patch_shape)
    # patch_half_shape = torch.require(torch.round(patch_shape / 2), dtype=int)
    img_shape=imgs.shape[2]
    
    list_patches = []
    out_grid=torch.zeros([imgs.shape[0],imgs.shape[1],patch_shape[0],patch_shape[1]])
    landmark_grid_size=out_grid.size()
    # landmark_grid_size[2]=patch_shape[0]
    # landmark_grid_size[3]=patch_shape[1]
    # pdb.set_trace()
    for i in range(num_landm):
        #scale
        scale=scales[:,0]
        # scale=1.5*scale+0.5
        transformed_scale=scale*patch_shape[0]/img_shape   #0-1,  --> 0.5-2
        # transformed_scale=(1.5*transformed_scale+0.5)/img_shape
        # translation
        land=landmarks[:,i,:]/(img_shape*0.5)-1    #[-1,1]
        land=land.reshape(-1,1,2,1)
        # transform_matrix=torch.zeros([batch_size,3,2])
        
        sin_angle=angles[:,i]#).clone()
        #

        # pdb.set_trace()
        transform_matrix = F.pad(land, (2,0,0,0)).squeeze(1)
        transform_matrix[:,0,0]=transformed_scale[:,0]*torch.cos(sin_angle)
        transform_matrix[:,1,1]=transformed_scale[:,0]*torch.cos(sin_angle)
        transform_matrix[:,0,1]=-transformed_scale[:,0]*torch.sin(sin_angle)
        transform_matrix[:,1,0]=transformed_scale[:,0]*torch.sin(sin_angle)
        # transform_matrix[:,]
        # patch_grid=F.affine_grid(transform_matrix, imgs.size())
        patch_grid=F.affine_grid(transform_matrix, landmark_grid_size,align_corners=False)

        

        # patch_grid = (sampling_grid[None, :, :, :] + land[:, None, None, :])/(img_shape*0.5)-1
        sing_land_patch= F.grid_sample(imgs, patch_grid,align_corners=False)
        break
        # list_patches.append(sing_land_patch)
    # # pdb.set_trace()
    # list_patches=torch.stack(list_patches,dim=2)#.shape
    # B, c, patches_num,w,h = list_patches.shape
    # row=int(np.sqrt(patches_num))
    # list_patches=list_patches.reshape(B,c,row,row,w,h)
    # list_patches=list_patches.permute(0,1,2,4,3,5)
    # list_patches=list_patches.reshape(B,c,w*int(np.sqrt(patches_num)),h*int(np.sqrt(patches_num)))
                      #.astype('int32')
    # pdb.set_trace()
    return torch.squeeze(sing_land_patch)#list_patches.cuda()
    # return list_patches   

class ran_down_upsample(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, qf_min=50,qf_max=112):
        self.prob = p
        self.qf_min = qf_min
        self.qf_max = qf_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img
        
        qf = random.randrange(self.qf_min, self.qf_max )
        img=F.interpolate(img,size=qf)
        img=F.interpolate(img,size=112)
        return img
def randomJPEGcompression(image):
    qf = random.randrange(10, 100)
    outputIoStream = io.BytesIO()
    
    
    image=image.numpy()
    image=np.transpose(image,(1,2,0))
    image = Image.fromarray(image)
    image.save(outputIoStream, "JPEG", quality=qf, optimice=True)
    outputIoStream.seek(0)
    out_img=np.transpose(np.array(Image.open(outputIoStream)),(2,0,1))
    return torch.tensor(out_img
        )

# def ran_down_upsample(image,img_size=112):
#     qf = random.randrange(50, img_size)
#     image=F.interpolate(image,size=qf)
#     image=F.interpolate(image,size=112)
#     # outputIoStream = io.BytesIO()
#     # image.save(outputIoStream, "JPEG", quality=qf, optimice=True)
#     # outputIoStream.seek(0)
#     return image
class FaceDataset(data.Dataset):
    def __init__(self, path_imgrec, rand_mirror=False,random_resizecrop=False,img_size=112,rand_au=False,gen_att=False,config_str='rand-m2-mstd0.5-inc1',partition=1):#rand-m2-n3-mstd0.5-inc1
        self.rand_mirror = rand_mirror
        self.random_resizecrop=random_resizecrop
        self.rand_au=rand_au
        # self.ada_aug=ada_aug
        if self.random_resizecrop:
            if self.rand_au==True:
                # self.aa_transform = create_transform(
                #         input_size=img_size,
                #         is_training=True,
                #         # color_jitter=args.color_jitter,
                #         auto_augment='rand-m2-mstd0.5-inc1',#args.aa,
                #         # interpolation=args.train_interpolation,
                #         # re_prob=args.reprob,
                #         # re_mode=args.remode,
                #         # re_count=args.recount,
                #     )
                # pdb.set_trace()
                self.aa_transform=rand_aa_face.rand_augment_transform(
                            config_str=config_str,#'rand-m2-mstd0.5-inc1', 
                            hparams={'translate_const': 117}#, 'img_mean': (124, 116, 104)}
                        )
            self.trans=transforms.Compose([
                    # RandAugment(2, 2),
                    # transforms.RandomResizedCrop(size=(112, 112),
                    #                                             scale=(0.2, 1.0),
                    #                                             ratio=(0.75, 1.3333333333333333)),
                    # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0)
                    transforms.RandomResizedCrop(img_size,scale=(0.9,1.0)),
                    transforms.ColorJitter(brightness=0.1,contrast=0.1,saturation=0.1,hue=0.1),
                    transforms.RandomErasing(scale=(0.02, 0.1))#erasing
                    # aa_transform
            ])
            # if self.ada_aug:
            #     self.augmenter = Augmenter(0.2, 0.2, 0.2)
            #
            # flip_and_color_jitter = transforms.Compose([
            #     transforms.RandomResizedCrop(img_size,scale=(0.9,1.0)),
            #     transforms.RandomHorizontalFlip(p=0.5),

            #     # transforms.RandomApply(
            #     #     [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
            #     #     p=0.8
            #     # ),
            #     transforms.RandomApply(
            #     [transforms.Lambda(randomJPEGcompression)],p=0.05),
            #     transforms.RandomApply(
            #     [
            #         transforms.Resize(112)
            #     ],p=0.05),
            #     GaussianBlur(p=0.05),
            #     ran_down_upsample(p=0.01),
            #     transforms.ColorJitter(brightness=0.15,contrast=0.3,saturation=0.1,hue=0.1),
            #     transforms.RandomGrayscale(p=0.01),
            # ])


            # self.trans=transforms.Compose([
            #         # RandAugment(2, 2),
            #         transforms.RandomResizedCrop(size=(112, 112),
            #                                                     scale=(0.2, 1.0),
            #                                                     ratio=(0.75, 1.3333333333333333)),
            #         transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0),
            #         # transforms.RandomErasing(scale=(0.02, 0.1))#erasing
            #         # aa_transform
            # ])
        self.gen_att=gen_att
        if self.gen_att==True:
            self.trans_att =  transforms.Compose([
                              transforms.Resize((178,218)),
                              transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])
            # self.trans.transforms.insert(0, RandAugment(2, 2))
            
            #transforms.RandomResizedCrop(img_size,scale=(0.9,1.0))
        assert path_imgrec
        if path_imgrec:
            logging.info('loading recordio %s...',
                         path_imgrec)
            path_imgidx = path_imgrec[0:-4] + ".idx"
            print(path_imgrec, path_imgidx)
            self.imgrec = recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
            s = self.imgrec.read_idx(0)
            header, _ = recordio.unpack(s)
            if header.flag > 0:
                print('header0 label', header.label)
                self.header0 = (int(header.label[0]), int(header.label[1]))
                # assert(header.flag==1)
                # self.imgidx = range(1, int(header.label[0]))
                self.imgidx = []
                self.id2range = {}
                self.seq_identity = range(int(header.label[0]), int(header.label[1]))
                for identity in self.seq_identity:
                    s = self.imgrec.read_idx(identity)
                    header, _ = recordio.unpack(s)
                    a, b = int(header.label[0]), int(header.label[1])
                    count = b - a
                    self.id2range[identity] = (a, b)
                    self.imgidx += range(a, b)
                print('id2range', len(self.id2range))
            else:
                self.imgidx = list(self.imgrec.keys)
            self.seq = self.imgidx
            self.path_imgrec=path_imgrec

            if partition:
                len_sammples=np.int64(len(self.seq)*partition)
                self.seq=self.seq[:len_sammples]
                # pdb.set_trace()

    def __getitem__(self, index):
        idx = self.seq[index]
        s = self.imgrec.read_idx(idx)
        header, s = recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        _data = mx.image.imdecode(s)
        if self.rand_mirror:
            _rd = random.randint(0, 1)
            if _rd == 1:
                _data = mx.ndarray.flip(data=_data, axis=1)

        _data = nd.transpose(_data, axes=(2, 0, 1))
        _data = _data.asnumpy()
        #webface 
        # pdb.set_trace()
        #ms1m  bgr
        # if 'webface' in self.path_imgrec:  
        #     _data=_data[::-1,:,:]
        # pdb.set_trace()
        if 'ms1m' not in self.path_imgrec:  
            _data=_data[::-1,:,:]
        # #flip att
        # _data=_data[::-1,:,:]
        #randaugment
        if self.rand_au==True:
            
            _data=np.transpose(_data,(1,2,0))
            _data = Image.fromarray(_data)
            # _data.save("./img_check/{}_test_ori_webface_rec.jpeg".format(index))
            _data=self.aa_transform(_data)
            _data=np.array(_data)
            # _data_pil=np.transpose(_data,(1,2,0))
            #visualize
            # _data_pil = Image.fromarray(_data)
            # _data_pil.save("./img_check/{}_test_trans.jpeg".format(index))
            # output original
            _data=np.transpose(_data,(2,0,1))
            #output trans
            # _data=np.transpose(_data_pil,(2,0,1))

        _data1=_data.copy()
        # pdb.set_trace()
        # #visualize
        # _data=np.transpose(_data,(1,2,0))
        # _data_pil = Image.fromarray(_data)
        # _data_pil.save("./img_check/{}_test_trans_web2.jpeg".format(index))
        img = torch.from_numpy(_data1)
        # pdb.set_trace()
        if self.random_resizecrop:
            # img=self.aa_transform(img)
            img=self.trans(img)
            # img=self.aa_transform(img)
        if self.gen_att==True:
            img=np.array(img)
            #after flip
            # _data=np.transpose(_data,(2,0,1))
            
            img=np.transpose(img,(1,2,0))
            img = Image.fromarray(img)
            img=self.trans_att(img)
        
        return img, label

    def __len__(self):
        return len(self.seq)
class FaceDataset_adaaug(data.Dataset):
    def __init__(self, path_imgrec, rand_mirror=False,random_resizecrop=False,img_size=112,rand_au=False,gen_att=False,config_str='rand-m2-mstd0.5-inc1',ada_aug=False,partition=1):#rand-m2-n3-mstd0.5-inc1
        self.rand_mirror = rand_mirror
        self.random_resizecrop=random_resizecrop
        self.rand_au=rand_au
        self.ada_aug=ada_aug
        if self.random_resizecrop:
            if self.rand_au==True:
                # self.aa_transform = create_transform(
                #         input_size=img_size,
                #         is_training=True,
                #         # color_jitter=args.color_jitter,
                #         auto_augment='rand-m2-mstd0.5-inc1',#args.aa,
                #         # interpolation=args.train_interpolation,
                #         # re_prob=args.reprob,
                #         # re_mode=args.remode,
                #         # re_count=args.recount,
                #     )
                # pdb.set_trace()
                self.aa_transform=rand_aa_face.rand_augment_transform(
                            config_str=config_str,#'rand-m2-mstd0.5-inc1', 
                            hparams={'translate_const': 117}#, 'img_mean': (124, 116, 104)}
                        )
            self.trans=transforms.Compose([
                    # RandAugment(2, 2),
                    # transforms.RandomResizedCrop(size=(112, 112),
                    #                                             scale=(0.2, 1.0),
                    #                                             ratio=(0.75, 1.3333333333333333)),
                    # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0)
                    # transforms.RandomResizedCrop(img_size,scale=(0.9,1.0)),
                    transforms.ColorJitter(brightness=0.5,contrast=0.5,saturation=0.5,hue=0),
                    # transforms.RandomErasing(scale=(0.02, 0.1))#erasing
                    # aa_transform
            ])
            if self.ada_aug:
                self.augmenter = Augmenter(0.2, 0.2, 0.2)
            #
            # flip_and_color_jitter = transforms.Compose([
            #     transforms.RandomResizedCrop(img_size,scale=(0.9,1.0)),
            #     transforms.RandomHorizontalFlip(p=0.5),

            #     # transforms.RandomApply(
            #     #     [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
            #     #     p=0.8
            #     # ),
            #     transforms.RandomApply(
            #     [transforms.Lambda(randomJPEGcompression)],p=0.05),
            #     transforms.RandomApply(
            #     [
            #         transforms.Resize(112)
            #     ],p=0.05),
            #     GaussianBlur(p=0.05),
            #     ran_down_upsample(p=0.01),
            #     transforms.ColorJitter(brightness=0.15,contrast=0.3,saturation=0.1,hue=0.1),
            #     transforms.RandomGrayscale(p=0.01),
            # ])


            # self.trans=transforms.Compose([
            #         # RandAugment(2, 2),
            #         transforms.RandomResizedCrop(size=(112, 112),
            #                                                     scale=(0.2, 1.0),
            #                                                     ratio=(0.75, 1.3333333333333333)),
            #         transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0),
            #         # transforms.RandomErasing(scale=(0.02, 0.1))#erasing
            #         # aa_transform
            # ])
        self.gen_att=gen_att
        if self.gen_att==True:
            self.trans_att =  transforms.Compose([
                              transforms.Resize((178,218)),
                              transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])
            # self.trans.transforms.insert(0, RandAugment(2, 2))
            
            #transforms.RandomResizedCrop(img_size,scale=(0.9,1.0))
        assert path_imgrec
        if path_imgrec:
            logging.info('loading recordio %s...',
                         path_imgrec)
            path_imgidx = path_imgrec[0:-4] + ".idx"
            print(path_imgrec, path_imgidx)
            self.imgrec = recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
            s = self.imgrec.read_idx(0)
            header, _ = recordio.unpack(s)
            if header.flag > 0:
                print('header0 label', header.label)
                self.header0 = (int(header.label[0]), int(header.label[1]))
                # assert(header.flag==1)
                # self.imgidx = range(1, int(header.label[0]))
                self.imgidx = []
                self.id2range = {}
                self.seq_identity = range(int(header.label[0]), int(header.label[1]))
                for identity in self.seq_identity:
                    s = self.imgrec.read_idx(identity)
                    header, _ = recordio.unpack(s)
                    a, b = int(header.label[0]), int(header.label[1])
                    count = b - a
                    self.id2range[identity] = (a, b)
                    self.imgidx += range(a, b)
                print('id2range', len(self.id2range))
            else:
                self.imgidx = list(self.imgrec.keys)
            self.seq = self.imgidx
            self.path_imgrec=path_imgrec

            if partition:
                len_sammples=np.int64(len(self.seq)*partition)
                self.seq=self.seq[:len_sammples]
                # pdb.set_trace()

    def __getitem__(self, index):
        idx = self.seq[index]
        s = self.imgrec.read_idx(idx)
        header, s = recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        _data = mx.image.imdecode(s)
        if self.rand_mirror:
            _rd = random.randint(0, 1)
            if _rd == 1:
                _data = mx.ndarray.flip(data=_data, axis=1)

        _data = nd.transpose(_data, axes=(2, 0, 1))
        _data = _data.asnumpy()
        #webface 
        # pdb.set_trace()
        #ms1m  bgr
        # if 'webface' in self.path_imgrec:  
        #     _data=_data[::-1,:,:]
        # pdb.set_trace()
        if 'ms1m' not in self.path_imgrec:  
            _data=_data[::-1,:,:]
        # #flip att
        # _data=_data[::-1,:,:]
        #randaugment
        if self.rand_au==True:
            
            _data=np.transpose(_data,(1,2,0))
            _data = Image.fromarray(_data)
            # _data.save("./img_check/{}_test_ori_webface_rec.jpeg".format(index))
            _data=self.aa_transform(_data)
            _data=np.array(_data)
            # _data_pil=np.transpose(_data,(1,2,0))
            #visualize
            # _data_pil = Image.fromarray(_data)
            # _data_pil.save("./img_check/{}_test_trans.jpeg".format(index))
            # output original
            _data=np.transpose(_data,(2,0,1))
            #output trans
            # _data=np.transpose(_data_pil,(2,0,1))

        _data1=_data.copy()
        # pdb.set_trace()
        # #visualize
        # _data=np.transpose(_data,(1,2,0))
        # _data_pil = Image.fromarray(_data)
        # _data_pil.save("./img_check/{}_test_trans_web2.jpeg".format(index))
        img = torch.from_numpy(_data1)
        # pdb.set_trace()
        if self.random_resizecrop:
            # img=self.aa_transform(img)
            img=np.array(img)
            img=np.transpose(img,(1,2,0))
            img = Image.fromarray(img)
            img=self.augmenter.augment(img)
            img=np.array(img)
            img=np.transpose(img,(2,0,1))
            img = torch.from_numpy(img)
            img=self.trans(img)
            # img=self.aa_transform(img)
        if self.gen_att==True:
            img=np.array(img)
            #after flip
            # _data=np.transpose(_data,(2,0,1))
            
            img=np.transpose(img,(1,2,0))
            img = Image.fromarray(img)
            img=self.trans_att(img)
        
        return img, label

    def __len__(self):
        return len(self.seq)

class FaceDataset_syn_aug(data.Dataset):
    def __init__(self, path_imgrec, rand_mirror=False,random_resizecrop=False,img_size=112,rand_au=False,gen_att=False,config_str='rand-m2-mstd0.5-inc1',partition=1):#rand-m2-n3-mstd0.5-inc1
        self.rand_mirror = rand_mirror
        self.random_resizecrop=random_resizecrop
        self.rand_au=rand_au
        if self.random_resizecrop:
            if self.rand_au==True:
                # self.aa_transform = create_transform(
                #         input_size=img_size,
                #         is_training=True,
                #         # color_jitter=args.color_jitter,
                #         auto_augment='rand-m2-mstd0.5-inc1',#args.aa,
                #         # interpolation=args.train_interpolation,
                #         # re_prob=args.reprob,
                #         # re_mode=args.remode,
                #         # re_count=args.recount,
                #     )
                # pdb.set_trace()
                self.aa_transform=rand_aa_face.rand_augment_transform(
                            config_str=config_str,#'rand-m2-mstd0.5-inc1', 
                            hparams={'translate_const': 117}#, 'img_mean': (124, 116, 104)}
                        )
            # self.trans=transforms.Compose([
            #         # RandAugment(2, 2),
                    
            #         transforms.RandomResizedCrop(img_size,scale=(0.9,1.0)),
            #         transforms.ColorJitter(brightness=0.1,contrast=0.1,saturation=0.1,hue=0.1),
            #         # transforms.RandomErasing(scale=(0.02, 0.1))#erasing
            #         # aa_transform
            # ])
            # #
            self.trans = transforms.Compose([
                transforms.RandomResizedCrop(img_size,scale=(0.9,1.0)),
                transforms.RandomHorizontalFlip(p=0.5),

                # transforms.RandomApply(
                #     [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                #     p=0.8
                # ),
                
                transforms.RandomApply(
                [transforms.Lambda(randomJPEGcompression)],p=0.05),
                # transforms.RandomApply(
                # [
                #     transforms.Resize(112)
                # ],p=),
                GaussianBlur(p=0.05),
                ran_down_upsample(p=0.01),
                transforms.ColorJitter(brightness=0.15,contrast=0.3,saturation=0.1,hue=0.1),
                transforms.RandomGrayscale(p=0.01),
            ])
            # self.trans = transforms.Compose([
            #     transforms.RandomResizedCrop(img_size,scale=(0.9,1.0)),
            #     transforms.RandomHorizontalFlip(p=0.5),

            #     # transforms.RandomApply(
            #     #     [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
            #     #     p=0.8
            #     # ),
            #     transforms.RandomApply(
            #     [transforms.Lambda(randomJPEGcompression)],p=1),
            #     transforms.RandomApply(
            #     [
            #         transforms.Resize(112)
            #     ],p=0.05),
            #     GaussianBlur(p=0.05),
            #     ran_down_upsample(p=0.01),
            #     transforms.ColorJitter(brightness=0.15,contrast=0.3,saturation=0.1,hue=0.1),
            #     transforms.RandomGrayscale(p=0.01),
            # ])

            # self.trans=transforms.Compose([
            #         # RandAugment(2, 2),
            #         transforms.RandomResizedCrop(size=(112, 112),
            #                                                     scale=(0.2, 1.0),
            #                                                     ratio=(0.75, 1.3333333333333333)),
            #         transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0),
            #         # transforms.RandomErasing(scale=(0.02, 0.1))#erasing
            #         # aa_transform
            # ])
        self.gen_att=gen_att
        if self.gen_att==True:
            self.trans_att =  transforms.Compose([
                              transforms.Resize((178,218)),
                              transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])
            # self.trans.transforms.insert(0, RandAugment(2, 2))
            
            #transforms.RandomResizedCrop(img_size,scale=(0.9,1.0))
        assert path_imgrec
        if path_imgrec:
            logging.info('loading recordio %s...',
                         path_imgrec)
            path_imgidx = path_imgrec[0:-4] + ".idx"
            print(path_imgrec, path_imgidx)
            self.imgrec = recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
            s = self.imgrec.read_idx(0)
            header, _ = recordio.unpack(s)
            if header.flag > 0:
                print('header0 label', header.label)
                self.header0 = (int(header.label[0]), int(header.label[1]))
                # assert(header.flag==1)
                # self.imgidx = range(1, int(header.label[0]))
                self.imgidx = []
                self.id2range = {}
                self.seq_identity = range(int(header.label[0]), int(header.label[1]))
                for identity in self.seq_identity:
                    s = self.imgrec.read_idx(identity)
                    header, _ = recordio.unpack(s)
                    a, b = int(header.label[0]), int(header.label[1])
                    count = b - a
                    self.id2range[identity] = (a, b)
                    self.imgidx += range(a, b)
                print('id2range', len(self.id2range))
            else:
                self.imgidx = list(self.imgrec.keys)
            self.seq = self.imgidx
            self.path_imgrec=path_imgrec

            if partition:
                len_sammples=np.int64(len(self.seq)*partition)
                self.seq=self.seq[:len_sammples]
                # pdb.set_trace()

    def __getitem__(self, index):
        idx = self.seq[index]
        s = self.imgrec.read_idx(idx)
        header, s = recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        _data = mx.image.imdecode(s)
        if self.rand_mirror:
            _rd = random.randint(0, 1)
            if _rd == 1:
                _data = mx.ndarray.flip(data=_data, axis=1)

        _data = nd.transpose(_data, axes=(2, 0, 1))
        _data = _data.asnumpy()
        #webface 
        # pdb.set_trace()
        #ms1m  bgr
        # if 'webface' in self.path_imgrec:  
        #     _data=_data[::-1,:,:]
        # pdb.set_trace()
        if 'ms1m' not in self.path_imgrec:  
            _data=_data[::-1,:,:]
        # #flip att
        # _data=_data[::-1,:,:]
        #randaugment
        if self.rand_au==True:
            
            _data=np.transpose(_data,(1,2,0))
            _data = Image.fromarray(_data)
            # _data.save("./img_check/{}_test_ori_webface_rec.jpeg".format(index))
            _data=self.aa_transform(_data)
            _data=np.array(_data)
            # _data_pil=np.transpose(_data,(1,2,0))
            #visualize
            # _data_pil = Image.fromarray(_data)
            # _data_pil.save("./img_check/{}_test_trans.jpeg".format(index))
            # output original
            _data=np.transpose(_data,(2,0,1))
            #output trans
            # _data=np.transpose(_data_pil,(2,0,1))

        _data1=_data.copy()
        # pdb.set_trace()
        # #visualize
        # _data=np.transpose(_data,(1,2,0))
        # _data_pil = Image.fromarray(_data)
        # _data_pil.save("./img_check/{}_test_trans_web2.jpeg".format(index))
        img = torch.from_numpy(_data1)
        
        if self.random_resizecrop:
            # img=self.aa_transform(img)
            img=self.trans(img).float()
            # pdb.set_trace()
            # do_it = random.random() <= 0.2#self.prob
            # if do_it:
            #     img=Affine_transform_s_a_t(img)
            # img=self.aa_transform(img)
        if self.gen_att==True:
            img=np.array(img)
            #after flip
            # _data=np.transpose(_data,(2,0,1))
            
            img=np.transpose(img,(1,2,0))
            img = Image.fromarray(img)
            img=self.trans_att(img)
        
        return img, label

    def __len__(self):
        return len(self.seq)



class FaceDataset_withatt(data.Dataset):
    def __init__(self, path_imgrec,att_csv_path='att_label_flip.csv', rand_mirror=False,random_resizecrop=False,img_size=112,rand_au=False,gen_att=False,mconfig_str='rand-m2-mstd0.5-inc1',partition=1):#rand-m2-n3-mstd0.5-inc1
        self.rand_mirror = rand_mirror
        self.random_resizecrop=random_resizecrop
        self.rand_au=rand_au
        if self.random_resizecrop:
            if self.rand_au==True:
                # self.aa_transform = create_transform(
                #         input_size=img_size,
                #         is_training=True,
                #         # color_jitter=args.color_jitter,
                #         auto_augment='rand-m2-mstd0.5-inc1',#args.aa,
                #         # interpolation=args.train_interpolation,
                #         # re_prob=args.reprob,
                #         # re_mode=args.remode,
                #         # re_count=args.recount,
                #     )
                # pdb.set_trace()
                self.aa_transform=rand_aa_face.rand_augment_transform(
                            config_str=config_str,#'rand-m2-mstd0.5-inc1', 
                            hparams={'translate_const': 117}#, 'img_mean': (124, 116, 104)}
                        )
            self.trans=transforms.Compose([
                    # RandAugment(2, 2),
                    
                    # transforms.RandomResizedCrop(img_size,scale=(0.9,1.0)),
                    transforms.ColorJitter(brightness=0.1,contrast=0.1,saturation=0.1,hue=0.1),
                    transforms.RandomErasing(scale=(0.02, 0.1))#erasing
                    # aa_transform
            ])
        self.gen_att=gen_att
        if self.gen_att==True:
            self.trans_att =  transforms.Compose([
                              transforms.Resize((178,218)),
                              transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])
            # self.trans.transforms.insert(0, RandAugment(2, 2))
            
            #transforms.RandomResizedCrop(img_size,scale=(0.9,1.0))
        assert path_imgrec
        if path_imgrec:
            logging.info('loading recordio %s...',
                         path_imgrec)
            path_imgidx = path_imgrec[0:-4] + ".idx"
            print(path_imgrec, path_imgidx)
            self.imgrec = recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
            s = self.imgrec.read_idx(0)
            header, _ = recordio.unpack(s)
            if header.flag > 0:
                print('header0 label', header.label)
                self.header0 = (int(header.label[0]), int(header.label[1]))
                # assert(header.flag==1)
                # self.imgidx = range(1, int(header.label[0]))
                self.imgidx = []
                self.id2range = {}
                self.seq_identity = range(int(header.label[0]), int(header.label[1]))
                for identity in self.seq_identity:
                    s = self.imgrec.read_idx(identity)
                    header, _ = recordio.unpack(s)
                    a, b = int(header.label[0]), int(header.label[1])
                    count = b - a
                    self.id2range[identity] = (a, b)
                    self.imgidx += range(a, b)
                print('id2range', len(self.id2range))
            else:
                self.imgidx = list(self.imgrec.keys)
            self.seq = self.imgidx
            self.path_imgrec=path_imgrec

            if partition:
                len_sammples=np.int64(len(self.seq)*partition)
                self.seq=self.seq[:len_sammples]
            # pdb.set_trace()
            with open(att_csv_path, newline='') as csvfile:
                # spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
                self.spamreader = list(csv.reader(csvfile, delimiter=' '))#, quotechar='|')

    def __getitem__(self, index):
        idx = self.seq[index]
        s = self.imgrec.read_idx(idx)
        header, s = recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        #load att
        # pdb.set_trace()
        # att_label=self.spamreader[index][0].split(',')#idx
        att_label=np.asarray(self.spamreader[index][0].split(',')).astype(float)
        _data = mx.image.imdecode(s)
        if self.rand_mirror:
            _rd = random.randint(0, 1)
            if _rd == 1:
                _data = mx.ndarray.flip(data=_data, axis=1)

        _data = nd.transpose(_data, axes=(2, 0, 1))
        _data = _data.asnumpy()
        #webface 
        # pdb.set_trace()
        #ms1m  bgr
        # if 'webface' in self.path_imgrec:  
        #     _data=_data[::-1,:,:]
        if 'ms1m' not in self.path_imgrec:  
            _data=_data[::-1,:,:]
        # #flip att
        # _data=_data[::-1,:,:]
        #randaugment
        if self.rand_au==True:
            
            _data=np.transpose(_data,(1,2,0))
            _data = Image.fromarray(_data)
            # _data.save("./img_check/{}_test_ori_webface_rec.jpeg".format(index))
            _data=self.aa_transform(_data)
            _data=np.array(_data)
            # _data_pil=np.transpose(_data,(1,2,0))
            #visualize
            # _data_pil = Image.fromarray(_data)
            # _data_pil.save("./img_check/{}_test_trans.jpeg".format(index))
            # output original
            _data=np.transpose(_data,(2,0,1))
            #output trans
            # _data=np.transpose(_data_pil,(2,0,1))

        _data1=_data.copy()
        img = torch.from_numpy(_data1)
        # pdb.set_trace()
        if self.random_resizecrop:
            # img=self.aa_transform(img)
            img=self.trans(img)
            # img=self.aa_transform(img)
        if self.gen_att==True:
            img=np.array(img)
            #after flip
            # _data=np.transpose(_data,(2,0,1))
            
            img=np.transpose(img,(1,2,0))
            img = Image.fromarray(img)
            img=self.trans_att(img)
        return img, label,att_label

    def __len__(self):
        return len(self.seq)



class FaceDataset_webface(datasets.ImageFolder):
    def __init__(self, path_imgrec,filepath='Webface_list.json', rand_mirror=False,random_resizecrop=False,loader=datasets.folder.default_loader,img_size=112,rand_au=False,config_str='rand-m2-mstd0.5-inc1'):
        transform=None,
        target_transform=None
        loader=datasets.folder.default_loader
        is_valid_file=None
        # pdb.set_trace()
        if os.path.exists(filepath):
            f=open(filepath,'r')
            out = json.load(f)
            self.samples=out
        else:
            super(FaceDataset_webface, self).__init__(path_imgrec,
                                                       transform=transform,
                                                       target_transform=target_transform,
                                                       loader=loader,
                                                       is_valid_file=is_valid_file)
            out=self.samples
            f=open(filepath,'w')
            # out = f(self, directory, *args, **kwargs)
            json.dump(out,f)
            # filepath.write_text(json.dumps(out))
        
        # self.samples=datasets.ImageFolder()
        self.rand_mirror = rand_mirror
        self.random_resizecrop=random_resizecrop
        self.rand_au=rand_au
        if self.random_resizecrop:
            if self.rand_au==True:
                # self.aa_transform = create_transform(
                #         input_size=img_size,
                #         is_training=True,
                #         # color_jitter=args.color_jitter,
                #         auto_augment='rand-m2-mstd0.5-inc1',#args.aa,
                #         # interpolation=args.train_interpolation,
                #         # re_prob=args.reprob,
                #         # re_mode=args.remode,
                #         # re_count=args.recount,
                #     )
                # pdb.set_trace()
                self.aa_transform=rand_aa_face.rand_augment_transform(
                            config_str=config_str,#'rand-m2-mstd0.5-inc1', 
                            hparams={'translate_const': 117}#, 'img_mean': (124, 116, 104)}
                        )
            self.trans=transforms.Compose([
                    # RandAugment(2, 2),
                    
                    transforms.RandomResizedCrop(img_size,scale=(0.9,1.0)),
                    transforms.ColorJitter(brightness=0.1,contrast=0.1,saturation=0.1,hue=0.1),
                    transforms.RandomErasing(scale=(0.02, 0.1))#erasing
                    # aa_transform
            ])
            # self.trans.transforms.insert(0, RandAugment(2, 2))
            
            #transforms.RandomResizedCrop(img_size,scale=(0.9,1.0))
        assert path_imgrec
        self.root = path_imgrec
        self.loader=loader
        # if path_imgrec:
        #     logging.info('loading recordio %s...',
        #                  path_imgrec)
        #     path_imgidx = path_imgrec[0:-4] + ".idx"
        #     print(path_imgrec, path_imgidx)
        #     self.imgrec = recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        #     s = self.imgrec.read_idx(0)
        #     header, _ = recordio.unpack(s)
        #     if header.flag > 0:
        #         print('header0 label', header.label)
        #         self.header0 = (int(header.label[0]), int(header.label[1]))
        #         # assert(header.flag==1)
        #         # self.imgidx = range(1, int(header.label[0]))
        #         self.imgidx = []
        #         self.id2range = {}
        #         self.seq_identity = range(int(header.label[0]), int(header.label[1]))
        #         for identity in self.seq_identity:
        #             s = self.imgrec.read_idx(identity)
        #             header, _ = recordio.unpack(s)
        #             a, b = int(header.label[0]), int(header.label[1])
        #             count = b - a
        #             self.id2range[identity] = (a, b)
        #             self.imgidx += range(a, b)
        #         print('id2range', len(self.id2range))
        #     else:
        #         self.imgidx = list(self.imgrec.keys)
        #     self.seq = self.imgidx

    def __getitem__original(self, index):
        idx = self.seq[index]
        s = self.imgrec.read_idx(idx)
        header, s = recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        _data = mx.image.imdecode(s)
        if self.rand_mirror:
            _rd = random.randint(0, 1)
            if _rd == 1:
                _data = mx.ndarray.flip(data=_data, axis=1)

        _data = nd.transpose(_data, axes=(2, 0, 1))
        _data = _data.asnumpy()
        #randaugment
        if self.rand_au==True:
            # pdb.set_trace()
            _data=np.transpose(_data,(1,2,0))
            _data = Image.fromarray(_data)
            # _data.save("./img_check/{}_test_ori.jpeg".format(index))
            _data=self.aa_transform(_data)
            _data=np.array(_data)
            # _data_pil=np.transpose(_data,(1,2,0))
            #visualize
            # _data_pil = Image.fromarray(_data)
            # _data_pil.save("./img_check/{}_test_trans.jpeg".format(index))
            # output original
            _data=np.transpose(_data,(2,0,1))
            #output trans
            # _data=np.transpose(_data_pil,(2,0,1))


        img = torch.from_numpy(_data)
        # pdb.set_trace()
        if self.random_resizecrop:
            # img=self.aa_transform(img)
            img=self.trans(img)
            # img=self.aa_transform(img)
        return img, label

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        # pdb.set_trace()
        path, target = self.samples[index]
        # sample = self.loader(path)
        sample=cv2.imread(path)
        sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
        # pdb.set_trace()
        # if 'WebFace' in self.root:
        #     # swap rgb to bgr since image is in rgb for webface
        #     sample = Image.fromarray(np.asarray(sample)[:,:,::-1])
        sample=np.asarray(sample)
        if self.rand_mirror:
            _rd = random.randint(0, 1)
            if _rd == 1:
                # _data = mx.ndarray.flip(data=_data, axis=1)
                
                sample=np.fliplr(sample)
        # sample, _ = self.augment(sample)
        # if self.transform is not None:
        #     sample = self.transform(sample)
        # if self.target_transform is not None:
        #     target = self.target_transform(target)
        sample=np.transpose(sample,(2,0,1))
        _data=sample
        if self.rand_au==True:
            # pdb.set_trace()
            _data=np.transpose(_data,(1,2,0))
            _data = Image.fromarray(_data)
            # _data.save("./img_check/{}_test_ori_webface_cv2.jpeg".format(index))
            _data=self.aa_transform(_data)
            _data=np.array(_data)
            # _data_pil=np.transpose(_data,(1,2,0))
            #visualize
            # _data_pil = Image.fromarray(_data)
            # _data_pil.save("./img_check/{}_test_trans.jpeg".format(index))
            # output original
            _data=np.transpose(_data,(2,0,1))
            #output trans
            # _data=np.transpose(_data_pil,(2,0,1))


        img = torch.from_numpy(_data)
        # pdb.set_trace()
        if self.random_resizecrop:
            # img=self.aa_transform(img)
            img=self.trans(img)
            # img=self.aa_transform(img)
        sample=img

        return sample, target

    def __len__(self):
        return len(self.samples)


class FaceDataset_contrastive(data.Dataset):
    def __init__(self, path_imgrec, rand_mirror=False,random_resizecrop=False,img_size=112,rand_au=False,config_str='rand-m2-mstd0.5-inc1'):
        self.rand_mirror = rand_mirror
        self.random_resizecrop=random_resizecrop
        self.rand_au=rand_au
        if self.random_resizecrop:
            if self.rand_au==True:
                # self.aa_transform = create_transform(
                #         input_size=img_size,
                #         is_training=True,
                #         # color_jitter=args.color_jitter,
                #         auto_augment='rand-m2-mstd0.5-inc1',#args.aa,
                #         # interpolation=args.train_interpolation,
                #         # re_prob=args.reprob,
                #         # re_mode=args.remode,
                #         # re_count=args.recount,
                #     )
                # pdb.set_trace()
                self.aa_transform=rand_aa_face.rand_augment_transform(
                            config_str=config_str,#'rand-m2-mstd0.5-inc1', 
                            hparams={'translate_const': 117}#, 'img_mean': (124, 116, 104)}
                        )
            self.trans=transforms.Compose([
                    # RandAugment(2, 2),
                    
                    transforms.RandomResizedCrop(img_size,scale=(0.9,1.0)),
                    transforms.ColorJitter(brightness=0.1,contrast=0.1,saturation=0.1,hue=0.1),
                    transforms.RandomErasing(scale=(0.02, 0.1))#erasing
                    # aa_transform
            ])
            # self.trans.transforms.insert(0, RandAugment(2, 2))
            
            #transforms.RandomResizedCrop(img_size,scale=(0.9,1.0))
        assert path_imgrec
        # pdb.set_trace()
        if path_imgrec:
            logging.info('loading recordio %s...',
                         path_imgrec)
            path_imgidx = path_imgrec[0:-4] + ".idx"
            print(path_imgrec, path_imgidx)
            self.imgrec = recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
            s = self.imgrec.read_idx(0)
            header, _ = recordio.unpack(s)
            if header.flag > 0:
                print('header0 label', header.label)
                self.header0 = (int(header.label[0]), int(header.label[1]))
                # assert(header.flag==1)
                # self.imgidx = range(1, int(header.label[0]))
                self.imgidx = []
                self.id2range = {}
                self.seq_identity = range(int(header.label[0]), int(header.label[1]))
                for identity in self.seq_identity:
                    s = self.imgrec.read_idx(identity)
                    header, _ = recordio.unpack(s)
                    a, b = int(header.label[0]), int(header.label[1])
                    count = b - a
                    self.id2range[identity] = (a, b)
                    self.imgidx += range(a, b)
                print('id2range', len(self.id2range))
            else:
                self.imgidx = list(self.imgrec.keys)
            self.seq = self.imgidx

    def __getitem__(self, index):
        # pdb.set_trace()
        idx = self.seq[index]
        s = self.imgrec.read_idx(idx)
        header, s = recordio.unpack(s)
        label = header.label



        if not isinstance(label, numbers.Number):
            label = label[0]
        _data = mx.image.imdecode(s)
        if self.rand_mirror:
            _rd = random.randint(0, 1)
            if _rd == 1:
                _data = mx.ndarray.flip(data=_data, axis=1)

        _data = nd.transpose(_data, axes=(2, 0, 1))
        _data = _data.asnumpy()
        #randaugment
        if self.rand_au==True:
            # pdb.set_trace()
            _data=np.transpose(_data,(1,2,0))
            _data = Image.fromarray(_data)
            # _data.save("./img_check/{}_test_ori.jpeg".format(index))
            _data=self.aa_transform(_data)
            _data=np.array(_data)
            # _data_pil=np.transpose(_data,(1,2,0))
            #visualize
            # _data_pil = Image.fromarray(_data)
            # _data_pil.save("./img_check/{}_test_trans.jpeg".format(index))
            # output original
            _data=np.transpose(_data,(2,0,1))
            #output trans
            # _data=np.transpose(_data_pil,(2,0,1))


        img = torch.from_numpy(_data)
       
        if self.random_resizecrop:
            # img=self.aa_transform(img)
            img=self.trans(img)
            # img=self.aa_transform(img)


        #sample pn
        # pdb.set_trace()
        current_range=self.id2range[self.seq_identity[np.int64(label)]]
        sample_pn=torch.randint(2,[1])
        if sample_pn==0: #negative
            ran_number=torch.randint(len(self.seq),[1])
            idx_pn = self.seq[ran_number]
            while idx_pn in range(current_range[0],current_range[1]):
                ran_number=torch.randint(len(self.seq),[1])
                idx_pn = self.seq[ran_number]
        else:
            ran_number=torch.randint(current_range[0],current_range[1],[1])
            idx_pn = self.seq[index]
        img_pn,label_pn=self.generated_pn_img(idx_pn)
        # pdb.set_trace()
        return img, label, img_pn, sample_pn,label_pn

    def generated_pn_img(self,idx_pn):
        idx=idx_pn
        s = self.imgrec.read_idx(idx)
        header, s = recordio.unpack(s)
        label = header.label



        if not isinstance(label, numbers.Number):
            label = label[0]
        _data = mx.image.imdecode(s)
        if self.rand_mirror:
            _rd = random.randint(0, 1)
            if _rd == 1:
                _data = mx.ndarray.flip(data=_data, axis=1)

        _data = nd.transpose(_data, axes=(2, 0, 1))
        _data = _data.asnumpy()
        #randaugment
        if self.rand_au==True:
            # pdb.set_trace()
            _data=np.transpose(_data,(1,2,0))
            _data = Image.fromarray(_data)
            # _data.save("./img_check/{}_test_ori.jpeg".format(index))
            _data=self.aa_transform(_data)
            _data=np.array(_data)
            # _data_pil=np.transpose(_data,(1,2,0))
            #visualize
            # _data_pil = Image.fromarray(_data)
            # _data_pil.save("./img_check/{}_test_trans.jpeg".format(index))
            # output original
            _data=np.transpose(_data,(2,0,1))
            #output trans
            # _data=np.transpose(_data_pil,(2,0,1))


        img = torch.from_numpy(_data)
        # pdb.set_trace()
        if self.random_resizecrop:
            # img=self.aa_transform(img)
            img=self.trans(img)
            # img=self.aa_transform(img)
        return img,label
    def __len__(self):
        return len(self.seq)




if __name__ == '__main__':
    root = '/raid/Data/faces_webface_112x112/train.rec'
    embed()
    dataset = FaceDataset(path_imgrec =root, rand_mirror = False)
    trainloader = data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2, drop_last=False)
    print(len(dataset))
    for data, label in trainloader:
        print(data.shape, label)