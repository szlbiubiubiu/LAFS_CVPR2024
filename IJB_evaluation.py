# coding: utf-8

import os
import pickle

import matplotlib
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import timeit
import sklearn
import argparse
from sklearn.metrics import roc_curve, auc

from menpo.visualize.viewmatplotlib import sample_colours_from_colourmap
from prettytable import PrettyTable
from pathlib import Path
import sys
import warnings
import platform
# import onnx
import math
# from util import utils as utils
from einops import rearrange, repeat
# if 'Alienware' in platform.node():
#     os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# else:
#     os.environ['CUDA_VISIBLE_DEVICES'] = '3'
sys.path.insert(0, "../")
# sys.path.append('/homes/zs003/projects/paper_face/paper_face/vit_pytorch_my')
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='do ijb test')
# general
# parser.add_argument('--model-prefix', default='/home/zhonglin/project/paper_face/paper_face/results/49_12vit_30epoch_lr5e4/Backbone_VIT_land_Epoch_29_Batch_454839_Time_2021-07-03-05-44_checkpoint.pth', help='path to load model.')
#/import/nobackup_mmv_ioannisp/zs003/face_rec/pretrain_net/backbone.pth
# parser.add_argument('--image-path', default='/home/zhonglin/mount_folder/dataset/face_rec/IJB/IJB_release/IJBC', type=str, help='')#
#/home/zhonglin/mount_folder/dataset/face_rec/IJB/IJB_release/IJBC
#/import/nobackup_mmv_ioannisp/zs003/face_rec/IJB_release/IJBC
parser.add_argument('--result-dir', default='ms1mv3_arcface_r50', type=str, help='')
parser.add_argument('--batch-size', default=360, type=int, help='')#480
parser.add_argument('--network', default='iresnet50', type=str, help='')
parser.add_argument('--job', default='ms1mv3_arcface_r50', type=str, help='job name')
parser.add_argument('--target', default='IJBB', type=str, help='target, set to IJBC or IJBB')
args = parser.parse_args()

target = args.target
# model_path = args.model_prefix
# image_path = args.image_path
do_onnx=False

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
output_path='/home/zhonglin/project/paper_face/paper_face/onnx_check'
model_path='/data/scratch/acw569/checkpoint/sp_check/webface_webland_34epoch_mixup0501_aug01/Backbone_VIT_land_8_Epoch_34_Batch_359865_Time_2024-09-15-10-19_checkpoint.pth'
image_path='/data/scratch/acw569/face/IJB/IJB_release/IJBB'




result_dir = args.result_dir
gpu_id = None
use_norm_score = True  # if Ture, TestMode(N1)
use_detector_score = True  # if Ture, TestMode(D1)
use_flip_test = True  # if Ture, TestMode(F1)
job = args.job
batch_size = args.batch_size

import cv2
import numpy as np
import torch
from skimage import transform as trans
# import backbones
import pdb
from PIL import Image
# from vit_modify import ViT_stn_land
class Embedding(object):
    def __init__(self, prefix, data_shape, batch_size=1):
        image_size = (112, 112)
        self.image_size = image_size
        # pdb.set_trace()
        weight = torch.load(prefix,map_location=lambda storage, loc: storage.cuda(0))#vit+my resnet
        # weight = torch.load(prefix, map_location='cpu')['state_dict']  #adaface
        # pdb.set_trace()
        # resnet = eval("backbones.{}".format(args.network))(False).cuda()
        # resnet = eval("backbones.{}".format(args.network))(False).cuda()
        # resnet=ViT_stn_land(
        #             image_size = 112,
        #             patch_size = 16,
        #             num_classes = 512,
        #             dim = 512,
        #             depth = 12,
        #             heads = 8,
        #             mlp_dim = 2048,
        #             dropout=0.1,
        #             emb_dropout=0.1
        #         ).cuda()#to(conf.device)# 
        # pdb.set_trace()
        # # from vit_pytorch_my.vits_face import ViT_face_landmark
        # from vit_pytorch_my.vit_face import ViT_face,ViT_face_landmark,ViT_face_landmark_patch8,ViT_face_landmark_halfpatchsize, ViT_face_landmark_astoken,ViT_face_landmark_halfpatchsize_globaltoken,ViT_face_landmark_globaltoken,ViT_face_landmark_patch8_global
        # from vit_pytorch_my.vit_face import ViT_face_landmark_patch8_overlap,ViT_face_landmark_halfpatchsize_scale_affine,ViT_face_landmark_scale_affine,ViT_face_landmark_patch8_scale_affine,ViT_face_landmark_largepatch_inner,ViT_face_landmark_scale_affine_global,ViT_face_landmark_halfpatchsize_scale_affine_global
        # from vit_pytorch_my.vit_face import ViT_face_landmark_patch8_scale_affine_global,ViT_face_landmark_halfpatchsize_testnewloss,ViT_face_landmark_patch8_global_sysgraph,ViT_face_landmark_patch8_global_graph_lessparam_more,ViTs_face_overlap
        # from vit_pytorch_my.vit_face import ViT_face_landmark_patch8_att
        # # from vit_pytorch_my.cross_vit import cross_VIT,cross_VIT_landmark
        from functools import partial
        NUM_CLASS=205990#360232,205990,93431,  56000,ir-se-50 12000/10000
        from face_pre_pro.ViT_face import ViT_face_landmark_patch8
        resnet= ViT_face_landmark_patch8(
                         loss_type = 'CosFace',
                         GPU_ID = '0',
                         num_class = NUM_CLASS,#205990   93431# 56000
                         image_size=112,
                         patch_size=8,#8 14
                         dim=768,#512 ,768
                         depth=12,#20,12
                         heads=11,
                         mlp_dim=2048,
                         dropout=0.1,
                         emb_dropout=0.1,
                         with_land=True
                     ).cuda()
        
        # ####VIT load
        # # resnet.load_state_dict(weight,strict=True)
        # # model = torch.nn.DataParallel(torch.nn.DataParallel(resnet))
        model = torch.nn.DataParallel(resnet)
        model.load_state_dict(weight,strict=True)
        
        ##adaface load
        
        # print('statedict keys:',statedict.keys())
        # print('backbone keys:',backbone.keys())
        # backbone.load_state_dict({k.replace("module.", ''): v for k, v in statedict.items() if 'module.' in k},strict=True)
        # resnet.load_state_dict({k.replace("module.", ''): v for k, v in weight.items() if 'module.' in k},strict=True)

        # resnet.load_state_dict({k.replace("model.", ''): v for k, v in weight.items() if 'model.' in k},strict=False)
        # model = torch.nn.DataParallel(resnet)
        self.model = model
        self.model.eval()
        # pdb.set_trace()
        if do_onnx:
            convert_onnx(model,path_module=model_path,output_path=output_path,simplify=False)
            sys.exit()
        src = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)
        src[:, 0] += 8.0
        self.src = src
        self.batch_size = batch_size
        self.data_shape = data_shape


        self.pre_land=False   
        self.keep_land=False
        # pdb.set_trace()
        if self.pre_land==True:
            from vit_pytorch_my.vit_face import face_landmark_4simmin_glo_loc
            
            self.landmarkcnn=face_landmark_4simmin_glo_loc(loss_type = 'CosFace',
                                GPU_ID = None,
                                num_class = 30000,
                                num_patches=196,
                                image_size=112,
                                patch_size=8,#8
                                dim=512,#512
                                depth=12,#20
                                heads=11,#8
                                mlp_dim=2560,
                                dropout=0.1,
                                emb_dropout=0.1)
            self.landmarkcnn=self.landmarkcnn.cuda()
            #/import/nobackup_mmv_ioannisp/zs003/checkpoints/face_rec/ssl_results/simmin_vit_land_real/simmim_pretrain/simmim_pretrain__vit_face_100epo/ckpt_epoch_99.pth
            
            # load_part_checkpoint_landmark_fromsimmim(path='/root/face/check/ckpt_epoch_99.pth',model=landmarkcnn,pretrain_name=['stn','output'])
            # load_part_checkpoint_landmark_fromsimmim(path='/import/nobackup_mmv_ioannisp/zs003/checkpoints/face_rec/ssl_results/simmin_vit_land_real_gnn2trans/simmim_pretrain/simmim_pretrain__vit_face_100epo/ckpt_epoch_99.pth',model=landmarkcnn,pretrain_name=['stn','output'])
            # load_part_checkpoint_landmark(path='/import/nobackup_mmv_ioannisp/zs003/checkpoints/face_rec/results/VGG_landmark_new144/Backbone_VIT_land_8_Epoch_34_Batch_148113_Time_2022-09-05-20-37_checkpoint.pth',model=self.landmarkcnn,pretrain_name=['stn','output'])    
            #webface
            # load_part_checkpoint_landmark(path='/import/nobackup_mmv_ioannisp/zs003/checkpoints/face_rec/results/webface_196landmark/Backbone_VIT_land_8_Epoch_34_Batch_327225_Time_2022-05-05-10-34_checkpoint.pth',model=self.landmarkcnn,pretrain_name=['stn','output'])    
            #hpc
            # load_part_checkpoint_landmark(path='/data/scratch/acw569/precheckpoint/webface_196land_sp/Backbone_VIT_land_8_Epoch_34_Batch_327225_Time_2022-05-05-10-34_checkpoint.pth',model=self.landmarkcnn,pretrain_name=['stn','output'])
            #waixingren
            # load_part_checkpoint_landmark(path='/home/zhonglin/mount_folder/dataset/checkpoints/web_196_land/Backbone_VIT_land_8_Epoch_34_Batch_327225_Time_2022-05-05-10-34_checkpoint.pth',model=self.landmarkcnn,pretrain_name=['stn','output'])    
            
            
            
            #ms1m
            # load_part_checkpoint_landmark(path='/import/nobackup_mmv_ioannisp/zs003/checkpoints/face_rec/results/4gpu_landmark_augall_again_stnsmalldecay/Backbone_VIT_land_8_Epoch_34_Batch_523881_Time_2021-07-31-11-07_checkpoint.pth',model=self.landmarkcnn,pretrain_name=['stn','output'])
            # load_part_checkpoint_landmark(path='/import/nobackup_mmv_ioannisp/zs003/checkpoints/face_rec/results/webface_196landmark/Backbone_VIT_land_8_Epoch_34_Batch_327225_Time_2022-05-05-10-34_checkpoint.pth',model=self.landmarkcnn,pretrain_name=['stn','output'])
            self.landmarkcnn.eval()
            # if knowledge_dis:
            transf_cit = torch.nn.MSELoss()

        

    def get(self, rimg, landmark,index=0):

        assert landmark.shape[0] == 68 or landmark.shape[0] == 5
        assert landmark.shape[1] == 2
        if landmark.shape[0] == 68:
            landmark5 = np.zeros((5, 2), dtype=np.float32)
            landmark5[0] = (landmark[36] + landmark[39]) / 2
            landmark5[1] = (landmark[42] + landmark[45]) / 2
            landmark5[2] = landmark[30]
            landmark5[3] = landmark[48]
            landmark5[4] = landmark[54]
        else:
            landmark5 = landmark
        tform = trans.SimilarityTransform()
        tform.estimate(landmark5, self.src)
        M = tform.params[0:2, :]
        img = cv2.warpAffine(rimg,
                             M, (self.image_size[1], self.image_size[0]),
                             borderValue=0.0)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   #resnet rgb,,mine bgr
        # pdb.set_trace()
        # index=0
        _data_pil = Image.fromarray(img)
        # _data_pil.save("./img_check/{}_test_trans_ijb.jpeg".format(index))
        img_flip = np.fliplr(img)# keep this one
        _data_flip_pil = Image.fromarray(img_flip)
        # _data_flip_pil.save("./img_check/{}_test_trans_ijb_flip.jpeg".format(index))
        img = np.transpose(img, (2, 0, 1))  # 3*112*112, RGB
        img_flip = np.transpose(img_flip, (2, 0, 1))
        input_blob = np.zeros((2, 3, self.image_size[1], self.image_size[0]), dtype=np.uint8)
        input_blob[0] = img
        input_blob[1] = img_flip
        return input_blob

    @torch.no_grad()
    def forward_db(self, batch_data):
        imgs = torch.Tensor(batch_data).cuda()
        imgs.div_(255).sub_(0.5)#.div_(0.5)   #my 0.5, arcface 1   #retina data, 255
        # pdb.set_trace()
        # if self.pre_land==True:
        #     imgs=torch.Tensor(imgs).cuda()
        #     land_label,img_reconstructed=self.landmarkcnn(imgs.float())#div 255/2
        #     # land_label,img_reconstructed=landmarkcnn(images[0])
        #     #reconstructed image to embedding
        #     if not self.keep_land:
        #         imgs = rearrange(img_reconstructed, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = landmarkcnn.patch_size, p2 = landmarkcnn.patch_size)
        #         # batch_data=batch_data.cpu().numpy()#np.array(batch_data)
        feat = self.model(imgs)
        feat = feat.reshape([self.batch_size, 2 * feat.shape[1]])
        return feat.cpu().numpy()

    @torch.no_grad()
    def forward_db_visual(self, batch_data):
        # pdb.set_trace()
        imgs = torch.Tensor(batch_data).cuda()
        imgs.div_(255).sub_(0.5)#.div_(0.5)   #my 0.5, arcface 1   #retina data, 255
        # feat = self.model(imgs)
        if self.pre_land==True:
            # imgs=torch.Tensor(imgs).cuda()
            land_label,img_reconstructed=self.landmarkcnn(imgs.float())#div 255/2
            # land_label,img_reconstructed=landmarkcnn(images[0])
            #reconstructed image to embedding
            if not self.keep_land:
                imgs = rearrange(img_reconstructed, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = self.landmarkcnn.patch_size, p2 = self.landmarkcnn.patch_size)
        feat,theta = self.model(imgs,visualize=True)
        feat = feat.reshape([self.batch_size, 2 * feat.shape[1]])
        return feat.cpu().numpy(),theta#.cpu().numpy()
    

# 将一个list尽量均分成n份，限制len(list)==n，份数大于原list内元素个数则分配空list[]
def divideIntoNstrand(listTemp, n):
    twoList = [[] for i in range(n)]
    for i, e in enumerate(listTemp):
        twoList[i % n].append(e)
    return twoList


def read_template_media_list(path):
    # ijb_meta = np.loadtxt(path, dtype=str)
    ijb_meta = pd.read_csv(path, sep=' ', header=None).values
    templates = ijb_meta[:, 1].astype(np.int)
    medias = ijb_meta[:, 2].astype(np.int)
    return templates, medias


# In[ ]:


def read_template_pair_list(path):
    # pairs = np.loadtxt(path, dtype=str)
    pairs = pd.read_csv(path, sep=' ', header=None).values
    # print(pairs.shape)
    # print(pairs[:, 0].astype(np.int))
    t1 = pairs[:, 0].astype(np.int)
    t2 = pairs[:, 1].astype(np.int)
    label = pairs[:, 2].astype(np.int)
    return t1, t2, label


# In[ ]:


def read_image_feature(path):
    with open(path, 'rb') as fid:
        img_feats = pickle.load(fid)
    return img_feats
def calculate_overlap_near(theta,patch_size):
    theta=theta.cpu().numpy()
    # pdb.set_trace()
    out_mean=[]
    half_patch_size=np.int16(patch_size*0.5)
    for sin_sample in theta:
        overlap_map=np.zeros([len(sin_sample)])
        for i in range(len(sin_sample)):
            sin_theta=sin_sample[i]
            X=sin_theta

        # X = [0,0] # Your cooridinate
            x1 = X[0]
            y1= X[1]

            array = sin_sample#[[1,1],[0,1],[1,0],[-2,2]] # Sample array of data
            smallestDistance = 9999 # Make it big so it can be replaced immediately, essentially just a placeholder

            # for point in array:
            for j in range(len(array)):
                point=array[j]
                x2 = point[0]
                y2 = point[1]
                separation = math.hypot(x2 - x1, y2 - y1) #Distance equation in easy format
                if separation < smallestDistance and separation!=0:  # Could make this <= instead of < if you want to replace any ties for closest point
                    smallestDistance = separation
                    closestPoint = point
                    smallest_index=j
            # pdb.set_trace()
            #calculate the overlap
            sin_theta=np.int64(np.around(sin_theta))
            sin_map=np.zeros([112,112])
            x_min_sin=max(0,sin_theta[0]-half_patch_size)
            x_max_sin=min(111,sin_theta[0]+half_patch_size)
            y_min_sin=max(0,sin_theta[1]-half_patch_size)
            y_max_sin=min(111,sin_theta[1]+half_patch_size)

            sin_map[x_min_sin:x_max_sin,y_min_sin:y_max_sin]=1


            #
            right_theta=sin_sample[smallest_index]
            right_theta=np.int64(np.around(right_theta))
            right_map=np.zeros([112,112])
            x_min_rig=max(0,right_theta[0]-half_patch_size)
            x_max_rig=min(111,right_theta[0]+half_patch_size)
            y_min_rig=max(0,right_theta[1]-half_patch_size)
            y_max_rig=min(111,right_theta[1]+half_patch_size)

            right_map[x_min_rig:x_max_rig,y_min_rig:y_max_rig]=1
            # right_map[right_theta[0]-half_patch_size:right_theta[0]+half_patch_size,right_theta[1]-half_patch_size:right_theta[1]+half_patch_size]=1
            out_map=sin_map+right_map
            out_index=np.where(out_map==2)
            overlap_map[i]=len(out_index[0])/(patch_size*patch_size)
        # pdb.set_trace()
        one_mean=np.mean(overlap_map)
        out_mean+=[one_mean]
    # pdb.set_trace()
    return out_mean
        # for i in range(len(sin_sample)):
        #     sin_theta=sin_sample[i]
        #     search_nodes=np.delete(sin_sample.copy(),i,axis=0)
        #     closest_node(sin_theta,sin_sample)



    # for sin_sample in theta:
    #     half_patch_size=np.int16(patch_size*0.5)
    #     overlap_map=np.zeros([len(sin_sample),len(sin_sample)])
    #     for i in range(len(sin_sample)):
    #         sin_theta=sin_sample[i].cpu().numpy()
    #         sin_theta=np.int64(np.around(sin_theta))
    #         sin_map=np.zeros([112,112])
    #         x_min_sin=max(0,sin_theta[0]-half_patch_size)
    #         x_max_sin=min(111,sin_theta[0]+half_patch_size)
    #         y_min_sin=max(0,sin_theta[1]-half_patch_size)
    #         y_max_sin=min(111,sin_theta[1]+half_patch_size)

    #         sin_map[x_min_sin:x_max_sin,y_min_sin:y_max_sin]=1
    #         # sin_map[sin_theta[0]-half_patch_size:sin_theta[0]+half_patch_size,sin_theta[1]-half_patch_size:sin_theta[1]+half_patch_size]=1
    #         for j in range(len(sin_sample)):
    #             if i==j:
    #                 continue
    #             right_theta=sin_sample[j].cpu().numpy()
    #             right_theta=np.int64(np.around(right_theta))
    #             right_map=np.zeros([112,112])
    #             x_min_rig=max(0,right_theta[0]-half_patch_size)
    #             x_max_rig=min(111,right_theta[0]+half_patch_size)
    #             y_min_rig=max(0,right_theta[1]-half_patch_size)
    #             y_max_rig=min(111,right_theta[1]+half_patch_size)

    #             right_map[x_min_rig:x_max_rig,y_min_rig:y_max_rig]=1
    #             # right_map[right_theta[0]-half_patch_size:right_theta[0]+half_patch_size,right_theta[1]-half_patch_size:right_theta[1]+half_patch_size]=1
    #             out_map=sin_map+right_map
    #             out_index=np.where(out_map==2)
    #             overlap_map[i,j]=len(out_index[0])/(len(sin_sample)*len(sin_sample))
    #     pdb.set_trace()
    #     one_mean=np.mean(overlap_map)
    #     out_mean+=[one_mean]
    


# In[ ]:

def get_image_feature(img_path, files_list, model_path, epoch, gpu_id, save_samples=True,overlap=False,logpath='ijb'):
    batch_size = args.batch_size
    data_shape = (3, 112, 112)

    files = files_list
    print('files:', len(files))
    rare_size = len(files) % batch_size
    faceness_scores = []
    batch = 0
    img_feats = np.empty((len(files), 768*2), dtype=np.float32)#768*3,512*2

    batch_data = np.empty((2 * batch_size, 3, 112, 112))
    embedding = Embedding(model_path, data_shape, batch_size)
    over_lap_all=[]
    for img_index, each_line in enumerate(files[:len(files) - rare_size]):
        name_lmk_score = each_line.strip().split(' ')
        img_name = os.path.join(img_path, name_lmk_score[0])
        img = cv2.imread(img_name)
        lmk = np.array([float(x) for x in name_lmk_score[1:-1]],
                       dtype=np.float32)
        lmk = lmk.reshape((5, 2))
        input_blob = embedding.get(img, lmk,index=img_index)

        batch_data[2 * (img_index - batch * batch_size)][:] = input_blob[0]
        batch_data[2 * (img_index - batch * batch_size) + 1][:] = input_blob[1]
        
        if (img_index + 1) % batch_size == 0:
            print('batch', batch)
            # pdb.set_trace()
            if not save_samples:
                img_feats[batch * batch_size:batch * batch_size +
                                         batch_size][:] = embedding.forward_db(batch_data)
            else:
                
            # if (visualize==True) and (overlap==True):
                img_feats[batch * batch_size:batch * batch_size +
                                         batch_size][:],theta = embedding.forward_db_visual(batch_data)
                # emb,theta = backbone(batch.to(device),visualize=True)#.cpu()
                # over_lap=calculate_overlap(theta,patch_size=16)
                if overlap==True:
                    over_lap=calculate_overlap_near(theta,patch_size=28)
                    # pdb.set_trace()
                    over_lap_all+=over_lap
                # embeddings[idx:idx + batch_size] = emb.cpu()
                # pdb.set_trace()
                # utils.save_patch(batch_data,batch_data,theta,patch_size=embedding.model.patch_size,save_folder=logpath,iter1=batch,epoch=0,step=0)
            batch += 1
        faceness_scores.append(name_lmk_score[-1])

    batch_data = np.empty((2 * rare_size, 3, 112, 112))
    embedding = Embedding(model_path, data_shape, rare_size)
    for img_index, each_line in enumerate(files[len(files) - rare_size:]):
        name_lmk_score = each_line.strip().split(' ')
        img_name = os.path.join(img_path, name_lmk_score[0])
        img = cv2.imread(img_name)
        lmk = np.array([float(x) for x in name_lmk_score[1:-1]],
                       dtype=np.float32)
        lmk = lmk.reshape((5, 2))
        input_blob = embedding.get(img, lmk)
        batch_data[2 * img_index][:] = input_blob[0]
        batch_data[2 * img_index + 1][:] = input_blob[1]
        if (img_index + 1) % rare_size == 0:
            print('batch', batch)
            if not save_samples:
                img_feats[len(files) -
                        rare_size:][:] = embedding.forward_db(batch_data)
            else:
            # if (visualize==True) and (overlap==True):
                img_feats[batch * batch_size:batch * batch_size +
                                         batch_size][:],theta = embedding.forward_db_visual(batch_data)
                # emb,theta = backbone(batch.to(device),visualize=True)#.cpu()
                # over_lap=calculate_overlap(theta,patch_size=16)
                if overlap==True:
                    over_lap=calculate_overlap_near(theta,patch_size=28)
                    # pdb.set_trace()
                    over_lap_all+=over_lap
                # embeddings[idx:idx + batch_size] = emb.cpu()
                # save_patch(batch_data,batch,theta,patch_size=embedding.backbone.patch_size,save_folder=logpath,iter1=batch_count,epoch=epoch,step=step)
            batch += 1
        faceness_scores.append(name_lmk_score[-1])
    faceness_scores = np.array(faceness_scores).astype(np.float32)
    # img_feats = np.ones( (len(files), 1024), dtype=np.float32) * 0.01
    # faceness_scores = np.ones( (len(files), ), dtype=np.float32 )
    over_lap_mean=np.mean(over_lap_all)
    over_lap_var=np.var(over_lap_all)
    print ('mean:'+str(over_lap_mean))
    print ('var:'+str(over_lap_var))
    return img_feats, faceness_scores


# In[ ]:


def image2template_feature(img_feats=None, templates=None, medias=None):
    # ==========================================================
    # 1. face image feature l2 normalization. img_feats:[number_image x feats_dim]
    # 2. compute media feature.
    # 3. compute template feature.
    # ==========================================================
    unique_templates = np.unique(templates)
    template_feats = np.zeros((len(unique_templates), img_feats.shape[1]))

    for count_template, uqt in enumerate(unique_templates):

        (ind_t,) = np.where(templates == uqt)
        face_norm_feats = img_feats[ind_t]
        face_medias = medias[ind_t]
        unique_medias, unique_media_counts = np.unique(face_medias,
                                                       return_counts=True)
        media_norm_feats = []
        for u, ct in zip(unique_medias, unique_media_counts):
            (ind_m,) = np.where(face_medias == u)
            if ct == 1:
                media_norm_feats += [face_norm_feats[ind_m]]
            else:  # image features from the same video will be aggregated into one feature
                media_norm_feats += [
                    np.mean(face_norm_feats[ind_m], axis=0, keepdims=True)
                ]
        media_norm_feats = np.array(media_norm_feats)
        # media_norm_feats = media_norm_feats / np.sqrt(np.sum(media_norm_feats ** 2, -1, keepdims=True))
        template_feats[count_template] = np.sum(media_norm_feats, axis=0)
        if count_template % 2000 == 0:
            print('Finish Calculating {} template features.'.format(
                count_template))
    # template_norm_feats = template_feats / np.sqrt(np.sum(template_feats ** 2, -1, keepdims=True))
    template_norm_feats = sklearn.preprocessing.normalize(template_feats)
    # print(template_norm_feats.shape)
    return template_norm_feats, unique_templates


# In[ ]:


def verification(template_norm_feats=None,
                 unique_templates=None,
                 p1=None,
                 p2=None):
    # ==========================================================
    #         Compute set-to-set Similarity Score.
    # ==========================================================
    template2id = np.zeros((max(unique_templates) + 1, 1), dtype=int)
    for count_template, uqt in enumerate(unique_templates):
        template2id[uqt] = count_template

    score = np.zeros((len(p1),))  # save cosine distance between pairs

    total_pairs = np.array(range(len(p1)))
    batchsize = 100000  # small batchsize instead of all pairs in one batch due to the memory limiation
    sublists = [
        total_pairs[i:i + batchsize] for i in range(0, len(p1), batchsize)
    ]
    total_sublists = len(sublists)
    for c, s in enumerate(sublists):
        feat1 = template_norm_feats[template2id[p1[s]]]
        feat2 = template_norm_feats[template2id[p2[s]]]
        similarity_score = np.sum(feat1 * feat2, -1)
        score[s] = similarity_score.flatten()
        if c % 10 == 0:
            print('Finish {}/{} pairs.'.format(c, total_sublists))
    return score


# In[ ]:
def verification2(template_norm_feats=None,
                  unique_templates=None,
                  p1=None,
                  p2=None):
    template2id = np.zeros((max(unique_templates) + 1, 1), dtype=int)
    for count_template, uqt in enumerate(unique_templates):
        template2id[uqt] = count_template
    score = np.zeros((len(p1),))  # save cosine distance between pairs
    total_pairs = np.array(range(len(p1)))
    batchsize = 100000  # small batchsize instead of all pairs in one batch due to the memory limiation
    sublists = [
        total_pairs[i:i + batchsize] for i in range(0, len(p1), batchsize)
    ]
    total_sublists = len(sublists)
    for c, s in enumerate(sublists):
        feat1 = template_norm_feats[template2id[p1[s]]]
        feat2 = template_norm_feats[template2id[p2[s]]]
        similarity_score = np.sum(feat1 * feat2, -1)
        score[s] = similarity_score.flatten()
        if c % 10 == 0:
            print('Finish {}/{} pairs.'.format(c, total_sublists))
    return score

def convert_onnx(net, path_module, output_path,output_name='model_196', opset=12, simplify=False):
    output=os.path.join(output_path, "%s.onnx" % output_name)
    assert isinstance(net, torch.nn.Module)
    img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.int32)
    img = img.astype(np.float)
    img = (img / 255. - 0.5) / 0.5  # torch style norm
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    net=net.to('cpu')
    # weight = torch.load(path_module)
    # net.load_state_dict(weight)
    # net.eval()
    # torch.onnx.export(net, img, output, keep_initializers_as_inputs=False, verbose=False, opset_version=opset)
    pdb.set_trace()
    torch.onnx.export(net.module, img, output, keep_initializers_as_inputs=False, verbose=False, opset_version=opset)
    model = onnx.load(output)
    graph = model.graph
    graph.input[0].type.tensor_type.shape.dim[0].dim_param = 'None'
    if simplify:
        from onnxsim import simplify
        model, check = simplify(model)
        assert check, "Simplified ONNX model could not be validated"
    onnx.save(model, output)

def read_score(path):
    with open(path, 'rb') as fid:
        img_feats = pickle.load(fid)
    return img_feats

def load_part_checkpoint_landmark(path,model,pretrain_name=['stn','output']):
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
    for name, param in model.named_parameters():
        # if not param.requires_grad:
        if pretrain_name[0] in name or pretrain_name[1] in name:
            # pdb.set_trace()
            param.requires_grad = False


# # Step1: Load Meta Data

# In[ ]:

assert target == 'IJBC' or target == 'IJBB'

# =============================================================
# load image and template relationships for template feature embedding
# tid --> template id,  mid --> media id
# format:
#           image_name tid mid
# =============================================================
start = timeit.default_timer()
templates, medias = read_template_media_list(
    os.path.join('%s/meta' % image_path,
                 '%s_face_tid_mid.txt' % target.lower()))
stop = timeit.default_timer()
print('Time: %.2f s. ' % (stop - start))

# In[ ]:

# =============================================================
# load template pairs for template-to-template verification
# tid : template id,  label : 1/0
# format:
#           tid_1 tid_2 label
# =============================================================
start = timeit.default_timer()
p1, p2, label = read_template_pair_list(
    os.path.join('%s/meta' % image_path,
                 '%s_template_pair_label.txt' % target.lower()))
stop = timeit.default_timer()
print('Time: %.2f s. ' % (stop - start))

# # Step 2: Get Image Features

# In[ ]:

# =============================================================
# load image features
# format:
#           img_feats: [image_num x feats_dim] (227630, 512)
# =============================================================
start = timeit.default_timer()
img_path = '%s/loose_crop' % image_path
img_list_path = '%s/meta/%s_name_5pts_score.txt' % (image_path, target.lower())
img_list = open(img_list_path)
files = img_list.readlines()
# files_list = divideIntoNstrand(files, rank_size)
files_list = files

# img_feats
# for i in range(rank_size):
img_feats, faceness_scores = get_image_feature(img_path, files_list,
                                               model_path, 0, gpu_id)
stop = timeit.default_timer()
print('Time: %.2f s. ' % (stop - start))
print('Feature Shape: ({} , {}) .'.format(img_feats.shape[0],
                                          img_feats.shape[1]))

# # Step3: Get Template Features

# In[ ]:

# =============================================================
# compute template features from image features.
# =============================================================
start = timeit.default_timer()
# ==========================================================
# Norm feature before aggregation into template feature?
# Feature norm from embedding network and faceness score are able to decrease weights for noise samples (not face).
# ==========================================================
# 1. FaceScore （Feature Norm）
# 2. FaceScore （Detector）

if use_flip_test:
    # concat --- F1
    # img_input_feats = img_feats
    # add --- F2
    img_input_feats = img_feats[:, 0:img_feats.shape[1] //
                                     2] + img_feats[:, img_feats.shape[1] // 2:]
else:
    img_input_feats = img_feats[:, 0:img_feats.shape[1] // 2]

if use_norm_score:
    img_input_feats = img_input_feats
else:
    # normalise features to remove norm information
    img_input_feats = img_input_feats / np.sqrt(
        np.sum(img_input_feats ** 2, -1, keepdims=True))

if use_detector_score:
    print(img_input_feats.shape, faceness_scores.shape)
    img_input_feats = img_input_feats * faceness_scores[:, np.newaxis]
else:
    img_input_feats = img_input_feats

template_norm_feats, unique_templates = image2template_feature(
    img_input_feats, templates, medias)
stop = timeit.default_timer()
print('Time: %.2f s. ' % (stop - start))

# # Step 4: Get Template Similarity Scores

# In[ ]:

# =============================================================
# compute verification scores between template pairs.
# =============================================================
start = timeit.default_timer()
score = verification(template_norm_feats, unique_templates, p1, p2)
stop = timeit.default_timer()
print('Time: %.2f s. ' % (stop - start))

# In[ ]:
save_path = os.path.join(result_dir, args.job)
# save_path = result_dir + '/%s_result' % target

if not os.path.exists(save_path):
    os.makedirs(save_path)

score_save_file = os.path.join(save_path, "%s.npy" % target.lower())
np.save(score_save_file, score)

# # Step 5: Get ROC Curves and TPR@FPR Table

# In[ ]:

files = [score_save_file]
methods = []
scores = []
for file in files:
    methods.append(Path(file).stem)
    scores.append(np.load(file))

methods = np.array(methods)
scores = dict(zip(methods, scores))
colours = dict(
    zip(methods, sample_colours_from_colourmap(methods.shape[0], 'Set2')))
x_labels = [10 ** -6, 10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1]
tpr_fpr_table = PrettyTable(['Methods'] + [str(x) for x in x_labels])
fig = plt.figure()
for method in methods:
    fpr, tpr, _ = roc_curve(label, scores[method])
    roc_auc = auc(fpr, tpr)
    fpr = np.flipud(fpr)
    tpr = np.flipud(tpr)  # select largest tpr at same fpr
    plt.plot(fpr,
             tpr,
             color=colours[method],
             lw=1,
             label=('[%s (AUC = %0.4f %%)]' %
                    (method.split('-')[-1], roc_auc * 100)))
    tpr_fpr_row = []
    tpr_fpr_row.append("%s-%s" % (method, target))
    for fpr_iter in np.arange(len(x_labels)):
        _, min_index = min(
            list(zip(abs(fpr - x_labels[fpr_iter]), range(len(fpr)))))
        tpr_fpr_row.append('%.2f' % (tpr[min_index] * 100))
    tpr_fpr_table.add_row(tpr_fpr_row)
plt.xlim([10 ** -6, 0.1])
plt.ylim([0.3, 1.0])
plt.grid(linestyle='--', linewidth=1)
plt.xticks(x_labels)
plt.yticks(np.linspace(0.3, 1.0, 8, endpoint=True))
plt.xscale('log')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC on IJB')
plt.legend(loc="lower right")
fig.savefig(os.path.join(save_path, '%s.pdf' % target.lower()))
print(tpr_fpr_table)
print(model_path)