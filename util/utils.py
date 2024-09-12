import torch
import torchvision.transforms as transforms
import torch.nn.functional as F

from .verification import evaluate,evaluate_token,evaluate_two

from datetime import datetime
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
from PIL import Image
import mxnet as mx
import io
import os, pickle, sklearn
import time
from IPython import embed
import pdb
import matplotlib.patches as patches
import torch.nn as nn
import matplotlib.pyplot as plt
import math
from einops import rearrange, repeat
def get_time():
    return (str(datetime.now())[:-10]).replace(' ', '-').replace(':', '-')


def load_bin(path, image_size=[112,112]):
    print('loading bin')
    bins, issame_list = pickle.load(open(path, 'rb'), encoding='bytes')
    data_list = []
    for flip in [0,1]:
        data = torch.zeros((len(issame_list)*2, 3, image_size[0], image_size[1]))
        data_list.append(data)
    for i in range(len(issame_list)*2):
        _bin = bins[i]
        img = mx.image.imdecode(_bin)
        if img.shape[1]!=image_size[0]:
            img = mx.image.resize_short(img, image_size[0])
        img = mx.nd.transpose(img, axes=(2, 0, 1))
        for flip in [0,1]:
            if flip==1:
                img = mx.ndarray.flip(data=img, axis=2)
            data_list[flip][i][:] = torch.tensor(img.asnumpy())
        # if i%1000==0:
        #     print('loading bin', i)
    print(data_list[0].shape)
    return data_list, issame_list


def get_val_pair(path, name):
    ver_path = os.path.join(path,name + ".bin")
    print(ver_path)
    assert os.path.exists(ver_path)
    data_set, issame = load_bin(ver_path)
    print('ver', name)
    return data_set, issame


def get_val_data(data_path, targets):
    assert len(targets) > 0
    vers = []
    for t in targets:
        data_set, issame = get_val_pair(data_path, t)
        vers.append([t, data_set, issame])
    return vers


def separate_irse_bn_paras(modules):
    if not isinstance(modules, list):
        modules = [*modules.modules()]
    paras_only_bn = []
    paras_wo_bn = []
    for layer in modules:
        if 'model' in str(layer.__class__):
            continue
        if 'container' in str(layer.__class__):
            continue
        else:
            if 'batchnorm' in str(layer.__class__):
                paras_only_bn.extend([*layer.parameters()])
            else:
                paras_wo_bn.extend([*layer.parameters()])

    return paras_only_bn, paras_wo_bn


def separate_resnet_bn_paras(modules):
    all_parameters = modules.parameters()
    paras_only_bn = []

    for pname, p in modules.named_parameters():
        if pname.find('bn') >= 0:
            paras_only_bn.append(p)
            
    paras_only_bn_id = list(map(id, paras_only_bn))
    paras_wo_bn = list(filter(lambda p: id(p) not in paras_only_bn_id, all_parameters))
    
    return paras_only_bn, paras_wo_bn


def separate_mobilefacenet_bn_paras(modules):
    if not isinstance(modules, list):
        modules = [*modules.modules()]
    paras_only_bn = []
    paras_wo_bn = []
    for layer in modules:
        if 'mobilefacenet' in str(layer.__class__) or 'container' in str(layer.__class__):
            continue
        if 'batchnorm' in str(layer.__class__):
            paras_only_bn.extend([*layer.parameters()])
        else:
            paras_wo_bn.extend([*layer.parameters()])

    return paras_only_bn, paras_wo_bn


def gen_plot(fpr, tpr):
    """Create a pyplot plot and save to buffer."""
    plt.figure()
    plt.xlabel("FPR", fontsize = 14)
    plt.ylabel("TPR", fontsize = 14)
    plt.title("ROC Curve", fontsize = 14)
    plot = plt.plot(fpr, tpr, linewidth = 2)
    buf = io.BytesIO()
    plt.savefig(buf, format = 'jpeg')
    buf.seek(0)
    plt.close()

    return buf

def test_forward(device, backbone, data_set):
    backbone = backbone.to(device)
    backbone.eval() # switch to evaluation mode
    #embed()
    #last_time1 = time.time()
    forward_time = 0
    carray = data_set[0]
        #print("carray:",carray.shape)
    idx = 0
    with torch.no_grad():
            while idx < 2000:
                batch = carray[idx:idx + 1]
                batch_device = batch.to(device)
                last_time = time.time()
                backbone(batch_device)
                forward_time += time.time() - last_time
                #if idx % 1000 ==0:
                #    print(idx, forward_time)
                idx += 1
    print("forward_time", 2000, forward_time, 2000/forward_time)
    return forward_time

def calculate_overlap(theta,patch_size):
    pdb.set_trace()
    out_mean=[]
    for sin_sample in theta:
        half_patch_size=np.int16(patch_size*0.5)
        overlap_map=np.zeros([len(sin_sample),len(sin_sample)])
        for i in range(len(sin_sample)):
            sin_theta=sin_sample[i].cpu().numpy()
            sin_theta=np.int64(np.around(sin_theta))
            sin_map=np.zeros([112,112])
            x_min_sin=max(0,sin_theta[0]-half_patch_size)
            x_max_sin=min(111,sin_theta[0]+half_patch_size)
            y_min_sin=max(0,sin_theta[1]-half_patch_size)
            y_max_sin=min(111,sin_theta[1]+half_patch_size)

            sin_map[x_min_sin:x_max_sin,y_min_sin:y_max_sin]=1
            # sin_map[sin_theta[0]-half_patch_size:sin_theta[0]+half_patch_size,sin_theta[1]-half_patch_size:sin_theta[1]+half_patch_size]=1
            for j in range(len(sin_sample)):
                if i==j:
                    continue
                right_theta=sin_sample[j].cpu().numpy()
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
                overlap_map[i,j]=len(out_index[0])/(len(sin_sample)*len(sin_sample))
        pdb.set_trace()
        one_mean=np.mean(overlap_map)
        out_mean+=[one_mean]
    return out_mean

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
    

def perform_val(multi_gpu, device, embedding_size, batch_size, backbone, data_set, issame, nrof_folds = 10,epoch=0,step=0,logpath='./',visualize=False,stn=None,overlap=False,pre_land=False,keep_land=False,landmarkcnn=None):
    # if multi_gpu:
    #     backbone = backbone.module # unpackage model from DataParallel
    #     backbone = backbone.to(device)
    # else:
    #     backbone = backbone.to(device)
    backbone.eval() # switch to evaluation mode

    embeddings_list = []
    dataset_count=0
    # pdb.set_trace()
    landmark_a_sin=np.array([0,0,112,112])
    landmark_index=np.array(list(range(17, 65))+[66])
    over_lap_all=[]
    for carray in data_set:
        idx = 0
        batch_count=0
        embeddings = np.zeros([len(carray), embedding_size])
        with torch.no_grad():
            while idx + batch_size <= len(carray):
                # pdb.set_trace()
                

                batch = carray[idx:idx + batch_size]
                batch=batch/255.0-0.5
                # landmark_a=[landmark_a_sin]*len(batch)

                # theta=stn.batch_forward(batch,landmark_a)
                # # theta=torch.tensor(theta)
                # theta=F.relu(theta-0)
                # theta=112-F.relu(112-theta)
                # theta=theta[:,landmark_index,:]
                
                #last_time = time.time()
                if pre_land==True:
                    # pdb.set_trace()
                    # batch=batch/255.0*2 -1#
                    land_label,img_reconstructed=landmarkcnn(batch.float().to(device))#div 255/2
                    # img_reconstructed=(img_reconstructed+1)*255.0/2
                    # land_label,img_reconstructed=landmarkcnn(images[0])
                    #reconstructed image to embedding
                    if not keep_land:
                        batch = rearrange(img_reconstructed, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = landmarkcnn.patch_size, p2 = landmarkcnn.patch_size)
                if visualize==True and batch_count in [10,20,30,40,45,50,5,15]:
                    # pdb.set_trace()
                    # embeddings[idx:idx + batch_size],theta = backbone(batch.to(device),visualize=True).cpu()
                    emb,theta = backbone(batch.to(device),visualize=True)#.cpu()
                    # over_lap=calculate_overlap(theta,patch_size=16)
                    # over_lap=calculate_overlap_near(theta,patch_size=16)
                    # emb,theta = backbone(batch.to(device),visualize=True,theta=theta)#.cpu()
                    embeddings[idx:idx + batch_size]=emb.cpu()
                    # save_patch(batch,batch,theta,patch_size=backbone.module.patch_size,save_folder=logpath,iter1=batch_count,epoch=epoch,step=step)
                    save_patch(batch,batch,theta,patch_size=backbone.patch_size,save_folder=logpath,iter1=batch_count,epoch=epoch,step=step)
                elif (visualize==True) and (overlap==True):
                    emb,theta = backbone(batch.to(device),visualize=True)#.cpu()
                    # over_lap=calculate_overlap(theta,patch_size=16)
                    over_lap=calculate_overlap_near(theta,patch_size=8)
                    # pdb.set_trace()
                    over_lap_all+=over_lap
                    embeddings[idx:idx + batch_size] = emb.cpu()
                else:
                    # pdb.set_trace()
                    embeddings[idx:idx + batch_size] = backbone(batch.to(device)).cpu()
                    # embeddings[idx:idx + batch_size] = backbone(batch.to(device),theta=theta).cpu()
                    # embeddings[idx:idx + batch_size]=emb.cpu()
                #batch_time = time.time() - last_time
                #print("batch_time", batch_size, batch_time)
                idx += batch_size
                batch_count+=1
            if idx < len(carray):
                batch = carray[idx:]
                embeddings[idx:] = backbone(batch.to(device)).cpu()
            if (visualize==True) and (overlap==True):
                emb,theta = backbone(batch.to(device),visualize=True)#.cpu()
                # over_lap=calculate_overlap(theta,patch_size=16)
                over_lap=calculate_overlap_near(theta,patch_size=8)
                # pdb.set_trace()
                over_lap_all+=over_lap
        
        embeddings_list.append(embeddings)
        dataset_count+=1
    # # pdb.set_trace()
    # over_lap_mean=np.mean(over_lap_all)
    # over_lap_var=np.var(over_lap_all)
    # print ('mean:'+str(over_lap_mean))
    # print ('var:'+str(over_lap_var))
    _xnorm = 0.0
    _xnorm_cnt = 0
    for embed in embeddings_list:
        for i in range(embed.shape[0]):
            _em = embed[i]
            _norm=np.linalg.norm(_em)
            _xnorm+=_norm
            _xnorm_cnt+=1
    _xnorm /= _xnorm_cnt
    # pdb.set_trace()
    embeddings = embeddings_list[0] + embeddings_list[1]
    embeddings = sklearn.preprocessing.normalize(embeddings)
    print(embeddings.shape)

    tpr, fpr, accuracy, best_thresholds = evaluate(embeddings, issame, nrof_folds)
    buf = gen_plot(fpr, tpr)
    roc_curve = Image.open(buf)
    roc_curve_tensor = transforms.ToTensor()(roc_curve)

    return accuracy.mean(), accuracy.std(), _xnorm, best_thresholds.mean(), roc_curve_tensor
def perform_val_ada(multi_gpu, device, embedding_size, batch_size, backbone, data_set, issame, nrof_folds = 10,epoch=0,step=0,logpath='./',visualize=False,stn=None,overlap=False,pre_land=False,keep_land=False,landmarkcnn=None):
    # if multi_gpu:
    #     backbone = backbone.module # unpackage model from DataParallel
    #     backbone = backbone.to(device)
    # else:
    #     backbone = backbone.to(device)
    backbone.eval() # switch to evaluation mode

    embeddings_list = []
    dataset_count=0
    # pdb.set_trace()
    landmark_a_sin=np.array([0,0,112,112])
    landmark_index=np.array(list(range(17, 65))+[66])
    over_lap_all=[]
    for carray in data_set:
        idx = 0
        batch_count=0
        embeddings = np.zeros([len(carray), embedding_size])
        with torch.no_grad():
            while idx + batch_size <= len(carray):
                # pdb.set_trace()
                

                batch = carray[idx:idx + batch_size]
                batch = batch.numpy()
                batch=batch[:,::-1,:,:]
                batch=torch.from_numpy(batch.copy())
                # batch=batch/255.0-0.5
                # landmark_a=[landmark_a_sin]*len(batch)

                # theta=stn.batch_forward(batch,landmark_a)
                # # theta=torch.tensor(theta)
                # theta=F.relu(theta-0)
                # theta=112-F.relu(112-theta)
                # theta=theta[:,landmark_index,:]
                
                #last_time = time.time()
                if pre_land==True:
                    # pdb.set_trace()
                    # batch=batch/255.0*2 -1#
                    land_label,img_reconstructed=landmarkcnn(batch.float().to(device))#div 255/2
                    # img_reconstructed=(img_reconstructed+1)*255.0/2
                    # land_label,img_reconstructed=landmarkcnn(images[0])
                    #reconstructed image to embedding
                    if not keep_land:
                        batch = rearrange(img_reconstructed, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = landmarkcnn.patch_size, p2 = landmarkcnn.patch_size)
                if visualize==True and batch_count in [10,20,30,40,45,50,5,15]:
                    # pdb.set_trace()
                    # embeddings[idx:idx + batch_size],theta = backbone(batch.to(device),visualize=True).cpu()
                    emb,theta = backbone(batch.to(device),visualize=True)#.cpu()
                    # over_lap=calculate_overlap(theta,patch_size=16)
                    # over_lap=calculate_overlap_near(theta,patch_size=16)
                    # emb,theta = backbone(batch.to(device),visualize=True,theta=theta)#.cpu()
                    embeddings[idx:idx + batch_size]=emb.cpu()
                    # save_patch(batch,batch,theta,patch_size=backbone.module.patch_size,save_folder=logpath,iter1=batch_count,epoch=epoch,step=step)
                    save_patch(batch,batch,theta,patch_size=backbone.patch_size,save_folder=logpath,iter1=batch_count,epoch=epoch,step=step)
                elif (visualize==True) and (overlap==True):
                    emb,theta = backbone(batch.to(device),visualize=True)#.cpu()
                    # over_lap=calculate_overlap(theta,patch_size=16)
                    over_lap=calculate_overlap_near(theta,patch_size=8)
                    # pdb.set_trace()
                    over_lap_all+=over_lap
                    embeddings[idx:idx + batch_size] = emb.cpu()
                else:
                    # pdb.set_trace()
                    embeddings[idx:idx + batch_size] = backbone(batch.to(device)).cpu()
                    # embeddings[idx:idx + batch_size] = backbone(batch.to(device),theta=theta).cpu()
                    # embeddings[idx:idx + batch_size]=emb.cpu()
                #batch_time = time.time() - last_time
                #print("batch_time", batch_size, batch_time)
                idx += batch_size
                batch_count+=1
            if idx < len(carray):
                batch = carray[idx:]
                batch = batch.numpy()
                batch=batch[:,::-1,:,:]
                batch=torch.from_numpy(batch.copy())
                embeddings[idx:] = backbone(batch.to(device)).cpu()
            if (visualize==True) and (overlap==True):
                emb,theta = backbone(batch.to(device),visualize=True)#.cpu()
                # over_lap=calculate_overlap(theta,patch_size=16)
                over_lap=calculate_overlap_near(theta,patch_size=8)
                # pdb.set_trace()
                over_lap_all+=over_lap
        
        embeddings_list.append(embeddings)
        dataset_count+=1
    # # pdb.set_trace()
    # over_lap_mean=np.mean(over_lap_all)
    # over_lap_var=np.var(over_lap_all)
    # print ('mean:'+str(over_lap_mean))
    # print ('var:'+str(over_lap_var))
    _xnorm = 0.0
    _xnorm_cnt = 0
    for embed in embeddings_list:
        for i in range(embed.shape[0]):
            _em = embed[i]
            _norm=np.linalg.norm(_em)
            _xnorm+=_norm
            _xnorm_cnt+=1
    _xnorm /= _xnorm_cnt
    # pdb.set_trace()
    embeddings = embeddings_list[0] + embeddings_list[1]
    embeddings = sklearn.preprocessing.normalize(embeddings)
    print(embeddings.shape)

    tpr, fpr, accuracy, best_thresholds = evaluate(embeddings, issame, nrof_folds)
    buf = gen_plot(fpr, tpr)
    roc_curve = Image.open(buf)
    roc_curve_tensor = transforms.ToTensor()(roc_curve)

    return accuracy.mean(), accuracy.std(), _xnorm, best_thresholds.mean(), roc_curve_tensor

def perform_val_twomodel(multi_gpu, device, embedding_size, batch_size, backbone,backbone2, data_set, issame, nrof_folds = 10,epoch=0,step=0,logpath='./',visualize=False,stn=None):
    # if multi_gpu:
    #     backbone = backbone.module # unpackage model from DataParallel
    #     backbone = backbone.to(device)
    # else:
    #     backbone = backbone.to(device)
    pdb.set_trace()
    backbone.eval() # switch to evaluation mode
    backbone2.eval()
    embeddings_list1 = []
    embeddings_list2 = []
    dataset_count=0
    # pdb.set_trace()
    # landmark_a_sin=np.array([0,0,112,112])
    # landmark_index=np.array(list(range(17, 65))+[66])
    for carray in data_set:
        idx = 0
        batch_count=0
        embeddings1 = np.zeros([len(carray), embedding_size])
        embeddings2=np.zeros([len(carray), embedding_size])
        with torch.no_grad():
            while idx + batch_size <= len(carray):
                # pdb.set_trace()
                

                batch = carray[idx:idx + batch_size]

                # landmark_a=[landmark_a_sin]*len(batch)

                # theta=stn.batch_forward(batch,landmark_a)
                # # theta=torch.tensor(theta)
                # theta=F.relu(theta-0)
                # theta=112-F.relu(112-theta)
                # theta=theta[:,landmark_index,:]

                #last_time = time.time()
                if visualize==True and batch_count in [10,20,30,40,45,50]:
                    # pdb.set_trace()
                    # embeddings[idx:idx + batch_size],theta = backbone(batch.to(device),visualize=True).cpu()
                    emb,theta = backbone(batch.to(device),visualize=True)#.cpu()
                    # emb,theta = backbone(batch.to(device),visualize=True,theta=theta)#.cpu()
                    embeddings[idx:idx + batch_size]=emb.cpu()
                    save_patch(batch,batch,theta,patch_size=backbone.module.patch_size,save_folder=logpath,iter1=batch_count,epoch=epoch,step=step)
                    # save_patch(batch,batch,theta,patch_size=backbone.patch_size,save_folder=logpath,iter1=batch_count,epoch=epoch,step=step)
                else:
                    embeddings1[idx:idx + batch_size] = backbone(batch.to(device)).cpu()
                    embeddings2[idx:idx + batch_size] = backbone2(batch.to(device)).cpu()
                    # embeddings[idx:idx + batch_size] = backbone(batch.to(device),theta=theta).cpu()
                    # embeddings[idx:idx + batch_size]=emb.cpu()
                #batch_time = time.time() - last_time
                #print("batch_time", batch_size, batch_time)
                idx += batch_size
                batch_count+=1
            if idx < len(carray):
                batch = carray[idx:]
                embeddings1[idx:] = backbone(batch.to(device)).cpu()
                embeddings2[idx:] = backbone2(batch.to(device)).cpu()
        # embeddings=0.5*(embeddings1+embeddings2)
        embeddings_list1.append(embeddings1)
        embeddings_list2.append(embeddings2)
        dataset_count+=1

    _xnorm = 0.0
    _xnorm_cnt = 0
    for embed in embeddings_list1:
        for i in range(embed.shape[0]):
            _em = embed[i]
            _norm=np.linalg.norm(_em)
            _xnorm+=_norm
            _xnorm_cnt+=1
    _xnorm /= _xnorm_cnt
    pdb.set_trace()
    embeddings1 = embeddings_list1[0] + embeddings_list1[1]
    embeddings1 = sklearn.preprocessing.normalize(embeddings1)
    # print(embeddings.shape)

    embeddings2 = embeddings_list2[0] + embeddings_list2[1]
    embeddings2 = sklearn.preprocessing.normalize(embeddings2)
    # print(embeddings.shape)

    tpr, fpr, accuracy, best_thresholds = evaluate_two(embeddings1, embeddings2, issame, nrof_folds)
    buf = gen_plot(fpr, tpr)
    roc_curve = Image.open(buf)
    roc_curve_tensor = transforms.ToTensor()(roc_curve)

    return accuracy.mean(), accuracy.std(), _xnorm, best_thresholds.mean(), roc_curve_tensor


def perform_val_token(multi_gpu, device, embedding_size, batch_size, backbone, data_set, issame, nrof_folds = 10,epoch=0,step=0,logpath='./',visualize=False,stn=None):
    # if multi_gpu:
    #     backbone = backbone.module # unpackage model from DataParallel
    #     backbone = backbone.to(device)
    # else:
    #     backbone = backbone.to(device)
    backbone.eval() # switch to evaluation mode

    embeddings_list = []
    dataset_count=0
    # pdb.set_trace()
    # landmark_a_sin=np.array([0,0,112,112])
    # landmark_index=np.array(list(range(17, 65))+[66])
    for carray in data_set:
        idx = 0
        batch_count=0
        embeddings = np.zeros([len(carray), embedding_size])
        with torch.no_grad():
            while idx + batch_size <= len(carray):
                # pdb.set_trace()
                

                batch = carray[idx:idx + batch_size]

                # landmark_a=[landmark_a_sin]*len(batch)

                # theta=stn.batch_forward(batch,landmark_a)
                # # theta=torch.tensor(theta)
                # theta=F.relu(theta-0)
                # theta=112-F.relu(112-theta)
                # theta=theta[:,landmark_index,:]

                #last_time = time.time()
                if visualize==True and batch_count in [10,20,30,40,45,50]:
                    # pdb.set_trace()
                    # embeddings[idx:idx + batch_size],theta = backbone(batch.to(device),visualize=True).cpu()
                    emb,theta = backbone(batch.to(device),visualize=True,save_token=True)#.cpu()
                    # emb,theta = backbone(batch.to(device),visualize=True,theta=theta)#.cpu()
                    embeddings[idx:idx + batch_size]=emb.cpu()
                    # save_patch(batch,batch,theta,patch_size=backbone.module.patch_size,save_folder=logpath,iter1=batch_count,epoch=epoch,step=step)
                    # 
                    save_patch(batch,batch,theta,patch_size=backbone.patch_size,save_folder=logpath,iter1=batch_count,epoch=epoch,step=step)
                    # attention_score=[]
                    # # pdb.set_trace()
                    # for i in range(len(backbone.transformer.layers)):
                    #     single_attention_layer=torch.unsqueeze(backbone.transformer.layers[-1][0].fn.fn.attention_score[0],0)
                    #     attention_score+=[single_attention_layer.cpu()]
                    # pdb.set_trace()
                    # attention_score=np.vstack(attention_score)
                    # visualize_attentionmap_new_landmark(batch[0],attention_score,single_fig_landmarks=theta[0])
                else:
                    
                    # embeddings[idx:idx + batch_size] = backbone(batch.to(device)).cpu()
                    results= backbone(batch.to(device),save_token=True)#.cpu()
                    pdb.set_trace()
                    # b=torch.cat((results[0],results[1]),dim=1)
                    embeddings[idx:idx + batch_size]=results[0].cpu()
                    # embeddings[idx:idx + batch_size]=b.cpu()
                    theta=results[2].cpu()  #landmark
                    # attention_score=backbone.transformer.layers[0][0].fn.fn.attention_score
                    attention_score=[]
                    # pdb.set_trace()
                    for i in range(len(backbone.transformer.layers)):
                        single_attention_layer=torch.unsqueeze(backbone.transformer.layers[-1][0].fn.fn.attention_score[1],0)
                        attention_score+=[single_attention_layer.cpu()]
                    # pdb.set_trace()
                    attention_score=np.vstack(attention_score)
                    # visualize_attentionmap_new_landmark(batch[0],attention_score,single_fig_landmarks=theta[0],count=batch_count)
                    # visualize_attentionmap_new(batch[0],attention_score,count=batch_count)
                    # visualize_attentionmap_DINO(batch[1],attention_score,count=batch_count)
                    # visualize_attentionmap_DINO(batch[0],attention_score,count=batch_count)
                    # if batch_count>70:
                    # visualize_attentionmap_DINO(batch[1],attention_score,count=batch_count)
                    visualize_attentionmap_DINO_landmark(batch[0],attention_score,single_fig_landmarks=theta[0],count=batch_count)

                    # #DINO 
                    # pdb.set_trace()
                    # attentions=torch.unsqueeze(backbone.transformer.layers[-1][0].fn.fn.attention_score[0],0)

                    # nh = attentions.shape[1]#number of head
                    # w_featmap=112//28
                    # h_featmap=112//28
                    
                    # attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

                    # attentions = attentions.reshape(nh, w_featmap, h_featmap)
                    # attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=28, mode="nearest")[0].cpu().numpy()
                    # # embeddings[idx:idx + batch_size] = backbone(batch.to(device),theta=theta).cpu()
                    # # embeddings[idx:idx + batch_size]=emb.cpu()
                #batch_time = time.time() - last_time
                #print("batch_time", batch_size, batch_time)
                idx += batch_size
                batch_count+=1
            if idx < len(carray):
                batch = carray[idx:]
                results= backbone(batch.to(device),save_token=True)#.cpu()
                b=torch.cat((results[0],results[1]),dim=1)
                # embeddings[idx:idx + batch_size]=results[1].cpu()
                embeddings[idx:]=b.cpu()
                # embeddings[idx:]=results[1].cpu()
                # embeddings[idx:] = backbone(batch.to(device)).cpu()
        embeddings_list.append(embeddings)
        dataset_count+=1

    _xnorm = 0.0
    _xnorm_cnt = 0
    # pdb.set_trace()
    for embed in embeddings_list:
        for i in range(embed.shape[0]):
            _em = embed[i]
            _norm=np.linalg.norm(_em)
            _xnorm+=_norm
            _xnorm_cnt+=1
    _xnorm /= _xnorm_cnt
    # pdb.set_trace()
    embeddings = embeddings_list[0] + embeddings_list[1]
    out_embedding=np.zeros_like(embeddings)
    # for i in range(embeddings.shape[1]):
    #     out_embedding[:,i,:] = sklearn.preprocessing.normalize(embeddings[:,i,:])
    out_embedding = sklearn.preprocessing.normalize(embeddings)
    # pdb.set_trace()
    embeddings=out_embedding
    print(embeddings.shape)

    tpr, fpr, accuracy, best_thresholds = evaluate_token(embeddings, issame, nrof_folds)
    buf = gen_plot(fpr, tpr)
    roc_curve = Image.open(buf)
    roc_curve_tensor = transforms.ToTensor()(roc_curve)

    return accuracy.mean(), accuracy.std(), _xnorm, best_thresholds.mean(), roc_curve_tensor

def visualize_attentionmap(single_input_image,single_attention_score,grid_size=4):
   
    import matplotlib.pyplot as plt
    import numpy as np;
    np.random.seed(0)
    # import seaborn as sns
    import cv2
    # pdb.set_trace()
    # sns.set()
    # pdb.set_trace()
    single_input_image=single_input_image.permute(1,2,0)
    single_input_image=single_input_image.numpy()
    single_input_image=single_input_image[:,:,[2,1,0]]
    reshaped = single_attention_score.reshape(
        (12, 11, grid_size ** 2 + 1, grid_size ** 2 + 1)
    )
    reshaped = reshaped.mean(axis=1)

    # Recursively multiply the weight matrices
    v = reshaped[-1]
    for n in range(1, len(reshaped)):
        v = np.matmul(v, reshaped[-1 - n])
    pdb.set_trace()
    mask = v[0, 1:].reshape(grid_size, grid_size)
    # mask = cv2.resize(mask -mask.min()/ (mask.max()-mask.min()), single_input_image.shape[:2])[..., np.newaxis]
    mask = cv2.resize(mask / mask.max(), single_input_image.shape[:2])[..., np.newaxis]
    # single_input_image[:,:,0]=(mask * single_input_image[:,:,0]).astype("uint8")
    # result_img=single_input_image.astype("uint8")
    
    result_img=(mask * single_input_image).astype("uint8")
    cv2.imwrite('save_as_a_png.png',result_img)
    # return (mask * image).astype("uint8")
    # df=single_attention_score[0].cpu()
    # df_norm_col=(df-df.mean())/df.std()
    # heatmap = sns.heatmap(df_norm_col)
    # plt.savefig('save_as_a_png.png')

def visualize_attentionmap_new(single_input_image,single_attention_score,grid_size=4,cmap="jet",count=0):
   
    import matplotlib.pyplot as plt
    import numpy as np;
    np.random.seed(0)
    # import seaborn as sns
    import cv2
    # pdb.set_trace()
    # sns.set()
    # pdb.set_trace()
    single_input_image=single_input_image.permute(1,2,0)
    single_input_image=single_input_image.numpy()
    single_input_image=single_input_image[:,:,[2,1,0]]
    reshaped = single_attention_score.reshape(
        (12, 11, grid_size ** 2 + 1, grid_size ** 2 + 1)
    )
    reshaped = reshaped.mean(axis=1)

    # Recursively multiply the weight matrices
    v = reshaped[-1]
    for n in range(1, len(reshaped)):
        v = np.matmul(v, reshaped[-1 - n])
    # pdb.set_trace()
    mask = v[0, 1:].reshape(grid_size, grid_size)
    mask = cv2.resize((mask -mask.min())/ (mask.max()-mask.min()), single_input_image.shape[:2])[..., np.newaxis]*255.0
    # mask = cv2.resize(mask / mask.max(), single_input_image.shape[:2])[..., np.newaxis]
    # single_input_image[:,:,0]=(mask * single_input_image[:,:,0]).astype("uint8")
    # result_img=single_input_image.astype("uint8")
    # pdb.set_trace()
    result_img=(mask * single_input_image).astype("uint8")
    cv2.imwrite('save_as_a_png.png',result_img)

    plt.imshow(single_input_image.astype("uint8"), alpha=1)
    plt.imshow(mask, alpha=0.5, interpolation='nearest', cmap=cmap)
    # plt.savefig('./attention_map/{}.png',%count)
    plt.savefig('./pure_attention/pure_vit_{count}.png'.format(count=count))
    # return (mask * image).astype("uint8")
    # df=single_attention_score[0].cpu()
    # df_norm_col=(df-df.mean())/df.std()
    # heatmap = sns.heatmap(df_norm_col)
    # plt.savefig('save_as_a_png.png')
def visualize_attentionmap_DINO(single_input_image,single_attention_score,grid_size=14,cmap="jet",count=0):
   
 
    # import numpy as np;
    # np.random.seed(0)
    # # import seaborn as sns
    # import cv2
    # pdb.set_trace()
    # sns.set()
    pdb.set_trace()
    single_input_image=single_input_image.permute(1,2,0)
    single_input_image=single_input_image.numpy()
    single_input_image=single_input_image[:,:,[2,1,0]]
    reshaped = single_attention_score.reshape(
        (12, 11, grid_size ** 2 + 1, grid_size ** 2 + 1)
    )
    # pdb.set_trace()
    nh=11
    w_featmap=grid_size#112//28
    h_featmap=grid_size#112//28
    attentions=torch.tensor(reshaped[0,:,0,1:].reshape(nh, -1))
    # reshaped = reshaped.mean(axis=1)
    # pdb.set_trace()
    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=112//grid_size, mode="nearest")[0].cpu().numpy()
    # # Recursively multiply the weight matrices
    # v = reshaped[-1]
    # for n in range(1, len(reshaped)):
    #     v = np.matmul(v, reshaped[-1 - n])
    # # pdb.set_trace()
    # mask = v[0, 1:].reshape(grid_size, grid_size)
    # mask = cv2.resize((mask -mask.min())/ (mask.max()-mask.min()), single_input_image.shape[:2])[..., np.newaxis]*255.0
    # # mask = cv2.resize(mask / mask.max(), single_input_image.shape[:2])[..., np.newaxis]
    # # single_input_image[:,:,0]=(mask * single_input_image[:,:,0]).astype("uint8")
    # # result_img=single_input_image.astype("uint8")
    # # pdb.set_trace()
    # result_img=(mask * single_input_image).astype("uint8")
    # cv2.imwrite('save_as_a_png.png',result_img)
    for i in range(nh):
        # if i!=5:
        #     continue
        plt.imshow(single_input_image.astype("uint8"), alpha=1)
        plt.imshow(attentions[i], alpha=0.5, interpolation='nearest', cmap=cmap)
        # plt.savefig('./attention_map/{}.png',%count)
        plt.savefig('./pure_attention/pure_vit_{count}_{i}.png'.format(count=count,i=i))
    # return (mask * image).astype("uint8")
    # df=single_attention_score[0].cpu()
    # df_norm_col=(df-df.mean())/df.std()
    # heatmap = sns.heatmap(df_norm_col)
    # plt.savefig('save_as_a_png.png')


def visualize_attentionmap_DINO_landmark(single_input_image,single_attention_score,single_fig_landmarks,grid_size=14,cmap="jet",count=0):
   
 
    # import numpy as np;
    # np.random.seed(0)
    # # import seaborn as sns
    # import cv2
    # pdb.set_trace()
    # sns.set()
    # pdb.set_trace()
    single_input_image=single_input_image.permute(1,2,0)
    single_input_image=single_input_image.numpy()
    single_input_image=single_input_image[:,:,[2,1,0]]
    reshaped = single_attention_score.reshape(
        (12, 11, grid_size ** 2 + 1, grid_size ** 2 + 1)
    )
    pdb.set_trace()
    nh=11
    w_featmap=grid_size#112//28
    h_featmap=grid_size#112//28
    attentions=torch.tensor(reshaped[0,:,0,1:].reshape(nh, -1))
    # reshaped = reshaped.mean(axis=1)
    # pdb.set_trace()
    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=112//grid_size, mode="nearest")[0].cpu().numpy()
    # # Recursively multiply the weight matrices
    # v = reshaped[-1]
    # for n in range(1, len(reshaped)):
    #     v = np.matmul(v, reshaped[-1 - n])
    # # pdb.set_trace()
    # mask = v[0, 1:].reshape(grid_size, grid_size)
    # mask = cv2.resize((mask -mask.min())/ (mask.max()-mask.min()), single_input_image.shape[:2])[..., np.newaxis]*255.0
    # # mask = cv2.resize(mask / mask.max(), single_input_image.shape[:2])[..., np.newaxis]
    # # single_input_image[:,:,0]=(mask * single_input_image[:,:,0]).astype("uint8")
    # # result_img=single_input_image.astype("uint8")
    # # pdb.set_trace()
    # result_img=(mask * single_input_image).astype("uint8")
    # cv2.imwrite('save_as_a_png.png',result_img)
    new_landmar=np.zeros_like(attentions)
    landmark_size=single_input_image.shape[1]//grid_size
    half_landmark_size=landmark_size//2
    # original_
    
    for i in range(grid_size ** 2 ):
        x=i//grid_size
        y=i%grid_size
        coor_xy=np.rint(single_fig_landmarks[i]).numpy().astype(int)
        if coor_xy[0]-half_landmark_size<=0:
            coor_xy[0]=coor_xy[0]+ np.abs(coor_xy[0]-half_landmark_size)
        if coor_xy[1]-half_landmark_size<=0:
            coor_xy[1]=coor_xy[1]+ np.abs(coor_xy[1]-half_landmark_size)
        if coor_xy[0]+half_landmark_size>=112:
            coor_xy[0]=coor_xy[0]- np.abs(coor_xy[0]+half_landmark_size-112)
        if coor_xy[1]+half_landmark_size>=112:
            coor_xy[1]=coor_xy[1]- np.abs(coor_xy[1]+half_landmark_size-112)
        new_landmar[:,coor_xy[1]-half_landmark_size:coor_xy[1]+half_landmark_size, coor_xy[0]-half_landmark_size:coor_xy[0]+half_landmark_size]   +=attentions[:,x*landmark_size:(x+1)*landmark_size, y*landmark_size:(y+1)*landmark_size]
    # pdb.set_trace()
    for i in range(nh):
        # if i!=5:
        #     continue
        plt.imshow(single_input_image.astype("uint8"), alpha=1)
        plt.imshow(new_landmar[i], alpha=0.5, interpolation='nearest', cmap=cmap)
        # plt.savefig('./attention_map/{}.png',%count)
        plt.savefig('./landmark_attention/pure_vit_{count}_{i}.png'.format(count=count,i=i))
    # return (mask * image).astype("uint8")
    # df=single_attention_score[0].cpu()
    # df_norm_col=(df-df.mean())/df.std()
    # heatmap = sns.heatmap(df_norm_col)
    # plt.savefig('save_as_a_png.png')

def visualize_attentionmap_new_landmark(single_input_image,single_attention_score,single_fig_landmarks,grid_size=14,cmap="jet",count=0):
   
    # import matplotlib.pyplot as plt
    # import numpy as np;
    np.random.seed(0)
    # import seaborn as sns
    import cv2
    # pdb.set_trace()
    # sns.set()
    pdb.set_trace()
    single_input_image=single_input_image.permute(1,2,0)
    single_input_image=single_input_image.numpy()
    single_input_image=single_input_image[:,:,[2,1,0]]
    reshaped = single_attention_score.reshape(
        (12, 11, grid_size ** 2 + 1, grid_size ** 2 + 1)
    )
    reshaped = reshaped.mean(axis=1)

    # Recursively multiply the weight matrices
    v = reshaped[-1]
    for n in range(1, len(reshaped)):
        v = np.matmul(v, reshaped[-1 - n])
    # pdb.set_trace()
    mask = v[0, 1:].reshape(grid_size, grid_size)
    mask = cv2.resize((mask -mask.min())/ (mask.max()-mask.min()), single_input_image.shape[:2])[..., np.newaxis]*255.0
    #move to landmark location
    new_landmar=np.zeros_like(mask)
    landmark_size=single_input_image.shape[1]//grid_size
    half_landmark_size=landmark_size//2
    # original_
    # pdb.set_trace()
    for i in range(grid_size ** 2 ):
        x=i//grid_size
        y=i%grid_size
        coor_xy=np.rint(single_fig_landmarks[i]).numpy().astype(int)
        if coor_xy[0]-half_landmark_size<=0:
            coor_xy[0]=coor_xy[0]+ np.abs(coor_xy[0]-half_landmark_size)
        if coor_xy[1]-half_landmark_size<=0:
            coor_xy[1]=coor_xy[1]+ np.abs(coor_xy[1]-half_landmark_size)
        if coor_xy[0]+half_landmark_size>=112:
            coor_xy[0]=coor_xy[0]- np.abs(coor_xy[0]+half_landmark_size-112)
        if coor_xy[1]+half_landmark_size>=112:
            coor_xy[1]=coor_xy[1]- np.abs(coor_xy[1]+half_landmark_size-112)
        new_landmar[coor_xy[1]-half_landmark_size:coor_xy[1]+half_landmark_size, coor_xy[0]-half_landmark_size:coor_xy[0]+half_landmark_size:]+=mask[x*landmark_size:(x+1)*landmark_size, y*landmark_size:(y+1)*landmark_size,:]

    # mask = cv2.resize(mask / mask.max(), single_input_image.shape[:2])[..., np.newaxis]
    # single_input_image[:,:,0]=(mask * single_input_image[:,:,0]).astype("uint8")
    # result_img=single_input_image.astype("uint8")
    # pdb.set_trace()
    # result_img=(mask * single_input_image).astype("uint8")
    # cv2.imwrite('save_as_a_png.png',result_img)
    # pdb.set_trace()
    plt.imshow(single_input_image.astype("uint8"), alpha=1)
    plt.imshow(new_landmar, alpha=0.5, interpolation='nearest', cmap=cmap)
    plt.savefig('./landmark_attention/pure_vit_{count}.png'.format(count=count))
    # return (mask * image).astype("uint8")
    # df=single_attention_score[0].cpu()
    # df_norm_col=(df-df.mean())/df.std()
    # heatmap = sns.heatmap(df_norm_col)
    # plt.savefig('save_as_a_png.png')
def perform_val_deit(multi_gpu, device, embedding_size, batch_size, backbone, dis_token, data_set, issame, nrof_folds = 10):
    if multi_gpu:
        backbone = backbone.module # unpackage model from DataParallel
        backbone = backbone.to(device)
    else:
        backbone = backbone.to(device)
    backbone.eval() # switch to evaluation mode

    embeddings_list = []
    for carray in data_set:
        idx = 0
        embeddings = np.zeros([len(carray), embedding_size])
        with torch.no_grad():
            while idx + batch_size <= len(carray):
                batch = carray[idx:idx + batch_size]
                #last_time = time.time()
                #embed()
                fea,token = backbone(batch.to(device), dis_token.to(device))
                embeddings[idx:idx + batch_size] = fea.cpu()
                #batch_time = time.time() - last_time
                #print("batch_time", batch_size, batch_time)
                idx += batch_size
            if idx < len(carray):
                batch = carray[idx:]
                embeddings[idx:] = backbone(batch.to(device)).cpu()
        embeddings_list.append(embeddings)

    _xnorm = 0.0
    _xnorm_cnt = 0
    for embed in embeddings_list:
        for i in range(embed.shape[0]):
            _em = embed[i]
            _norm=np.linalg.norm(_em)
            _xnorm+=_norm
            _xnorm_cnt+=1
    _xnorm /= _xnorm_cnt

    embeddings = embeddings_list[0] + embeddings_list[1]
    embeddings = sklearn.preprocessing.normalize(embeddings)
    print(embeddings.shape)

    tpr, fpr, accuracy, best_thresholds = evaluate(embeddings, issame, nrof_folds)
    buf = gen_plot(fpr, tpr)
    roc_curve = Image.open(buf)
    roc_curve_tensor = transforms.ToTensor()(roc_curve)

    return accuracy.mean(), accuracy.std(), _xnorm, best_thresholds.mean(), roc_curve_tensor

def buffer_val(writer, db_name, acc, std, xnorm, best_threshold, roc_curve_tensor, batch):
    writer.add_scalar('Accuracy/{}_Accuracy'.format(db_name), acc, batch)
    writer.add_scalar('Std/{}_Std'.format(db_name), std, batch)
    writer.add_scalar('XNorm/{}_XNorm'.format(db_name), xnorm, batch)
    writer.add_scalar('Threshold/{}_Best_Threshold'.format(db_name), best_threshold, batch)
    writer.add_image('ROC/{}_ROC_Curve'.format(db_name), roc_curve_tensor, batch)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val   = 0
        self.avg   = 0
        self.sum   = 0
        self.count = 0

    def update(self, val, n = 1):
        self.val   = val
        self.sum   += val * n
        self.count += n
        self.avg   = self.sum / self.count

'''
def train_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred    = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res
'''

def train_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred    = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    #embed()
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res[0]

def extract_patches_pytorch_gridsample(imgs, landmarks, patch_shape,num_landm=49):#numpy
    """ Extracts patches from an image.
    Args:
        imgs: a numpy array of dimensions [batch_size, width, height, channels]
        landmarks: a numpy array of dimensions [num_patches, 2]
        patch_shape: (width, height)
    Returns:
        a numpy array [num_patches, width, height, channels]
    """

    # patch_shape = np.array(patch_shape)
    # patch_shape = np.array(patch_shape)
    # patch_half_shape = torch.require(torch.round(patch_shape / 2), dtype=int)
    img_shape=imgs.shape[2]
    device=landmarks.device
    imgs=imgs.to(device)
    # pdb.set_trace()
    list_patches = []
    patch_half_shape=patch_shape/2
    start = -patch_half_shape
    end = patch_half_shape
    # sampling_grid = torch.meshgrid[start[0]:end[0], start[1]:end[1]]
    sampling_grid = torch.meshgrid(torch.arange(start[0],end[0]),torch.arange(start[1],end[1]))#         start[0]:end[0], start[1]:end[1]]
    sampling_grid=torch.stack(sampling_grid,dim=0).to(device)#cuda()
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
#visualize
def save_patch(images,patched_images,landmarks,patch_size=28,save_folder='patch_imgs',iter1=0,epoch=0,step=0,keep_num=20):
    
    # inv_normalize = transforms.Normalize(
    #             mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    #             std=[1/0.229, 1/0.224, 1/0.255]
    #         )
    # topil=transforms.ToPILImage()
    # images=images.cpu().numpy()#np.array(images)
    # patched_images=patched_images.cpu().numpy()#np.array(patched_images)
    # landmarks=landmarks.cpu().numpy()#np.array(landmarks)
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    # mean=[0.485, 0.456, 0.406]
    # std=[0.229, 0.224, 0.225]
    # mean=[0.485, 0.456, 0.406]
    # std=[0.229, 0.224, 0.225]
    # images[:, 0, :, :] = images[:, 0, :, :] * std[0] + mean[0]
    # images[:, 1, :, :] = images[:, 1, :, :] * std[1] + mean[1]
    # images[:, 2, :, :] = images[:, 2, :, :] * std[2] + mean[2]
    # pdb.set_trace()
    # images=0.5*(images+1)
    # images=images+0.5
    images=images/255.0
    patch_shape=torch.tensor([patch_size,patch_size])
    
    patched_images=extract_patches_pytorch_gridsample(images.cuda(),landmarks,patch_shape=patch_shape,num_landm=4*4)
    
    # patched_images[:, 0, :, :] = patched_images[:, 0, :, :] * std[0] + mean[0]
    # patched_images[:, 1, :, :] = patched_images[:, 1, :, :] * std[1] + mean[1]
    # patched_images[:, 2, :, :] = patched_images[:, 2, :, :] * std[2] + mean[2]
    images=images.cpu().numpy()#np.array(images)
    patched_images=patched_images.cpu().numpy()#np.array(patched_images)
    landmarks=landmarks.cpu().numpy()#np.array(landmarks)
    
    for i in range(keep_num):
        sing_img=images[i]
        #bgr to rgb
        sing_patchedimg=patched_images[i]
        #bgr to rgb
        sing_img=np.transpose(sing_img,[1,2,0])
        sing_img = sing_img.copy()#[...,::-1]
        sing_patchedimg=np.transpose(sing_patchedimg,[1,2,0])
        sing_patchedimg = sing_patchedimg.copy()#[...,::-1]
        sing_land=landmarks[i]

        save_folder_spe=os.path.join(save_folder,'epoch{}_step{}'.format(epoch,step))
        if not os.path.exists(save_folder_spe):
            os.mkdir(save_folder_spe)
         
        save_name=os.path.join(save_folder,'epoch{}_step{}/iter{}_order{}.png'.format(epoch,step,iter1,i))
        
        plot_landmark(sing_img,sing_patchedimg,sing_land,patch_shape=patch_size,save_path=save_name)

# rand_num=range(8)
rand_num=range(144)
# rand_num=np.random.randint(0,15,size=[5])
print(rand_num)
def plot_landmark(original_img, patch_img,landmarks,patch_shape=16,save_path='example.png',plot_rec=False):
    #landmark format num*2
    number = len(rand_num)
    cmap = plt.get_cmap('Spectral')
    colors = [cmap(i) for i in np.linspace(0, 1, number)]
    half_shape=patch_shape/2
    coords = []
    landmarks=landmarks[rand_num]
    for land in landmarks:
        if land[0]==0 or land[1]==0:
            continue
        if land[0]>=111 or land[1]>=111:
            continue
        width=patch_shape
        height=patch_shape
        left=land[0]-half_shape
        top=land[1]-half_shape

        coords.append([left,top,width,height,land[0],land[1]])
    # plt.figure()
    # pdb.set_trace()
    fig=plt.figure()
    # ax = fig.add_subplot(1,3,1)
    # ax.imshow(original_img)
    # ax = fig.add_subplot(1,3,2)
    # ax.imshow(patch_img)
    # plt.subplot(1, 2, 1)
    # plt.imshow(original_img)
    # plt.subplot(1, 2, 2)
    # plt.imshow(patch_img)
    ax = fig.add_subplot(1,1,1)
    # fig, ax = plt.subplots(1, 3,3)
    ax.imshow(original_img)
    currentAxis = fig.gca()
    
    #
    # pdb.set_trace()
    for index, coord in enumerate(coords):
        # coord=coords[10]
        if plot_rec:
            rect = patches.Rectangle((coord[0], coord[1]), coord[2], coord[3], 
                                    linewidth=3, edgecolor=colors[index],facecolor='none')
            currentAxis.add_patch(rect)
        else:
            # if left==0 or top==0:
            #     continue
            # if (left+width)=112 or (top+height)==112:
            #     continue
            ax.plot(coord[4], coord[5],color=colors[index],marker = "o",markersize=8)
    # pdb.set_trace()
    # plt.show()
    plt.savefig(save_path,bbox_inches='tight', pad_inches=0)
    plt.close('all')
    # print(0)