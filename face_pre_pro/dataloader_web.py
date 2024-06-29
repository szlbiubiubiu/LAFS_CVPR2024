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
# from util import rand_aa_face
# from RandAugment import RandAugment
from IPython import embed
import pdb
from PIL import Image
import torchvision.datasets as datasets
import json
import os
import cv2


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

class FaceDataset(data.Dataset):
    def __init__(self, path_imgrec,dino_trans=None, rand_mirror=False,random_resizecrop=False,img_size=112,rand_au=False,config_str='rand-m2-mstd0.5-inc1',sifenzhiyi=False,partition=None
    ,filepath_id_nidex='ms1m_random_index.json'):#rand-m2-n3-mstd0.5-inc1
    
    '''filepath_id_nidex is used to sample random images from ms1mv3 dataset, 
        not suitable for 1M WebFace dataset
    '''
        self.rand_mirror = rand_mirror
        self.random_resizecrop=random_resizecrop
        self.rand_au=rand_au
        self.dino_trans=dino_trans
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
        if path_imgrec:
            logging.info('loading recordio %s...',
                         path_imgrec)
            path_imgidx = path_imgrec[0:-4] + ".idx"
            print(path_imgrec, path_imgidx)
            self.imgrec = recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
            s = self.imgrec.read_idx(0)
            header, _ = recordio.unpack(s)
            # pdb.set_trace()
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

            # pdb.set_trace()
            # if partition:
            #     len_sammples=np.int64(len(self.seq)*partition)
            #     self.seq=self.seq[:len_sammples]
            #取后1/4
            if partition:
                
                if os.path.exists(filepath_id_nidex):
                    with open(filepath_id_nidex, 'r') as f:
                        # f=open(filepath_id_nidex,'r')
                        out = json.load(f)
                        # self.samples=out
                        # pdb.set_trace()
                        ran_ord=out['index']
                else:
                    len_sammples=np.int64(len(self.seq)*0.4)
                    ran_ord=torch.randperm(len(self.seq))[:len_sammples].tolist()
                    # pdb.set_trace()
                    with open(filepath_id_nidex, 'w') as f:
                        # f=open(filepath_id_nidex,'w')
                        # out = f(self, directory, *args, **kwargs)
                        # json.dump({'a': list(self.id_index)},f,cls=NumpyEncoder)
                        dic={}
                        dic['index']=ran_ord#.tolist()
                        json.dump( dic,f)


                # len_sammples=np.int64(len(self.seq)*0.12)
                # ran_ord=torch.randperm(len(self.seq))[:len_sammples]
                # self.seq=self.seq[ran_ord]
                self.seq = [self.seq[i] for i in ran_ord]
                # len_sammples=np.int64(len(self.seq)*0.12)
                # self.seq=self.seq[-len_sammples:]
            # pdb.set_trace()

    def __getitem__(self, index):
        idx = self.seq[index]
        s = self.imgrec.read_idx(idx)
        # pdb.set_trace()
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
        # if 'webface' in self.path_imgrec:
        #     _data=_data[::-1,:,:]
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
        if self.dino_trans is not None:
        #DINO augmentation
            _data=np.transpose(_data,(1,2,0))
            _data = Image.fromarray(_data)
            img=self.dino_trans(_data)
        return img, label

    def __len__(self):
        return len(self.seq)

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
class FaceDataset_gen_1imgperid(data.Dataset):
    def __init__(self, path_imgrec,dino_trans=None, rand_mirror=False,random_resizecrop=False,img_size=112,rand_au=False,config_str='rand-m2-mstd0.5-inc1'):#rand-m2-n3-mstd0.5-inc1
        self.rand_mirror = rand_mirror
        self.random_resizecrop=random_resizecrop
        self.rand_au=rand_au
        self.dino_trans=dino_trans
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
        if path_imgrec:
            logging.info('loading recordio %s...',
                         path_imgrec)
            path_imgidx = path_imgrec[0:-4] + ".idx"
            print(path_imgrec, path_imgidx)
            self.imgrec = recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
            s = self.imgrec.read_idx(0)
            header, _ = recordio.unpack(s)
            # pdb.set_trace()
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
            # pdb.set_trace()
            self.path_imgrec=path_imgrec
            max_ids=205990
            self.id_index=np.zeros(max_ids)
            filepath_id_nidex='id_nidex.json'
            # pdb.set_trace()
            if os.path.exists(filepath_id_nidex):
                with open(filepath_id_nidex, 'r') as f:
                    # f=open(filepath_id_nidex,'r')
                    out = json.load(f)
                    # self.samples=out
                    # pdb.set_trace()
                    self.id_index=out['index']
            else:
                for index in range(len(self.seq)):
                    idx = self.seq[index]
                    s = self.imgrec.read_idx(idx)
                    # pdb.set_trace()
                    header, s = recordio.unpack(s)
                    label = header.label
                    if not isinstance(label, numbers.Number):
                        label = label[0]
                    label=np.int64(label)
                    print(label)
                    if self.id_index[label]==0:
                        self.id_index[label]=index
                    else:
                        continue
                # pdb.set_trace()
                with open(filepath_id_nidex, 'w') as f:
                    # f=open(filepath_id_nidex,'w')
                    # out = f(self, directory, *args, **kwargs)
                    # json.dump({'a': list(self.id_index)},f,cls=NumpyEncoder)
                    dic={}
                    dic['index']=self.id_index.tolist()
                    json.dump( dic,f)
            self.id_index=self.id_index[:np.int64(np.floor(0.75*max_ids))]
            
                
        
    def __getitem__(self, index):
        idx = self.seq[index]
        # pdb.set_trace()
        original_index=np.int64(self.id_index[index])

        # path, target = self.samples[original_index]
        s = self.imgrec.read_idx(original_index)
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
        if 'webface' in self.path_imgrec:
            _data=_data[::-1,:,:]
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
        # pdb.set_trace()
        # if self.dino_trans is not None:
        #DINO augmentation
        _data=np.transpose(_data,(1,2,0))
        _data = Image.fromarray(_data)
        img=self.dino_trans(_data)
        # _data=np.array(_data)
        # _data=np.transpose(_data,(2,0,1))
        # _data1=_data.copy()
        # img = torch.from_numpy(_data1)
        # # pdb.set_trace()
        # if self.random_resizecrop:
        #     # img=self.aa_transform(img)
        #     img=self.trans(img)
        #     # img=self.aa_transform(img)
        
        
        # img=img/255.0-0.5
        return img, label



    # def __getitem__(self, index):
    #     """
    #     Args:
    #         index (int): Index
    #     Returns:
    #         tuple: (sample, target) where target is class_index of the target class.
    #     """
    #     pdb.set_trace()
    #     original_index=self.id_index[index]

    #     path, target = self.samples[original_index]
    #     # sample = self.loader(path)
    #     sample=cv2.imread(path)
    #     sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
    #     # pdb.set_trace()
    #     # if 'WebFace' in self.root:
    #     #     # swap rgb to bgr since image is in rgb for webface
    #     #     sample = Image.fromarray(np.asarray(sample)[:,:,::-1])
    #     sample=np.asarray(sample)
    #     if self.rand_mirror:
    #         _rd = random.randint(0, 1)
    #         if _rd == 1:
    #             # _data = mx.ndarray.flip(data=_data, axis=1)
                
    #             sample=np.fliplr(sample)
    #     # sample, _ = self.augment(sample)
    #     # if self.transform is not None:
    #     #     sample = self.transform(sample)
    #     # if self.target_transform is not None:
    #     #     target = self.target_transform(target)
    #     sample=np.transpose(sample,(2,0,1))
    #     _data=sample
    #     if self.rand_au==True:
    #         # pdb.set_trace()
    #         _data=np.transpose(_data,(1,2,0))
    #         _data = Image.fromarray(_data)
    #         # _data.save("./img_check/{}_test_ori_webface_cv2.jpeg".format(index))
    #         _data=self.aa_transform(_data)
    #         _data=np.array(_data)
    #         # _data_pil=np.transpose(_data,(1,2,0))
    #         #visualize
    #         # _data_pil = Image.fromarray(_data)
    #         # _data_pil.save("./img_check/{}_test_trans.jpeg".format(index))
    #         # output original
    #         _data=np.transpose(_data,(2,0,1))
    #         #output trans
    #         # _data=np.transpose(_data_pil,(2,0,1))


    #     img = torch.from_numpy(_data)
    #     # pdb.set_trace()
    #     if self.random_resizecrop:
    #         # img=self.aa_transform(img)
    #         img=self.trans(img)
    #         # img=self.aa_transform(img)
    #     sample=img

    #     return sample, target

    def __len__(self):
        return len(self.id_index)

class FaceDataset_gen_5imgperid(data.Dataset):
    def __init__(self, path_imgrec,dino_trans=None, rand_mirror=False,random_resizecrop=False,img_size=112,rand_au=False,config_str='rand-m2-mstd0.5-inc1'):#rand-m2-n3-mstd0.5-inc1
        self.rand_mirror = rand_mirror
        self.random_resizecrop=random_resizecrop
        self.rand_au=rand_au
        self.dino_trans=dino_trans
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
        if path_imgrec:
            logging.info('loading recordio %s...',
                         path_imgrec)
            path_imgidx = path_imgrec[0:-4] + ".idx"
            print(path_imgrec, path_imgidx)
            self.imgrec = recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
            s = self.imgrec.read_idx(0)
            header, _ = recordio.unpack(s)
            # pdb.set_trace()
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
            # pdb.set_trace()
            self.path_imgrec=path_imgrec
            max_ids=205990
            self.id_index=np.zeros(max_ids,5)
            self.id_index_count=np.zeros(max_ids)
            filepath_id_nidex='id_nidex_5shot.json'
            # pdb.set_trace()
            if os.path.exists(filepath_id_nidex):
                with open(filepath_id_nidex, 'r') as f:
                    # f=open(filepath_id_nidex,'r')
                    out = json.load(f)
                    # self.samples=out
                    # pdb.set_trace()
                    self.id_index=out['index']
            else:
                for index in range(len(self.seq)):
                    idx = self.seq[index]
                    s = self.imgrec.read_idx(idx)
                    # pdb.set_trace()
                    header, s = recordio.unpack(s)
                    label = header.label
                    if not isinstance(label, numbers.Number):
                        label = label[0]
                    label=np.int64(label)
                    print(label)
                    if self.id_index_count[label]<5:
                        self.id_index[self.id_index_count[label]]=index
                        self.id_index_count[label]+=1
                        # self.id_index[label]+=index
                    else:
                        continue
                # pdb.set_trace()
                with open(filepath_id_nidex, 'w') as f:
                    # f=open(filepath_id_nidex,'w')
                    # out = f(self, directory, *args, **kwargs)
                    # json.dump({'a': list(self.id_index)},f,cls=NumpyEncoder)
                    dic={}
                    dic['index']=self.id_index.tolist()
                    json.dump( dic,f)
            
                
        
    def __getitem__(self, index):
        idx = self.seq[index]
        # pdb.set_trace()
        original_index=np.int64(self.id_index[index])

        # path, target = self.samples[original_index]
        s = self.imgrec.read_idx(original_index)
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
        if 'webface' in self.path_imgrec:
            _data=_data[::-1,:,:]
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
        # pdb.set_trace()
        # if self.dino_trans is not None:
        #DINO augmentation
        _data=np.transpose(_data,(1,2,0))
        _data = Image.fromarray(_data)
        img=self.dino_trans(_data)
        # _data=np.array(_data)
        # _data=np.transpose(_data,(2,0,1))
        # _data1=_data.copy()
        # img = torch.from_numpy(_data1)
        # # pdb.set_trace()
        # if self.random_resizecrop:
        #     # img=self.aa_transform(img)
        #     img=self.trans(img)
        #     # img=self.aa_transform(img)
        
        
        # img=img/255.0-0.5
        return img, label


    def __len__(self):
        return len(self.id_index)


if __name__ == '__main__':
    # root = '/raid/Data/faces_webface_112x112/train.rec'
    # embed()
    # dataset = FaceDataset(path_imgrec =root, rand_mirror = False)
    dataset = FaceDataset_gen_1imgperid(os.path.join('', 'train.rec'), rand_mirror=False,random_resizecrop=False,rand_au=False,config_str='rand-m1-mstd0.5-inc1')
    trainloader = data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0, drop_last=False)
    print(len(dataset))
    for data, label in trainloader:
        print(data.shape, label)