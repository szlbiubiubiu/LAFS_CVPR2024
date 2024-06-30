## **LAFS: Landmark-based Facial Self-supervised Learning for Face Recognition**

![LAFS](image/LAFS_img.jpg)
<!---
<p align="center">
    <img src="image/LAFS_aug_small_v3.pdf" alt="pdf" width="600"/>
</p> 
-->
<!--- 添加一下main图片，我没找到png版本的大图-->

This is the official PyTorch implementation of CVPR 2024 paper  ([LAFS: Landmark-based Facial Self-supervised Learning for Face Recognition](https://arxiv.org/abs/2403.08161)).

Our code is partly borrowed from DINO (https://github.com/facebookresearch/dino) and Insightface(https://github.com/deepinsight/insightface).

### Abstract
In this work we focus on learning facial representations that can be adapted to train effective face recognition models, particularly in the absence of labels. Firstly, compared with existing labelled face datasets, a vastly larger magnitude of unlabeled faces exists in the real world. We explore the learning strategy of these unlabeled facial images through self-supervised pretraining to transfer generalized face recognition performance. Moreover, motivated by one recent finding, that is, the face saliency area is critical for face recognition, in contrast to utilizing random cropped blocks of images for constructing augmentations in pretraining, we utilizcd e patches localized by extracted facial landmarks. This enables our method - namely Landmark-based Facial Self-supervised learning (LAFS), to learn key representation that is more critical for face recognition. We also incorporate two landmark-specific augmentations which introduce more diversity of landmark information to further regularize the learning. With learned landmark-based facial representations, we further adapt the representation for face recognition with regularization mitigating variations in landmark positions. Our method achieves significant improvement over the state-of-the-art on multiple face recognition benchmarks, especially on more challenging few-shot scenarios.

```bibtex
@InProceedings{Sun_2024_CVPR,
    author    = {Sun, Zhonglin and Feng, Chen and Patras, Ioannis and Tzimiropoulos, Georgios},
    title     = {LAFS: Landmark-based Facial Self-supervised Learning for Face Recognition},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024}
}
```
Please consider ***cite our paper and star the repo*** if you find this repo useful.



## To Do

- [x] LAFS Pretraining scripts
- [ ] DINO-face Pretraining scripts
- [ ] Checkpoints
- [ ] Finetuning scripts

Please stay tuned for more updates.
### Usage
1. Pytorch Version
```
torch==1.8.1+cu111;  torchvision==0.9.1+cu111
```
2. Dataset

- [x]MS1MV3    -- Please download from InsightFace(https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_)
- []WebFace4m

3. SSL Pretraining Command
```
python -m torch.distributed.launch --nproc_per_node=2 lafs_train.py
```
Note on 2A100 (40GB), the total pretraining training time would be around 2-3 days. 

### License
This project is licensed under the terms of the MIT license.
