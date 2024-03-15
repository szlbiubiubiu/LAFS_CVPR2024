## **LAFS: Landmark-based Facial Self-supervised Learning for Face Recognition**

<!--- 添加一下main图片，我没找到png版本的大图
<p align="center">
    <img src="sources/caption.png" alt="png" width="600"/>
</p>
-->

This is the official PyTorch implementation of CVPR 2024 paper  ([LAFS: Landmark-based Facial Self-supervised Learning for Face Recognition](https://arxiv.org/abs/2403.08161)).

### Abstract
In this work we focus on learning facial representations that can be adapted to train effective face recognition models, particularly in the absence of labels. Firstly, compared with existing labelled face datasets, a vastly larger magnitude of unlabeled faces exists in the real world. We explore the learning strategy of these unlabeled facial images through self-supervised pretraining to transfer generalized face recognition performance. Moreover, motivated by one recent finding, that is, the face saliency area is critical for face recognition, in contrast to utilizing random cropped blocks of images for constructing augmentations in pretraining, we utilize patches localized by extracted facial landmarks. This enables our method - namely \textbf{LA}ndmark-based \textbf{F}acial \textbf{S}elf-supervised learning~(\textbf{LAFS}), to learn key representation that is more critical for face recognition. We also incorporate two landmark-specific augmentations which introduce more diversity of landmark information to further regularize the learning. With learned landmark-based facial representations, we further adapt the representation for face recognition with regularization mitigating variations in landmark positions. Our method achieves significant improvement over the state-of-the-art on multiple face recognition benchmarks, especially on more challenging few-shot scenarios.

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

### Usage
Please stay tuned for more updates regarding the camera-ready version and code.

### License
This project is licensed under the terms of the MIT license.
