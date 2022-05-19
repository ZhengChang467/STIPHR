# STIP (extended from our [previous work](https://arxiv.org/pdf/2203.16084.pdf) in CVPR2022)

Zheng Chang,
Xinfeng Zhang,
Shanshe Wang,
Siwei Ma,
Wen Gao.

Official PyTorch Code for **"STIP: A Spatiotemporal Information-Preserving and Perception-Augmented Model for High-Resolution Video Prediction"**

This work is extended from our previous work STRPM, which has been accepted by CVPR2022. The [codes](https://github.com/ZhengChang467/STRPM) for STRPM have also been made public.

### Requirements
- PyTorch 1.7.1
- CUDA 11.0
- CuDNN 8.0.5
- python 3.6.7

### Installation
Create conda environment:
```bash
    $ conda create -n STIP python=3.6.7
    $ conda activate STIP
    $ pip install -r requirements.txt
    $ conda install pytorch==1.7 torchvision cudatoolkit=11.0 -c pytorch
```
Download repository:
```bash
    $ git clone git@github.com:ZhengChang467/STIPHR.git
```

### Test on the ucfsports dataset
```bash
    $ python STIP_run.py --dataset ucfsport
```
### Test on the Human3.6M dataset
```bash
    $ python STIP_run.py --dataset human36m
```
### Test on the SJTU4K dataset
```bash
    $ python STIP_run.py --dataset sjtu4k
```

We plan to share the training soon!

### Citation
Please cite the following paper if you feel this repository useful.
```bibtex
@article{chang2022strpm,
title={STRPM: A Spatiotemporal Residual Predictive Model for High-Resolution Video Prediction},
author={Chang, Zheng and Zhang, Xinfeng and Wang, Shanshe and Ma, Siwei and Gao, Wen},
journal={arXiv preprint arXiv:2203.16084},
year={2022}
}
```
### License
See [MIT License](https://github.com/ZhengChang467/STIPHR/blob/master/LICENSE)

