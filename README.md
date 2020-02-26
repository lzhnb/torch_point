# Pytorch Implementation of PointNet and PointNet++

*Authors: Zhihao Liang*

This repo is implementation for [PointNet](http://openaccess.thecvf.com/content_cvpr_2017/papers/Qi_PointNet_Deep_Learning_CVPR_2017_paper.pdf) and [PointNet++](http://papers.nips.cc/paper/7095-pointnet-deep-hierarchical-feature-learning-on-point-sets-in-a-metric-space.pdf) in pytorch.

> Note: This version fix cs230's [pattern](https://github.com/cs230-stanford/cs230-code-examples/tree/master/pytorch/vision) and [yanx27](https://github.com/yanx27/Pointnet_Pointnet2_pytorch#pytorch-implementation-of-pointnet-and-pointnet).

## Requirements
Python 3.6, torch 1.0.0 and other common packages listed in ```requirements.txt```.


## Installation
1. Clone this repository
```sh
git clone https://github.com/lzhnb/torch_point
cd torch_point
```
1. Create virtual env using Anaconda
```sh
conda create -n env python=3.6
conda activate env
```
3. Install dependencies
```sh
pip install -r requirements.txt
```
4. Download relative ```ModelNet40``` and ```ShapeNet``` Dataset
```sh
sh scripts/download.sh
```


## Task

Given a pointcloud representing different classes, predicts the correct label or semantic segmentation.


## Quickstart

- PointNet classification Task
```sh
python train.py
```
- PointNet segmentation Task  
  TODO
- PointNet++ classification Task  
  TODO
- PointNet++ segmentation Task  
  TODO

## Visualization

TODO

## TODO
- [x] PointNet Classification Task
- [x] Multi-GPU training
- [ ] PointNet Segmentation Task
- [ ] PointNet++ Architecture
- [ ] ```REAMDE.md``` completion
- [ ] Visualization completion
- [ ] Demo show

## Resources

- [PyTorch documentation](http://pytorch.org/docs/0.3.0/)
- [Tutorials](http://pytorch.org/tutorials/)
- [PyTorch warm-up](https://github.com/jcjohnson/pytorch-examples)
- [halimacc/pointnet3](https://github.com/halimacc/pointnet3)<br>
- [fxia22/pointnet.pytorch](https://github.com/fxia22/pointnet.pytorch)
- [yanx27/Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)


[SIGNS]: https://drive.google.com/file/d/1ufiR6hUKhXoAyiBNsySPkUwlvE_wfEHC/view?usp=sharing
