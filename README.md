# Interactive Spatial-Frequency Fusion Mamba for Multi-Modal Image Fusion 

<div align="center">

<!-- 这里把 2402.xxxxx 替换成你真实的 arXiv ID -->
<a href="[https://arxiv.org/abs/2402.xxxxx](https://arxiv.org/abs/2602.04405)">
  <img src="https://img.shields.io/badge/arXiv-2602.04405-b31b1b.svg" alt="arXiv">
</a>

<!-- 2. 技术栈 (Tech Stack: Python, PyTorch, etc.) -->
<a href="https://www.python.org/">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue" alt="Python">
</a>
<a href="https://pytorch.org/">
  <img src="https://img.shields.io/badge/PyTorch-2.00%2B-ee4c2c.svg" alt="PyTorch">
</a>

<!-- MIT License -->
<a href="https://opensource.org/licenses/MIT">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</a>

</div>

<div align="center">
  <img src="assets/teaser.png" width="800">
</div>

**ISFM** is a novel multi-modal image fusion framework designed to integrate complementary information from different modalities. Unlike traditional CNN- or Transformer-based methods that suffer from limited receptive fields or high computational cost, SFMFusion leverages Mamba to model long-range dependencies with linear complexity. Built upon this foundation, SFMFusion enhances Mamba with full spatial and frequency perceptions through the proposed Spatial-Frequency Enhanced Mamba Block, and efficiently couples fusion with image reconstruction via a three-branch structure. In addition, the Dynamic Fusion Mamba Block enables flexible feature aggregation across branches. Extensive experiments on six MMIF datasets demonstrate that SFMFusion achieves superior performance and provides a promising solution for multi-modal image fusion.

---

## 目录  
- [安装与环境](#安装与环境)  
- [数据准备](#数据准备)  
- [训练/测试](#训练)  


---

## 安装与环境  

```bash

# 创建虚拟环境
conda create -n ISFM python=3.9 -y
conda activate ISFM

# 安装依赖
pip install -r requirements.txt

```
---

## 数据准备

数据集下载链接：

| 数据集 | 下载链接 |
|:--------|:-----------|
| **MSRS** | [Download here](https://github.com/Linfeng-Tang/MSRS) | 
| **RoadScene** | [Download here](https://github.com/hanna-xu/RoadScene) | 
| **FMB** | [Download here](https://github.com/JinyuanLiu-CV/SegMiF) | 
| **Harvard** | [Download here](https://www.med.harvard.edu/AANLIB/home.html) | 


数据目录：  
```bash
data/
├── train/
│ ├── vi/ # 可见光图像
│ └── ir/ # 红外图像
└── test/
├── vi/
└── ir/
```

---
## 训练/测试  
训练：
```bash
python train.py --config configs/train.yaml
```
测试：
```bash
python test.py --config configs/test.yaml
```
