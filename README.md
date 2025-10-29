# ISFM  
**Interactive Spatial-Frequency Fusion Mamba for Multi-Modal Image Fusion**  

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)  
[![Python version](https://img.shields.io/badge/python-3.x-yellow.svg)]()  

> 本仓库是论文 **“Interactive Spatial-Frequency Fusion Mamba for Multi-Modal Image Fusion”** 的官方 PyTorch 实现。

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

数据集下载链接

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
