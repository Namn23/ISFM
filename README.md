This is official Pytorch implementation of “Interactive Spatial-Frequency Fusion Mamba for Multi-Modal Image Fusion”
# ISFM  
**This is official Pytorch implementation of “Interactive Spatial-Frequency Fusion Mamba for Multi-Modal Image Fusion”**  

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)  
[![Python version](https://img.shields.io/badge/python-3.x-yellow.svg)]()  

> 本仓库是论文 **“Interactive Spatial-Frequency Fusion Mamba for Multi-Modal Image Fusion”** 的官方 PyTorch 实现。

---

## 目录  
- [项目简介](#项目简介)  
- [特点与贡献](#特点与贡献)  
- [安装与环境](#安装与环境)  
- [数据准备](#数据准备)  
- [训练](#训练)  
- [测试 / 推理](#测试--推理)  
- [代码结构说明](#代码结构说明)  
- [超参数说明](#超参数说明)  
- [实验结果](#实验结果)  
- [可视化 / 示例](#可视化--示例)  
- [引用](#引用)  
- [许可协议](#许可协议)  
- [致谢](#致谢)  

---

## 项目简介  
在多模态影像融合（如红外 + 可见光、医学影像融合等）任务中，如何在 **空间域** 与 **频率域** 之间建立有效交互，以同时保留细节与频率信息，是一个挑战。  

**ISFM** 提出了一种 **交互式空间-频率融合结构（Interactive Spatial-Frequency Fusion Mamba）**，在网络内部空间信息与频率信息可以互为辅助、协同增强，从而提升融合影像的视觉效果与下游性能。  

本仓库实现了该模型，并提供训练、测试、可视化等完整流程的代码。

---

## 特点与贡献  

- 同时建模 **空间域** 与 **频率域** 特征，并进行交互融合  
- 结构模块设计灵活，可扩展到不同深度与尺度  
- 在多个公开数据集上取得优秀性能  
- 代码模块化，易于复现与扩展  

---

## 安装与环境  

```bash
git clone https://github.com/Namn23/ISFM.git
cd ISFM

# 创建虚拟环境（可选）
python3 -m venv venv
source venv/bin/activate   # Linux / macOS
# 或者 Windows: .\venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

---

## 数据准备

数据目录：  

data/
├── train/
│ ├── vi/ # 可见光图像
│ └── ir/ # 红外图像
└── test/
├── vis/
└── ir/

---
## 训练/测试  
训练：
```bash
python train.py --config configs/isfm_config.yaml
测试：
python eval/compute_metrics.py --pred_dir outputs/fused --gt_dir data/test/gt --metrics psnr ssim mi entropy

