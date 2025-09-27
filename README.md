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
git clone https://github.com/Namn23/ISFM.git
cd ISFM

# 创建虚拟环境（可选）
python3 -m venv venv
source venv/bin/activate   # Linux / macOS
# 或者 Windows: .\venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

```
---

## 数据准备

数据目录：  
```bash
data/
├── train/
│ ├── vi/ # 可见光图像
│ └── ir/ # 红外图像
└── test/
├── vis/
└── ir/
```

---
## 训练/测试  
训练：
```bash
python train.py --config configs/isfm_config.yaml
```
测试：
```bash
python eval/compute_metrics.py --pred_dir outputs/fused --gt_dir data/test/gt --metrics psnr ssim mi entropy
```
