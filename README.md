# Interactive Spatial-Frequency Fusion Mamba for Multi-Modal Image Fusion 

<div align="center">

<!-- 这里把 2402.xxxxx 替换成你真实的 arXiv ID -->
<a href="[https://arxiv.org/abs/2602.04405](https://arxiv.org/abs/2602.04405)">
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
  <a href="https://arxiv.org/abs/2602.04405">TIP 2025 Paper</a>
</div>


<div align="center">
  <img src="assets/ISFM.png" width="100%">
</div>

**ISFM** is a novel Mamba-based interactive spatial-frequency fusion framework designed for Multi-Modal Image Fusion (MMIF). It aims to fully exploit the complementarity of domain-specific characteristics by incorporating frequency information into the spatial fusion process and leveraging Mamba to capture long-range dependencies.
Specifically, we propose a Multi-scale Frequency Fusion to adaptively integrates low-frequency and high-frequency components of different modalities in multiple scales. 
To fully explore the complementarity of domain-specific characteristics, we propose an Interactive Spatial-Frequency Fusion including a Frequency-Guided Mamba and a Frequency-Guided Gate.
By combining these modules, our ISFM comprehensively integrates complementary information in the spatial and frequency domains. Extensive experiments on six MMIF datasets demonstrate that our method can achieve better performance than other state-of-the-art methods.


## News  
Exciting news! Our paper has been accepted by the TIP 2025! 🎉🎉 [Paper](https://arxiv.org/abs/2602.04405)


## Table of Contents
- [Introduction](#introduction)  
- [Contributions](#contributions)
- [Results](#results)
- [Visualizations](#visualizations)
- [Reproduction](#reproduction)
- [Citation](#citation)
  
## Introduction  
ISFM is a Mamba-based interactive spatial-frequency fusion framework for Multi-Modal Image Fusion (MMIF). This repository provides the training and testing code, along with pretrained weights for reproducing the results in our paper.

## Contributions  
- We introduce a novel Interactive Spatial-Frequency Fusion Mamba (ISFM) framework for MMIF. It provides a distinct perspective for spatial-frequency fusion.
- We propose a Multi-scale Frequency Fusion (MFF) to effectively fuse frequency information across multiple scales. In addition, we propose an Interactive Spatial Frequency Fusion (ISF) to fully exploit the complementarity of spatial-frequency information.
- Extensive experiments on IVIF and MIF tasks validate the effectiveness of our method. We also validate our method in helping high-level computer vision tasks.

## Results  
<div align="center">
  <img src="assets/IVIF_result.png" width="100%">
</div>
<div align="center">
  <img src="assets/MIF_result.png" width="100%">
</div>

## Visualizations  
<div align="center">
  <img src="assets/msrs.png" width="100%">
</div>
<div align="center">
  <img src="assets/fmb.png" width="100%">
</div>
<div align="center">
  <img src="assets/roadscene.png" width="100%">
</div>
<div align="center">
  <img src="assets/mripet.png" width="100%">
</div>
<div align="center">
  <img src="assets/mrict.png" width="100%">
</div>
<div align="center">
  <img src="assets/mrispect.png" width="100%">
</div>

## Reproduction  

### Installation 

```bash

# Create a virtual environment
conda create -n ISFM python=3.9 -y
conda activate ISFM

# Install dependencies
pip install -r requirements.txt

```

### Datasets

We use the following datasets. Please organize the files following the directory structure.

| Datasets | Download link |
|:--------|:-----------|
| **MSRS** | [Download here](https://github.com/Linfeng-Tang/MSRS) | 
| **RoadScene** | [Download here](https://github.com/hanna-xu/RoadScene) | 
| **FMB** | [Download here](https://github.com/JinyuanLiu-CV/SegMiF) | 
| **Harvard** | [Download here](https://www.med.harvard.edu/AANLIB/home.html) | 


Directory structure：  
```bash
data/
├── train/
│ ├── vi/ # Visible image
│ └── ir/ # Infrared image
└── test/
├── vi/
└── ir/
```

### Usage 
1)Train：
```bash
python train.py --config configs/train.yaml
```
2)Test：
```bash
python test.py --config configs/test.yaml
```

## Citation  
If you find ISFM useful in your research, please consider citing:
```bibtex
@article{zhu2026isfm,
      title={Interactive Spatial-Frequency Fusion Mamba for Multi-Modal Image Fusion}, 
      author={Zhu, Yixin and Lv, Long and Zhang, Pingping and Liu, Xuehu and Tang, Tongdan and Tian, Feng and Sun, Weibing and Lu, Huchuan},
      journal={arXiv preprint arXiv:2602.04405},
      year={2026},
}

