# Learning 3D Garment Animation from Trajectories of A Piece of Cloth
\[[Paper](https://openreview.net/pdf?id=yeFx5NQmr7)\]

This is the official repository of "Learning 3D Garment Animation from Trajectories of A Piece of Cloth, NeurIPS 2024".

**Authors**: [Yidi Shao](https://ftbabi.github.io/), [Chen Change Loy](https://www.mmlab-ntu.com/person/ccloy/),  and [Bo Dai](http://daibo.info/).

**Acknowedgement**: This study is supported under the RIE2020 Industry Alignment Fund Industry Collaboration Projects (IAF-ICP) Funding Initiative, as well as cash and in-kind contributions from the industry partner(s). It is also supported by Singapore MOE AcRF Tier 2 (MOE-T2EP20221-0011) and partially funded by the Shanghai Artificial Intelligence Laboratory.

**Feel free to ask questions. I am currently working on some other stuff but will try my best to reply. Please don't hesitate to star!** 

## News
- 13 Dec, 2024: EUNet core codes released
- 24 Nov, 2024: Codes released

## Table of Content
1. [Video Demos](#video-demos)
2. [Dataset](#dataset)
3. [Code](#code)
4. [Citations](#citations)

## Video Demos
![](imgs/demo.gif)

## Dataset (Coming Soon)
While the dataset is not one of the main contribution,
we will release part of the data including a piece of cloth.

## Code
<!-- Codes are tested on Ubuntu 18 and cuda 11.3. -->
We train our model with 1 V100.

### Installation
```
# CUDA11.7 TORCH1.13
conda create -n EUNet python=3.9 pytorch==1.13.0 pytorch-cuda=11.7 torchvision==0.14.0 torchaudio==0.13.0 -c pytorch -c nvidia -y
conda activate EUNet

<!-- mim install mmcv-full==1.7.1 -->
pip install mmcv-full==1.7.1 -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13.0/index.html

pip3 install h5py pyrender trimesh numpy==1.24.3 tqdm plotly scipy chumpy einops smplx yapf==0.40.1 tensorboard
pip install  dgl==1.1.0 -f https://data.dgl.ai/wheels/cu117/repo.html
pip install  dglgo -f https://data.dgl.ai/wheels-test/repo.html

pip3 install -v -e .

ln -s path/to/data data
ln -s path/to/work_dirs work_dirs

conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
conda install -c bottler nvidiacub -y
conda install pytorch3d -c pytorch3d -y
```

### Training (EUNet)
```
python tools/train.py configs/potential_energy/base.py --work_dir PATH/TO/DIR
```

### Inference (EUNet)
```
python tools/test.py configs/potential_energy/base.py PATH/TO/CHECKPOINT
```

## Citations
```
@inproceedings{shao2024eunet,
  author = {Shao, Yidi and Loy, Chen Change and Dai, Bo},
  title = {Learning 3D Garment Animation from Trajectories of A Piece of Cloth},
  booktitle = {NeurIPS},
  year = {2024}
}
```