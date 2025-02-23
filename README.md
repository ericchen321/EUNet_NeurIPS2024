# Learning 3D Garment Animation from Trajectories of A Piece of Cloth
\[[Paper](https://openreview.net/pdf?id=yeFx5NQmr7)\]

This is the official repository of "Learning 3D Garment Animation from Trajectories of A Piece of Cloth, NeurIPS 2024".

**Authors**: [Yidi Shao](https://ftbabi.github.io/), [Chen Change Loy](https://www.mmlab-ntu.com/person/ccloy/),  and [Bo Dai](http://daibo.info/).

**Acknowedgement**: This study is supported under the RIE2020 Industry Alignment Fund Industry Collaboration Projects (IAF-ICP) Funding Initiative, as well as cash and in-kind contributions from the industry partner(s). It is also supported by Singapore MOE AcRF Tier 2 (MOE-T2EP20221-0011) and partially funded by the Shanghai Artificial Intelligence Laboratory.

**Feel free to ask questions. I am currently working on some other stuff but will try my best to reply. Please don't hesitate to star!** 

## News
- 23 Feb, 2025: code, pretrained model, and dataset.
    - All code released, including EUNet training/inference, MeshGraphNet constrained by EUNet training/inference. Pretrained models of EUNet and MeshGraphNet are also released.
    - Release the dataset (a piece of cloth) to train EUNet. Please download [here](https://entuedu-my.sharepoint.com/:u:/g/personal/yidi001_e_ntu_edu_sg/ERQvC2Y76VJEvziL4qjjN9cBvfJA1v5-IOpxM28dYl0dHQ?e=GzpRRB).
- 13 Dec, 2024: EUNet core codes released.
- 24 Nov, 2024: Codes released.

## Table of Content
1. [Video Demos](#video-demos)
2. [Dataset](#dataset)
3. [Code](#code)
4. [Citations](#citations)

## Video Demos
![](imgs/demo.gif)

## Dataset
While the dataset is not one of the main contribution,
we still release part of the data including a piece of cloth.
Please download [here](https://entuedu-my.sharepoint.com/:u:/g/personal/yidi001_e_ntu_edu_sg/ERQvC2Y76VJEvziL4qjjN9cBvfJA1v5-IOpxM28dYl0dHQ?e=GzpRRB) and extract to ```data/meta_cloth```.
```
|-- data
    └── meta_cloth
        |-- mesh_484 (trajectories)
        |-- mesh
        |   |-- mesh_484.obj
        |   └── mesh_484.json
        |-- entry_train_meta.txt
        |-- entry_test_meta.txt
        └── train_val_test.json
```

## Code
We train our model with one V100.

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
Download the pretrained model [here](https://entuedu-my.sharepoint.com/:u:/g/personal/yidi001_e_ntu_edu_sg/EV6f95S023JBmOHlZPGzvU8B0bN-ujifCWi6xmQfc7A6bg?e=9FKAkV) and put to ```work_dirs/eunet/material/latest.pth```.

```
python tools/test.py configs/potential_energy/base.py work_dirs/eunet/material/latest.pth
```

### Training Garment Models with EUNet
1. We follow [HOOD](https://github.com/dolorousrtur/hood) to train the neural simulator constrained by our EUNet. Please follow [HOOD](https://github.com/dolorousrtur/hood) to prepare the data first, and put to ```data/hood_data```.
2. Make sure the [VTO dataset](https://github.com/isantesteban/vto-dataset) is at ```data/hood_data/vto_dataset```. The data structure is as follows:
```
|-- data
    └── hood_data
        |-- aux_data
        |   |-- datasplits
        |   |-- garment_meshes
        |   |-- smpl
        |   |-- garments_dict.pkl
        |   └── smpl_aux.pkl
        └── vto_dataset
```
3. Execute the command:
```
python tools/train.py configs/dynamic_simulator/base.py --work_dir PATH/TO/WORK/DIR
```

### Inference Garment Models with EUNet
1. If using pretrained model, please download from [here](https://entuedu-my.sharepoint.com/:u:/g/personal/yidi001_e_ntu_edu_sg/EV6f95S023JBmOHlZPGzvU8B0bN-ujifCWi6xmQfc7A6bg?e=9FKAkV) and put to ```work_dirs/eunet/dynamics/latest.pth```.
2. Execute the following commands, which will save garment meshes into ```PATH/TO/SAVE/DIR``` in the form of ".pkl".
```
python tools/test.py configs/dynamic_simulator/test.py work_dirs/eunet/dynamics/latest.pth --show-dir PATH/TO/SAVE/DIR --show-options rollout=SEQ_NUM  # SEQ_NUM is the number of target sequence, e.g. rollout=4430
```

### Inference on Cloth3D
1. We sample sequences from the training set of Cloth3D. Please download the training set [here](https://chalearnlap.cvc.uab.es/dataset/38/data/72/description/) and extract to ```data/cloth3d```.
2. Download the SMPL model from [here], rename them to "model_f.pkl" and "model_m.pkl", and move to ```data/smpl```. The final structure should be as follows:
```
|-- data
    └── cloth3d
        └── entry_test.txt (provided)
        └── train
            └── 00000 (sequences)
        └── smpl
            |-- model_f.pkl
            |-- model_m.pkl
            └── segm_per_v_overlap.pkl (provided)
```
3. To evaluate the pretrained model, please execute the following command, which will output the errors for each/all sequences.
```
python tools/test.py configs/dynamic_simulator/test_cloth3d.py work_dirs/eunet/dynamics/latest.pth
```

| $l2$ (mm)| Tshirt  | Top  | Jumpsuit  | Dress  | Overall | Collision (%) |
| -------- | ------- | ------- | ------- | ------- | ------- | ------- |
| MGN+EUNet| $51.68\pm19.86$| $37.65\pm12.64$|$62.13\pm22.02$| $103.28\pm67.49$| $69.26\pm47.31$| $0.47\pm0.60$|


### Visualization
To visualize the meshes, please execute:
```
python tools/visualization.py --work_dir PATH/TO/SAVE/DIR --seq SEQ_NUM --frame FRAME_NUM
```
An example usage is ```python tools/visualization.py --work_dir PATH/TO/SAVE/DIR --seq 4430 --frame 1```.

You can hide/visualize corresponding meshes by clicking the legend on the top-right of the web page.

## Citations
```
@inproceedings{shao2024eunet,
  author = {Shao, Yidi and Loy, Chen Change and Dai, Bo},
  title = {Learning 3D Garment Animation from Trajectories of A Piece of Cloth},
  booktitle = {NeurIPS},
  year = {2024}
}
```