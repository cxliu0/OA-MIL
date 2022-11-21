# OA-MIL

This repository includes the official implementation of the paper:

**[Robust Object Detection With Inaccurate Bounding Boxes](https://arxiv.org/abs/2207.09697)**

European Conference on Computer Vision (ECCV), 2022

Chengxin Liu<sup>1</sup>, Kewei Wang<sup>1</sup>, Hao Lu<sup>1</sup>, Zhiguo Cao<sup>1</sup>, and Ziming Zhang<sup>2</sup>

<sup>1</sup>Huazhong University of Science and Technology, China

<sup>2</sup>Worcester Polytechnic Institute, USA

[Paper](https://arxiv.org/abs/2207.09697) | [Supplementary](https://drive.google.com/file/d/15dcrAn7iUsAThZyZKb_DnnXuChN2NIVc/view?usp=sharing)

## Highlights

- **Robust**: OA-MIL is robust to inaccuate box annotations, and also effective on clean data;
- **Generic**: Our formulation is general and applicable to both one-stage and two-stage detectors;
- **No extra parameters**: OA-MIL does not introduce extra model parameters.

## Installation

[![Python](https://img.shields.io/badge/python-3.7%20tested-brightgreen)](https://www.python.org/)
[![Pytorch](https://img.shields.io/badge/pytorch-1.4.0%2F1.10.0%20tested-brightgreen)](https://pytorch.org/)

- Set up environment

```
# env
conda create -n oamil python=3.7
conda activate oamil

# install pytorch
conda install pytorch==1.10.0 torchvision==0.11.0 -c pytorch -c conda-forge
```

- Install 

```
# clone 
git clone https://github.com/cxliu0/OA-MIL.git
cd OA-MIL

# install dependecies
pip install -r requirements/build.txt

# install mmcv (will take a while to process)
cd mmcv
MMCV_WITH_OPS=1 pip install -e . 

# install OA-MIL
cd ..
pip install -e .
```

## Data Preparation

- Download [VOC2007](https://pjreddie.com/projects/pascal-voc-dataset-mirror/) and [COCO](https://cocodataset.org/#download) datasets. We expect the direcory structure to be as follow:

```
OA-MIL
├── data
│    ├── VOCdevkit
│    │    ├── VOC2007
│    │        ├── Annotations
│    │        ├── ImageSets
│    │        ├── JPEGImages
│    ├── coco
│        ├── train2017
│        ├── val2017
│        ├── annotations
│            ├── instances_train2017.json
│            ├── instances_val2017.json
├── configs
├── mmcv
├── ...
```

- Generate noisy annotations:

```
# generate noisy VOC2007 (e.g., 40% noise)
python ./utils/gen_noisy_voc.py --box_noise_level 0.4

# generate noisy COCO (e.g., 40% noise)
python ./utils/gen_noisy_coco.py --box_noise_level 0.4
```

- Alternatively, the noisy annotation files (coco dataset) we used are available at [google drive](https://drive.google.com/file/d/1Ujr6sJDcXjWyxlGhuk6B8vmJj6FqA4z3/view?usp=sharing).

## Training

All models of OA-MIL are trained with a total batch size of 16.

- To train OA-MIL on VOC2007, run

```
sh train_voc07.sh
```

Please refer to [faster_rcnn_r50_fpn_voc_oamil.py](configs/_base_/models/faster_rcnn_r50_fpn_voc_oamil.py) for model configuration

- To train OA-MIL on COCO, run

```
sh train_coco.sh
```

Please refer to [faster_rcnn_r50_fpn_coco_oamil.py](configs/_base_/models/faster_rcnn_r50_fpn_coco_oamil.py) for model configuration

## Inference

- Modify [test.sh](test.sh)

```/path/to/model_config```: modify it to the path of model config, e.g., ```./configs/faster_rcnn/faster_rcnn_r50_fpn_1x_voc_oamil.py```

```/path/to/model_checkpoint```: modify it to the path of model checkpoint


- Run
```
sh test.sh
```

## FAQ

- #### Is OA-MIL applicable to clean data?

  Yes, OA-MIL is applicable to clean data. Here we show some results on the clean VOC2007 and COCO datasets:
  
  - VOC2007
  
  |       Method       |       mAP@0.5      |
  | :----------------: | :----------------: |
  |  FasterRCNN        |        77.2        |
  |  OA-MIL FasterRCNN |      **78.6**      |

  - COCO
  
  |       Method       |     AP    |     AP50    |     AP75    | 
  | :----------------: | :-------: | :---------: | :---------: |
  | FasterRCNN         |     37.9  |     58.1    |     40.9    |
  | OA-MIL FasterRCNN  |  **38.1** |   **58.1**  |   **41.4**  |
  
- #### Where are the noisy annotation files the paper used?

  - The noisy annotation files of coco dataset is available at [google drive](https://drive.google.com/file/d/1Ujr6sJDcXjWyxlGhuk6B8vmJj6FqA4z3/view?usp=sharing);
  - For GWHD dataset, please refer to [this issue](https://github.com/cxliu0/OA-MIL/issues/5#issuecomment-1260347950).


## Citation

If you find this work or code useful for your research, please cite:
```
@inproceedings{liu2022oamil,
  title={Robust Object Detection With Inaccurate Bounding Boxes},
  author={Liu, Chengxin and Wang, Kewei and Lu, Hao and Cao, Zhiguo and Zhang, Ziming},
  booktitle={Proceeding of European Conference on Computer Vision (ECCV)},
  year={2022}
}
```

## Acknowlegdement

This repository is based on [mmdetection](https://github.com/open-mmlab/mmdetection).
