🆕 [2026-03-25] :fire: MM-DINO has been published.

🆕 [2026-03-11] :fire: MM-DINO is now available.

# MM-DINO 🦖🦖🦖
[Yuan Qin](https://orcid.org/0009-0005-8953-9006)<sup>1,2</sup>,
Chanling Pan<sup>1,2</sup>,
Jinyun Chen<sup>1,2</sup>,
Ruibo Chen<sup>1,2</sup>,
Jiaxing Chen<sup>1,2</sup>,
and Ruichao Qu<sup>1,2</sup>. <br />
<sup>1</sup> Guangxi Zhuang Autonomous Region Institute of Natural Resources Remote Sensing
<sup>2</sup> Key Laboratory of China-ASEAN Satellite Remote Sensing Applications, Ministry of Natural Resources

[ :scroll: [`Paper`](https://ieeexplore.ieee.org/document/11456120)]

Reference PyTorch implementation and models for MM-DINO. For details, see the **[MM-DINO](https://ieeexplore.ieee.org/document/11456120)** paper.

## Overview


## Pretrained models

### DINOv3 pretrained backbones

:information_source: Please follow the link provided below to get access to all the model weights: once accepted, an e-mail will be sent with the complete list of URLs pointing to all the available model weights (both backbones and adapters). 

ViT models pretrained on web dataset (LVD-1689M):
<table style="margin: auto">
  <thead>
    <tr>
      <th>Model</th>
      <th>Parameters</th>
      <th>Pretraining<br/>Dataset</th>
      <th>Download</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ViT-S/16 distilled </td>
      <td align="right">21M</td>
      <td align="center">LVD-1689M</td>
      <td align="center"><a href="https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/">[link]</a></td>
    </tr>
    <tr>
      <td>ViT-S+/16 distilled</td>
      <td align="right">29M</td>
      <td align="center">LVD-1689M</td>
      <td align="center"><a href="https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/">[link]</a></td>
    </tr>
    <tr>
      <td>ViT-B/16 distilled</td>
      <td align="right">86M</td>
      <td align="center">LVD-1689M</td>
      <td align="center"><a href="https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/">[link]</a></td>
    </tr>
    <tr>
      <td>ViT-L/16 distilled</td>
      <td align="right">300M</td>
      <td align="center">LVD-1689M</td>
      <td align="center"><a href="https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/">[link]</a></td>
    </tr>
    <tr>
      <td>ViT-H+/16 distilled</td>
      <td align="right">840M</td>
      <td align="center">LVD-1689M</td>
      <td align="center"><a href="https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/">[link]</a></td>
    </tr>
    <tr>
      <td>ViT-7B/16</td>
      <td align="right">6,716M</td>
      <td align="center">LVD-1689M</td>
      <td align="center"><a href="https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/">[link]</a></td>
    </tr>
  </tbody>
</table>

ViT models pretrained on satellite dataset (SAT-493M):
<table style="margin: auto">
  <thead>
    <tr>
      <th>Model</th>
      <th>Parameters</th>
      <th>Pretraining<br/>Dataset</th>
      <th>Download</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ViT-L/16 distilled</td>
      <td align="right">300M</td>
      <td align="center">SAT-493M</td>
      <td align="center"><a href="https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/">[link]</a></td>
    </tr>
    <tr>
      <td>ViT-7B/16</td>
      <td align="right">6,716M</td>
      <td align="center">SAT-493M</td>
      <td align="center"><a href="https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/">[link]</a></td>
    </tr>
  </tbody>
</table>

### MM-DINO pretrained weights
<table style="margin: auto">
  <thead>
    <tr>
      <th>Model</th>
      <th>Modality</th>
      <th>mIoU</th>
      <th>Dataset</th>
      <th>Download</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>MM-DINO ViT-L/16</td>
      <td align="right">Uni</td>
      <td align="right">-</td>
      <td align="center">Vaihingen</td>
      <td align="center"><a href="">[Baidu Cloud]</a></td>
    </tr>
    <tr>
      <td>MM-DINO ViT-L/16</td>
      <td align="right">Multi</td>
      <td align="right">-</td>
      <td align="center">Vaihingen</td>
      <td align="center"><a href="">[Baidu Cloud]</a></td>
    </tr>
    <tr>
      <td>MM-DINO ViT-L/16</td>
      <td align="right">Uni</td>
      <td align="right">-</td>
      <td align="center">Potsdam</td>
      <td align="center"><a href="">[Baidu Cloud]</a></td>
    </tr>
    <tr>
      <td>MM-DINO ViT-L/16</td>
      <td align="right">Multi</td>
      <td align="right">-</td>
      <td align="center">Potsdam</td>
      <td align="center"><a href="">[Baidu Cloud]</a></td>
    </tr>
    <tr>
      <td>MM-DINO ViT-S/16</td>
      <td align="right">Uni</td>
      <td align="right">54.12%</td>
      <td align="center">WHU-OPT-SAR</td>
      <td align="center"><a href="https://pan.baidu.com/s/1QEpzjFZbb3rlDH9nWFeEuQ?pwd=msum">[Baidu Cloud]</a></td>
    </tr>
    <tr>
      <td>MM-DINO ViT-L/16</td>
      <td align="right">Uni</td>
      <td align="right">54.92%</td>
      <td align="center">WHU-OPT-SAR</td>
      <td align="center"><a href="https://pan.baidu.com/s/1QEpzjFZbb3rlDH9nWFeEuQ?pwd=msum">[Baidu Cloud]</a></td>
    </tr>
    <tr>
      <td>MM-DINO ViT-L/16</td>
      <td align="right">Uni</td>
      <td align="right">55.72%</td>
      <td align="center">WHU-OPT-SAR</td>
      <td align="center"><a href="https://pan.baidu.com/s/1QEpzjFZbb3rlDH9nWFeEuQ?pwd=msum">[Baidu Cloud]</a></td>
    </tr>
    <tr>
      <td>MM-DINO ViT-S/16</td>
      <td align="right">Multi</td>
      <td align="right">54.12%</td>
      <td align="center">WHU-OPT-SAR</td>
      <td align="center"><a href="https://pan.baidu.com/s/1QEpzjFZbb3rlDH9nWFeEuQ?pwd=msum">[Baidu Cloud]</a></td>
    </tr>
    <tr>
      <td>MM-DINO ViT-L/16</td>
      <td align="right">Multi</td>
      <td align="right">55.04%</td>
      <td align="center">WHU-OPT-SAR</td>
      <td align="center"><a href="https://pan.baidu.com/s/1QEpzjFZbb3rlDH9nWFeEuQ?pwd=msum">[Baidu Cloud]</a></td>
    </tr>
    <tr>
      <td>MM-DINO ViT-L/16 LoRA</td>
      <td align="right">Multi</td>
      <td align="right">55.92%</td>
      <td align="center">WHU-OPT-SAR</td>
      <td align="center"><a href="https://pan.baidu.com/s/1QEpzjFZbb3rlDH9nWFeEuQ?pwd=msum">[Baidu Cloud]</a></td>
    </tr>
  </tbody>
</table>

## Installation

The training and evaluation code requires PyTorch version >= 2.7.1 as well as a few other 3rd party packages. Note that the code has only been tested with the specified versions and also expects a Linux environment. To setup all the required dependencies for training and evaluation, please follow the instructions below:

*[micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html)* **(Recommended)** - Clone the repository and then create and activate a `dinov3` conda environment using the provided environment definition:

```shell
micromamba env create -f conda.yaml
micromamba activate dinov3
```

## Data preparation
You can modify the dataset paths in <MM-DINO>/tasks/segmentation/datasets/__init__.py.

### ISPRS Vaihingen, Potsdam

The Vaihingen root directory of the dataset should hold the following contents:

- optical image path: `<ROOT>/top/top_mosaic_09cm_area{}.tif`
- label path: `<ROOT>/gts_eroded_for_participants/top_mosaic_09cm_area1_noBoundary.tif`
- dsm path: `<ROOT>/dsm/dsm_09cm_matching_area1.tif`

You can download Vaihingen from here: [link](https://pan.baidu.com/s/16jtayj82a5PeIEFKxJ9xXg?pwd=gdh4)

The Potsdam root directory of the dataset should hold the following contents:

- optical image path: `<ROOT>/2_Ortho_RGB/top_potsdam_{}_RGB.tif`
- label path: `<ROOT>/5_Labels_for_participants_no_Boundary/top_potsdam_{}_label_noBoundary.tif`
- dsm path: `<ROOT>/1_DSM_normalisation/dsm_potsdam_{}_normalized_lastools.jpg`

You can download Potsdam from here: [link](https://pan.baidu.com/s/1Ypf4Fi_k7RaquHx6KuTVvg?pwd=c6dp)

### WHU-OPT-SAR

The WHU-OPT-SAR root directory of the dataset should hold the following contents:

- optical image path: `<ROOT>/optical/{}.tif`
- label path: `<ROOT>/lbl/{}.tif`
- sar path: `<ROOT>/sar/{}.tif`

You can download WHU-OPT-SAR from here: [link](https://github.com/AmberHen/WHU-OPT-SAR-dataset)

## Training
Before starting training, you need to download the DINOv3 backbone weights and place them in the corresponding path. You can modify the path in`<MM-DINO>/configs/MMDINO.py`.

### Fast setup: training MM-DINO ViT-L/16 on Vaihingen

```shell
torchrun --nproc_per_node=<GPUs NUM> ./tasks/segmentation/train_multi.py \
 --model-name DINOv3 \
 --num-modalities 1 \
 --dataset-name Vaihingen \
 --backbone-type dinov3_vitl16
```

## Evaluation

The training code regularly saves the weights. In order to evaluate the model, run the following evaluation on a single node:


### Evaluate on WHU-OPT-SAR dataset

```shell
torchrun --nproc_per_node=<GPUs NUM> ./tasks/segmentation/test.py \
 --model-name DINOv3 \
 --num-modalities 1 \
 --dataset-name WHU \
 --backbone-type dinov3_vits16 \
 --checkpoint-path <CHECKPOINT_PATH>
```

After the job is completed, the console will output the evaluation results. Additionally, you can find the visualization results in the specified output directory.
- `<MM-DINO>/vis_results` Contains visualization results and confusion matrix from the evaluation;

## Acknowledgment
Our implementation is mainly based on following repositories. Thanks for their good works.
* [DINOv3](https://github.com/facebookresearch/dinov3)

## Citing MM-DINO

If you find this repository useful, please consider giving a star :star: and citation :t-rex::

```
@ARTICLE{11456120,
  author={Qin, Yuan and Pan, Chanling and Chen, Jinyun and Chen, Ruibo and Chen, Jiaxing and Qu, Ruichao},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={MM-DINO: DINOv3-Based Universal Framework for Uni and Multimodal Remote Sensing Image Semantic Segmentation}, 
  year={2026},
  volume={64},
  number={4407012},
  pages={1-12},
  keywords={Remote sensing;Foundation models;Feature extraction;Adaptation models;Visualization;Decoding;Transformers;Training;Data models;Semantic segmentation;DINOv3;domain generalization;foundation models;multimodal fusion;parameter-efficient fine-tuning;remote sensing imagery;semantic segmentation},
  doi={10.1109/TGRS.2026.3677346}}
```
