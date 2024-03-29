# Benchmarking the robustness of Spatial-Temporal Models

This repositery contains the code for [**NeurIPs 
Benchmark and Dataset Track 2021 paper - Benchmarking the Robustness of Spatial-Temporal Models Against Corruptions**](https://arxiv.org/abs/2110.06513).

Python 2.7 and 3.7, Pytorch 1.7+, FFmpeg are required.

### Requirements

```buildoutcfg
pip3 install - requirements.txt
```
## Mini Kinetics-C
![image info](assets/mini_kinetics-c_samples.jpg)

Download original Kinetics400 from [link](https://deepmind.com/research/open-source/kinetics).

The Mini Kinetics-C contains half of the classes in Kinetics400. All the classes can be found in [mini-kinetics-200-classes.txt](data/mini-kinetics-200-classes.txt).

## Mini Kinetics-C Leaderboard

Corruption robustness of spatial-temporal models trained on clean Mini Kinetics and evaluated on Mini Kinetics-C.

| Approach | Reference | Backbone | Input Length| Sampling Method | Clean Accuracy |mPC | rPC |
| --- |--- |--- |--- |--- |--- |--- |--- |
|[TimeSformer](https://github.com/facebookresearch/TimeSformer)|[Gedas et al.](https://arxiv.org/pdf/2102.05095.pdf)| Transformer|32|Uniform|82.2|71.4|86.9
|[3D ResNet](https://github.com/kenshohara/3D-ResNets-PyTorch)| [K. Hara et al.](https://openaccess.thecvf.com/content_cvpr_2018/html/Hara_Can_Spatiotemporal_3D_CVPR_2018_paper.html)|ResNet-50|32|Uniform|73.0|59.2|81.1|
| [I3D](https://github.com/deepmind/kinetics-i3d)| [J. Carreira et al.](https://arxiv.org/abs/1705.07750)| InceptionV1|32|Uniform|70.5|57.7|81.8|
|[SlowFast 8x4](https://github.com/facebookresearch/SlowFast)| [C. Feichtenhofer at al.](https://arxiv.org/abs/1812.03982) |ResNet-50|32|Uniform|69.2|54.3|78.5|
|[3D ResNet](https://github.com/kenshohara/3D-ResNets-PyTorch)| [K. Hara et al.](https://openaccess.thecvf.com/content_cvpr_2018/html/Hara_Can_Spatiotemporal_3D_CVPR_2018_paper.html)|ResNet-18|32|Uniform|66.2|53.3|80.5|
|[TAM](https://github.com/IBM/bLVNet-TAM)| [Q.Fan et al.](https://arxiv.org/abs/1912.00869) |ResNet-50|32|Uniform|66.9|50.8|75.9|
|[X3D-M](https://github.com/facebookresearch/SlowFast)| [C. Feichtenhofer](https://arxiv.org/abs/2004.04730) |ResNet-50|32|Uniform|62.6|48.6|77.6


For fair comparison, it is recommended to submit the result of approach which follows the following settings:
Backbone of ResNet-50, Input Length of 32, Uniform Sampling at Clip Level. Any result on our benchmark can be submitted via pull request.

## Mini SSV2-C
![image info](assets/mini_ssv2-c_samples.jpg)

Download original Something-Something-V2 datset from [link](https://20bn.com/datasets/something-something).

The Mini SSV2-C contains half of the classes in Something-Something-V2. All the classes can be found in [mini-ssv2-87-classes.txt](data/mini-ssv2-87-classes.txt).

## Mini SSV2-C Leaderboard

Corruption robustness of spatial-temporal models trained on clean Mini SSV2 and evaluated on Mini SSV2-C.

| Approach | Reference | Backbone | Input Length| Sampling Method | Clean Accuracy |mPC | rPC |
| --- |--- |--- |--- |--- |--- |--- |--- |
|[TimeSformer](https://github.com/facebookresearch/TimeSformer)|[Gedas et al.](https://arxiv.org/pdf/2102.05095.pdf)| Transformer|16|Uniform|60.5 | 49.7 |82.1
| [I3D](https://github.com/deepmind/kinetics-i3d)| [J. Carreira et al.](https://arxiv.org/abs/1705.07750)| InceptionV1|32|Uniform|58.5|47.8|81.7|
|[3D ResNet](https://github.com/kenshohara/3D-ResNets-PyTorch)| [K. Hara et al.](https://openaccess.thecvf.com/content_cvpr_2018/html/Hara_Can_Spatiotemporal_3D_CVPR_2018_paper.html)|ResNet-50|32|Uniform|57.4|46.6|81.2|
|[TAM](https://github.com/IBM/bLVNet-TAM)| [Q.Fan et al.](https://arxiv.org/abs/1912.00869) |ResNet-50|32|Uniform|61.8|45.7|73.9|
|[3D ResNet](https://github.com/kenshohara/3D-ResNets-PyTorch)| [K. Hara et al.](https://openaccess.thecvf.com/content_cvpr_2018/html/Hara_Can_Spatiotemporal_3D_CVPR_2018_paper.html)|ResNet-18|32|Uniform|53.0|42.6|80.3|
|[X3D-M](https://github.com/facebookresearch/SlowFast)| [C. Feichtenhofer](https://arxiv.org/abs/2004.04730) |ResNet-50|32|Uniform|49.9|40.7|81.6|
|[SlowFast 8x4](https://github.com/facebookresearch/SlowFast)| [C. Feichtenhofer at al.](https://arxiv.org/abs/1812.03982) |ResNet-50|32|Uniform|48.7|38.4|78.8|



For fair comparison, it is recommended to submit the result of approach which follows the following settings:
Backbone of ResNet-50, Input Length of 32, Uniform Sampling at Clip Level. Any result on our benchmark can be submitted via pull request.

## Training and Evaluation

To help researchers reproduce the benchmark results provided in our leaderboard, we include a simple framework for training and evaluating the spatial-temporal models in the folder: benchmark_framework.

## Running the code

Assume the structure of data directories is the following:

```misc
~/
  datadir/
    mini_kinetics/
      train/
        .../ (directories of class names)
          ...(hdf5 file containing video frames)
    mini_kinetics-c/
      .../ (directories of corruption names)
        .../ (directories of severity level)
          .../ (directories of class names)
            ...(hdf5 file containing video frames)
```
Train I3D on the Mini Kinetics dataset with 4 GPUs and 16 CPU threads (for data loading). The input lenght is 32, the batch size is 32 and learning rate is 0.01.
```buildoutcfg
python3 train.py --threed_data --dataset mini_kinetics400 --frames_per_group 1 --groups 32 --logdir snapshots/ \
--lr 0.01 --backbone_net i3d -b 32 -j 16 --cuda 0,1,2,3

```
Test I3D on the Mini Kinetics-C dataset (pretrained model is loaded)
```buildoutcfg
python3 test_corruption.py --threed_data --dataset mini_kinetics400 --frames_per_group 1 --groups 32 --logdir snapshots/ \
--pretrained snapshots/mini_kinetics400-rgb-i3d_v2-ts-max-f32-cosine-bs32-e50-v1/model_best.pth.tar --backbone_net i3d -b 32 -j 16 -e --cuda 0,1,2,3

```
