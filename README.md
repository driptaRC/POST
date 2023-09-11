# POST

### Overview
This repository is a PyTorch implementation of the paper [Prior-guided Source-free Domain Adaptation for Human Pose Estimation](https://arxiv.org/abs/2308.13954) published at [ICCV 2023](https://iccv2023.thecvf.com/). 

Our code is based on the implementation of [UDA_PoseEstimation](https://github.com/VisionLearningGroup/UDA_PoseEstimation/tree/master) and [RegDA](https://github.com/thuml/Transfer-Learning-Library/tree/master/examples/domain_adaptation/keypoint_detection).

**Data Preparation**

As instructed by [UDA_PoseEstimation](https://github.com/VisionLearningGroup/UDA_PoseEstimation/tree/master), following datasets can be downloaded automatically:
- [Surreal Dataset](https://www.di.ens.fr/willow/research/surreal/data/)
- [LSP Dataset](http://sam.johnson.io/research/lsp.html)

You need to prepare the following datasets manually if you want to use them:
- [Human3.6M Dataset](http://vision.imar.ro/human3.6m/description.php)

### Train human pose prior
- See `prior` folder for instructions 



