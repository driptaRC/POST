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

### Source-free UDA

- SURREAL -> LSP
'''
python train_human_prior.py /data/AmitRoyChowdhury/dripta/surreal_processed /data/AmitRoyChowdhury/dripta/lsp -s SURREAL -t LSP --target-train LSP_mt --log logs/s2l/ --seed 0  --lambda_c 1 --epochs 70 --pretrain-epoch 40 --rotation_stu 60 --shear_stu -30 30 --translate_stu 0.05 0.05 --scale_stu 0.6 1.3 --color_stu 0.25 --blur_stu 0 --rotation_tea 60 --shear_tea -30 30 --translate_tea 0.05 0.05 --scale_tea 0.6 1.3 --color_tea 0.25 --blur_tea 0 -b 32 --mask-ratio 0.5 --k 1 --occlude-rate 0.5 --occlude-thresh 0.9 --prior prior/SURREAL/K_5/checkpoints/l2/prior_stage_3.pt --fix-head --fix-upsample --source-free --lambda_b 1e-3 --lambda_p 1e-6 --step_p 47
'''

