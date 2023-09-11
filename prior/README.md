# Train human prior

### Overview
This folder contains code associated with creating training data (real and noisy poses) and training the prior model. 

### Training
- Create poses and corresponding distance value
```
python create_data.py --dset SURREAL --dset-root <DATASET_PATH> --image-size 256 --heatmap-size 64 --k-dist 5  --k-faiss 500 --save-dir <SAVE_DIRECTORY>
```
- Format the created data
```
python prepare_data.py --clean-pose-file <DATASET_PATH/clean_poses.npy>  --noisy-pose-file <DATASET_PATH/noisy_poses.npy> --save-dir <SAVE_DIRECTORY>
```
- Train prior
```
python prior.py --data-root <DATASET_PATH> --out-dir <MODEL_SAVE_DIRECTORY>
