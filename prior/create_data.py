import os 
import argparse 
from tqdm import tqdm

import faiss
import faiss.contrib.torch_utils
import torch
import numpy as np

import sys
sys.path.append('..')
import lib.transforms.keypoint_detection as T
import lib.datasets as datasets

def faiss_idx_torch(poses):
    all_poses = np.copy(poses)
    all_poses = all_poses.reshape(len(poses), 26)
    data_all = torch.from_numpy(all_poses).to(device='cuda')
    res = faiss.StandardGpuResources()
    index = faiss.GpuIndexFlatL2(res, 26)
    index.train(data_all)
    index.add(data_all)
    return index, data_all

def dist_calc(quer_pose, nn_poses, k_dist, geodesic=False):
    if not geodesic:
        dist = quer_pose - nn_poses
        dis = torch.mean(torch.sqrt(torch.sum(dist * dist, dim=2)), dim=1)
    else:
        dis = torch.sum(torch.arccos(torch.sum(quer_pose*nn_poses,dim=2))**2,dim=1)/2.0
    val, idx = torch.topk(dis, k=k_dist, largest=False)
    return val, idx

def perturb_human_pose(keypoints, dispersion=8):
    pairs = [
        [8,9],   #0,-1
        [2,8],   #1,0
        [3,8],   #2,0
        [12,8],  #3,0
        [13,8],  #4,0
        [1,2],   #5,1
        [0,1],   #6,5
        [4,3],   #7,2
        [5,4],   #8,7
        [11,12], #9,3
        [10,11], #10,9
        [14,13], #11,4
        [15,14]  #12,11
    ]
    K = len(pairs) 
    # get orientations
    orientations = np.zeros((K,2))
    for i, pair in enumerate(pairs):
        a,b = pair
        vec = keypoints[b] - keypoints[a]
        norm = np.linalg.norm(vec) + 1e-10
        orientations[i] = vec/norm
    # perturb the orientations
    noise = np.random.vonmises(mu=0,kappa=dispersion,size=K)
    theta_og_0 = np.arccos(orientations[:,0])
    theta_og_1 = np.arcsin(orientations[:,1])
    theta_new_0 = theta_og_0 + noise
    theta_new_1 = theta_og_1 + noise
    new_orientations = np.stack((np.cos(theta_new_0), np.sin(theta_new_1)),axis=1) 
    return orientations, new_orientations

def main(args):
    image_size = (args.image_size, args.image_size)
    heatmap_size = (args.heatmap_size, args.heatmap_size)
    normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    transform = T.Compose([
        T.ToTensor(),
        normalize
    ])
    pose_dataset = datasets.__dict__[args.dset]
    og_pose_dataset = pose_dataset(root=args.dset_root, transforms=transform,
                                          image_size=image_size, heatmap_size=heatmap_size)
    N = len(og_pose_dataset)
    
    noise_params = [0.5, 1.0, 2.0, 4.0, 8.0]

    save_dir = os.path.join(args.save_dir, args.dset, f'K_{args.k_dist}', 'raw')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    clean_poses = np.zeros((N,13,2),dtype=np.float32)    
    noisy_poses = np.zeros((N*len(noise_params),13,2),dtype=np.float32)
    clean_labels = np.zeros(len(clean_poses))
    noisy_labels = np.zeros(len(noisy_poses))

    print('Creating noisy poses ...')
    j = 0
    for i in tqdm(range(N)):
        _,_,_,meta = og_pose_dataset[i]
        keypoints = meta['keypoint2d']
        for n in noise_params: 
            og_pose, new_pose = perturb_human_pose(keypoints, n)
            new_pose = new_pose/np.linalg.norm(new_pose, axis=1)[:,np.newaxis]
            noisy_poses[j] = new_pose
            j += 1        
        clean_poses[i] = og_pose  

    faiss_model, data_all = faiss_idx_torch(clean_poses)
    k_faiss = args.k_faiss
    k_dist = args.k_dist
    print('Labeling noisy poses ...')
    for i in tqdm(range(len(noisy_poses))):
        quer_pose = torch.from_numpy(noisy_poses[i]).cuda()
        inp_pose = quer_pose.reshape(1,26)
        # approx. KNN
        _, neighbors = faiss_model.search(inp_pose, k_faiss)
        nn_poses = data_all[neighbors].reshape(k_faiss, 13, 2)
        # KNN
        dist, _ = dist_calc(quer_pose, nn_poses, k_dist)
        noisy_labels[i] = torch.mean(dist).item()    

    clean_data = {'poses': clean_poses, 'labels': clean_labels}
    noisy_data = {'poses': noisy_poses, 'labels': noisy_labels}

    with open(os.path.join(save_dir, f'noisy_poses.npy'), 'wb') as f:
        np.save(f, noisy_data)
    with open(os.path.join(save_dir, f'clean_poses.npy'), 'wb') as f: 
        np.save(f, clean_data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create noisy data for learning prior')
    parser.add_argument('--dset', type=str, default='SURREAL',
                        help='prior pose dataset')
    parser.add_argument('--dset-root', type=str, default='/data/AmitRoyChowdhury/dripta/surreal_processed',
                        help='root path of the source dataset')
    parser.add_argument('--image-size', type=int, default=256,
                        help='input image size')
    parser.add_argument('--heatmap-size', type=int, default=64,
                        help='output heatmap size')
    parser.add_argument("--k-dist", type=int, default=5, 
                        help="K nearest neighbour")
    parser.add_argument("--k-faiss", type=int, default=500, 
                        help="K nearest neighbour FAISS")
    parser.add_argument("--save-dir", type=str, default='/data/AmitRoyChowdhury/dripta/prior_v1/',
                        help="where to save poses")
    args = parser.parse_args()
    main(args)
