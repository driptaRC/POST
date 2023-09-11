import os 
import argparse 
import numpy as np

def main(args):
    save_dir = os.path.join(args.save_dir, 'processed')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    clean_data = np.load(args.clean_pose_file, allow_pickle=True)
    with open(os.path.join(save_dir, f'real.npy'), 'wb') as f:
        np.save(f, clean_data)

    noisy_data = np.load(args.noisy_pose_file, allow_pickle=True)
    x = noisy_data[()]['poses']
    y = noisy_data[()]['labels']

    idx_1 = y>=0.5
    idx_2 = np.logical_and(y>0.3,y<0.5)    

    noisy_poses_1 = x[idx_1]
    noisy_poses_2 = x[idx_2]
    noisy_labels_1 = y[idx_1]    
    noisy_labels_2 = y[idx_2]

    noisy_data_1 = {'poses': noisy_poses_1, 'labels': noisy_labels_1}
    noisy_data_2 = {'poses': noisy_poses_2, 'labels': noisy_labels_2}

    with open(os.path.join(save_dir, f'synth_1.npy'), 'wb') as f:
        np.save(f, noisy_data_1)
    with open(os.path.join(save_dir, f'synth_2.npy'), 'wb') as f:
        np.save(f, noisy_data_2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare noisy data for learning prior')

    parser.add_argument("--clean-pose-file", type=str, default='/data/AmitRoyChowdhury/dripta/prior_v1/Human36M/K_5/raw/clean_poses.npy',
                        help="file containing clean poses")
    parser.add_argument("--noisy-pose-file", type=str, default='/data/AmitRoyChowdhury/dripta/prior_v1/Human36M/K_5/raw/noisy_poses.npy',
                        help="file containing noisy poses")
    parser.add_argument("--save-dir", type=str, default='/data/AmitRoyChowdhury/dripta/prior_v1/Human36M/K_5/',
                        help="directory to save all poses")
    
    args = parser.parse_args()
    main(args)
