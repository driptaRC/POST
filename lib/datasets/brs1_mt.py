import scipy.io as scio
import os, glob

from PIL import ImageFile
import torch
import numpy as np
from .keypoint_dataset import Body16KeypointDataset
from ..transforms.keypoint_detection import *
from .util import *
from ._util import download as download_data, check_exits


ImageFile.LOAD_TRUNCATED_IMAGES = True

class BRIAR_BRS1_mt(Body16KeypointDataset):
    """`BRIAR internal dataset`_

    Args:
        root (str): Root directory of dataset
        split (str, optional): PlaceHolder.
        task (str, optional): Placeholder.
        transforms (callable, optional): PlaceHolder.
        heatmap_size (tuple): (width, height) of the heatmap. Default: (64, 64)
        sigma (int): sigma parameter when generate the heatmap. Default: 2
    """
    def __init__(self, root, tasks=['100m','200m','400m','500m','uav','close_range'], image_size=(256, 256), k=1, 
                    transforms_base=None, transforms_stu=None, transforms_tea=None, **kwargs):
        
        available_tasks = ['100m','200m','400m','500m','uav','close_range']
        for t in tasks:
            assert t in available_tasks
        self.tasks = tasks
        self.transforms_base = Compose([ResizePad(image_size[0])]) + transforms_base
        self.transforms_stu = transforms_stu
        self.transforms_tea = transforms_tea
        self.k = k  

        samples = []
        for subject in os.listdir(root):
            if subject[0] != 'G':
                continue
            ranges = [x for x in os.listdir(os.path.join(root,subject,'field'))]

            for r in ranges: 
                if r not in self.tasks:
                    continue       
                frames_folder = glob.glob(os.path.join(root,subject,'field',r,'wb/*/frames'))
                frames_folder = frames_folder[0]
                all_frames = [os.path.join(frames_folder,x) for x in os.listdir(frames_folder)]
                all_frames.sort()
                
                bbox_file = glob.glob(os.path.join(root,subject,'field',r,'wb/*/det/*.txt'))[0]
                with open(bbox_file) as file:
                    bboxes = file.readlines()

                frame_indices = [int(bboxes[j].split(',')[0])-1 for j in range(len(bboxes))]

                for idx in range(len(frame_indices)):
                    if idx%5==0:  
                        i = frame_indices[idx]    
                        bbox = bboxes[idx].split(',') 
                        bbox = [float(d) for d in bbox[2:6]]
                        bbox = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]
                        samples.append((all_frames[i],bbox))     

        self.joints_index = (0, 1, 2, 3, 4, 5, 13, 13, 12, 13, 6, 7, 8, 9, 10, 11)
        self.visible = np.array([1.] * 6 + [0, 0] + [1.] * 8, dtype=np.float32)

        super(BRIAR_BRS1_mt, self).__init__(root, samples, image_size=image_size, **kwargs)

    def __getitem__(self, index):
        sample = self.samples[index]
        image_name = sample[0]
        bbox = sample[1]
        image = Image.open(image_name)
        w, h = image.size 
        keypoint2d = np.random.uniform(size=(16,2))*np.array([w,h])
        left, upper, right, lower = scale_box(bbox, w, h, 1.5)
        image = F.crop(image, upper, left, lower-upper+1, right-left+1)
        image = np.flip(image, axis=-1) 
        image = Image.fromarray(image)
        image, data = self.transforms_base(image, keypoint2d=keypoint2d, intrinsic_matrix=None)
        keypoint2d = data['keypoint2d']

        image_stu, data_stu = self.transforms_stu(image, keypoint2d=keypoint2d, intrinsic_matrix=None)
        keypoint2d_stu = data_stu['keypoint2d']
        aug_param_stu = data_stu['aug_param']

        meta_stu = {
            'image': image_name,
            'keypoint2d_ori': keypoint2d,  
            'keypoint2d_stu': keypoint2d_stu,  # （NUM_KEYPOINTS x 2）
            'aug_param_stu': aug_param_stu,
        }
        
        images_tea, targets_tea, target_weights_tea, metas_tea = [], [], [], []
        for _ in range(self.k):
            image_tea, data_tea = self.transforms_tea(image, keypoint2d=keypoint2d, intrinsic_matrix=None)
            keypoint2d_tea = data_tea['keypoint2d']
            aug_param_tea = data_tea['aug_param']


            meta_tea = {
                'image': image_name,
                'keypoint2d_tea': keypoint2d_tea,  # （NUM_KEYPOINTS x 2）
                'aug_param_tea': aug_param_tea,
            }
            images_tea.append(image_tea)
            metas_tea.append(meta_tea)

        temp = np.zeros_like(image_stu)
        return image_stu, temp, temp, meta_stu, images_tea, temp, temp, metas_tea
        
        