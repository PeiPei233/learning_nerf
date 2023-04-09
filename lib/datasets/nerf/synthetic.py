import torch
import os
from lib.config import cfg
import json
import imageio
import cv2
import numpy as np

class Dataset(torch.utils.data.Dataset):
    def __init__(self, **kwargs):
        super().__init__()
        scene = cfg.scene
        cams = kwargs['cams']
        data_root = kwargs['data_root']
        self.split = kwargs['split']
        self.input_ratio = kwargs['input_ratio']
        self.N_rays = cfg.task_arg.N_rays
        
        with open(os.path.join(data_root, scene, f'transforms_{self.split}.json')) as f:
            json_info = json.load(f)
        
        imgs = []
        poses = []
        for frame in json_info['frames']:
            img = imageio.imread(os.path.join(data_root, scene, frame['file_path'] + '.png'))
            if self.input_ratio != 1.:
                img = cv2.resize(img, None, fx=self.input_ratio, fy=self.input_ratio, interpolation=cv2.INTER_AREA)
            imgs.append(img)
            poses.append(frame['transform_matrix'])
        
        imgs = (np.array(imgs) / 255.).astype(np.float32)
        poses = np.array(poses).astype(np.float32)

        imgs = imgs[cams[0]:cams[1]:cams[2]]
        poses = poses[cams[0]:cams[1]:cams[2]]
        
        if cfg.task_arg.white_bkgd:
            imgs = imgs[..., :3] * imgs[..., -1:] + (1 - imgs[..., -1:])

        self.imgs = torch.Tensor(imgs)
        self.poses = torch.Tensor(poses)

        H, W = imgs[0].shape[:2]
        self.focal = 0.5 * W / np.tan(0.5 * json_info['camera_angle_x'])
        

    def __getitem__(self, index):
        img = self.imgs[index]
        pose = self.poses[index]

        rays_o = pose[:3, 3]    # pose @ [0, 0, 0, 1]
        H, W = img.shape[:2]
        i, j = torch.meshgrid(torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H), indexing='ij')
        i = i.T
        j = j.T
        rays_d = torch.stack([(i - W * 0.5) / self.focal, -(j - H * 0.5) / self.focal, -torch.ones_like(i)], -1)
        rays_d = torch.sum(rays_d[..., None, :] * pose[:3, :3], -1)

        img = img.reshape(-1, 3)
        rays_d = rays_d.reshape(-1, 3)
        N_rays = self.N_rays
        if self.split == 'train':
            select_ids = np.random.choice(H * W, self.N_rays, replace=False)
            rays_o = rays_o.expand(self.N_rays, -1)
            rays_d = rays_d[select_ids]
            target = img[select_ids]
        else:
            rays_o = rays_o.expand(H * W, -1)
            target = img
            N_rays = H * W

        return {
            'rays_o': rays_o,
            'rays_d': rays_d,
            'rgb': target,
            'H': H,
            'W': W,
            'focal': self.focal,
            'N_rays': N_rays,
        }
        

    def __len__(self):
        return len(self.imgs)