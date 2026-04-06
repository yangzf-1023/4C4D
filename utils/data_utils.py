import torch
from torch.utils.data import Dataset
import numpy as np
import cv2

class CameraDataset(Dataset):
    
    def __init__(self, viewpoint_stack, white_background):
        self.viewpoint_stack = viewpoint_stack
        self.bg = np.array([1.0, 1.0, 1.0], dtype=np.float32) if white_background else np.array([0.0, 0.0, 0.0], dtype=np.float32)
        
    def __getitem__(self, index):
        viewpoint_cam = self.viewpoint_stack[index]
        if viewpoint_cam.meta_only:
            viewpoint_image = self._load_and_process_image(viewpoint_cam)
        else:
            viewpoint_image = viewpoint_cam.image
            
        return viewpoint_image, viewpoint_cam
    
    def _load_and_process_image(self, viewpoint_cam):
        img = cv2.imread(viewpoint_cam.image_path, cv2.IMREAD_UNCHANGED)
        
        target_w, target_h = viewpoint_cam.resolution
        img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        
        img = img.astype(np.float32) / 255.0
        
        if img.shape[2] == 4:
            rgb = img[:, :, 2::-1]  # BGR -> RGB
            alpha = img[:, :, 3:4]
            blended = rgb * alpha + self.bg * (1.0 - alpha)
        else:
            blended = img[:, :, 2::-1]
        
        viewpoint_image = torch.from_numpy(blended.copy()).permute(2, 0, 1).contiguous().clamp(0.0, 1.0)
        
        return viewpoint_image
    
    def __len__(self):
        return len(self.viewpoint_stack)
    