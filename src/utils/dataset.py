import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np



class MultiViewImageDataset(Dataset):
    def __init__(self, 
                root_dir: str, 
                input_transform=None, 
                mask_transform=None,
                image_size=(224, 224)  # Add default image size
                ):
        self.root_dir = root_dir
        self.cam_dir = os.path.join(root_dir, 'cam')
        self.masked_dir = os.path.join(root_dir, 'masked')
        self.pc_dir = os.path.join(root_dir, 'pc')
        self.image_size = image_size

        self.image_names = sorted(os.listdir(self.cam_dir))

        if not self.image_names:
            raise ValueError(f"No images found in {self.cam_dir}")

        self.input_transform = input_transform or transforms.Compose([
            transforms.ToTensor(),
        ])
        self.mask_transform = mask_transform or transforms.Compose([
            transforms.ToTensor(),
        ])

    def update_image_names(self, new_image_names):
        """Update the dataset with new image names."""
        self.image_names = sorted(new_image_names)
        if not self.image_names:
            raise ValueError("No images found after updating image names.")

    def __len__(self):
        return len(self.image_names)

    def _load_image(self, path):
        """Load an image from the given path."""
        image = Image.open(path).convert('RGB')
        if self.image_size:
            image = image.resize(self.image_size, Image.Resampling.LANCZOS)
        return image
    
    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        
        cam_path = os.path.join(self.cam_dir, image_name)
        masked_path = os.path.join(self.masked_dir, image_name)
        point_cloud_path = os.path.join(self.pc_dir, image_name)
        
        cam_image = self._load_image(cam_path)
        masked_image = self._load_image(masked_path)
        point_cloud_image = self._load_image(point_cloud_path)

        return {
            'cam': self.input_transform(cam_image),
            'masked': self.mask_transform(masked_image),
            'point_cloud': self.input_transform(point_cloud_image),
            'name': image_name
        }

import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np



class JustCAM(MultiViewImageDataset):
    def __init__(self, 
                root_dir: str, 
                input_transform=None, 
                mask_transform=None,
                image_size=(224, 224)  # Add default image size
                ):
        super().__init__(root_dir, input_transform, mask_transform, image_size)
        self.cam_dir = os.path.join(root_dir, 'cam')
        self.masked_dir = os.path.join(root_dir, 'masked')
        self.image_names = sorted(os.listdir(self.cam_dir))


    def __getitem__(self, idx):
        dict = super().__getitem__(idx)
        return {
            'cam': dict['cam'],
            'masked': dict['masked'],
            'point_cloud': torch.zeros_like(dict['cam']),  # No point cloud for JustCAM
            'name': dict['name']
        }