import os
import numpy as np
import torch
from scipy import io
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms

class PersonDataset(Dataset):
    def __init__(self, input_path, target_path):
        super().__init__()
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(30),
        ])
        
        input_img = io.loadmat(input_path)['img_3d'].astype('float32')
        target_img = io.loadmat(target_path)['img_3d'].astype('float32')
        
        assert input_img.shape == target_img.shape
        
        # [C, H, W]
        input_img = torch.from_numpy(input_img).cpu().float().permute(2, 0, 1)
        target_img = torch.from_numpy(target_img).cpu().float().permute(2, 0, 1)
        
        self.input_img = input_img
        self.target_img = target_img
        
    def __len__(self):
        return len(self.input_img)
    
    def __getitem__(self, idx):
        input_img = self.input_img[idx, :, :].unsqueeze(0)
        target_img = self.target_img[idx, :, :].unsqueeze(0)

        if self.transform:  # 应用数据增强
            seed = torch.randint(0, 2**32, (1,)).item()
            torch.manual_seed(seed)
            input_img = self.transform(input_img)
            torch.manual_seed(seed)  # 保证input和target应用相同的变换
            target_img = self.transform(target_img)
            
        return input_img, target_img, torch.clone(self.input_img)

class TrainDataset():
    def __init__(self, input, target, batch_size):
        input_img = np.array([input +"/"+ x  for x in os.listdir(input)])
        target_img = np.array([target +"/"+ x  for x in os.listdir(target)])
        
        assert len(input_img) == len(target_img)
        
        input_img.sort()
        target_img.sort()

        self.batch_size = batch_size
        self.data = []
        for i in range(0, len(input_img)):
            dataset = PersonDataset(input_img[i], target_img[i])
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            self.data.append(dataloader)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        dataloader = self.data[idx]
        
        return dataloader
    
class TrainDataset2(Dataset):
    def __init__(self, input, target):
        super().__init__()
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(30),
        ])
        
        input_img = np.array([input +"/"+ x  for x in os.listdir(input)])
        target_img = np.array([target +"/"+ x  for x in os.listdir(target)])
        
        assert len(input_img) == len(target_img)
        
        input_img.sort()
        target_img.sort()
        
        self.input_data: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.target_data: list[tuple[torch.Tensor, torch.Tensor]] = []
        for i in range(0, len(input_img)):
            input_path = input_img[i]
            target_path = target_img[i]
            input: np.ndarray = io.loadmat(input_path)['img_3d'].astype('float32')
            target: np.ndarray = io.loadmat(target_path)['img_3d'].astype('float32')
            
            assert input.shape == target.shape
            
            input = torch.from_numpy(input).cpu().float().permute(2, 0, 1)
            target = torch.from_numpy(target).cpu().float().permute(2, 0, 1)
            
            for ii in range(0, input.size(0)):
                self.input_data.append([input[ii, :, :], input])
                self.target_data.append([target[ii, :, :], target])
    
    def __len__(self):
        return len(self.input_data)
    
    def __getitem__(self, idx):
        input_img, glob_input = self.input_data[idx]
        target_img, _ = self.target_data[idx]
        
        input_img = input_img.unsqueeze(0)
        target_img = target_img.unsqueeze(0)
        
        if self.transform:  # 应用数据增强
            seed = torch.randint(0, 2**32, (1,)).item()
            torch.manual_seed(seed)
            input_img = self.transform(input_img)
            torch.manual_seed(seed)
            target_img = self.transform(target_img)
            torch.manual_seed(seed)
            glob_input = self.transform(glob_input)
        
        return torch.clone(input_img) * 2 - 1, torch.clone(target_img) * 2 - 1, torch.clone(glob_input) * 2 - 1
    