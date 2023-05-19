from torchvision import transforms
from  torchvision import datasets
from torch.utils.data import random_split
from tqdm import tqdm 
from torch.utils.data import DataLoader
import json
import os 
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from files.utils import DATA_CONFIG
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

class Simple_Dataset:
    
    def __init__(self):
        self.train_size = DATA_CONFIG['data']['train_size']
        self.val_size = DATA_CONFIG['data']['val_size']
        self.val_in = DATA_CONFIG['train_val']['train_with_val']
        self.batch_size = DATA_CONFIG['data']['batch_size']
        
        normalize = transforms.Normalize(DATA_CONFIG['data']['train_transforms'][2]['Normalize']['mean'],
                                             DATA_CONFIG['data']['train_transforms'][2]['Normalize']['std'])
        resize = transforms.Resize(size=DATA_CONFIG['data']['train_transforms'][0]['Resize']['size'])
        make_tensor = transforms.ToTensor()
        self.train_transforms = transforms.Compose([make_tensor, normalize, resize])
        self.val_transforms = transforms.Compose([make_tensor, normalize, resize])
    
    def create_test_data(self):
        if DATA_CONFIG['data']['test_data']['create']:
            test_dataset = datasets.ImageFolder(DATA_CONFIG['data']['test_path'], transform=self.val_transforms)
            return DataLoader(test_dataset, batch_size=DATA_CONFIG['data']['batch_size'], shuffle=False, drop_last=True)
    
    def data_loaders(self):
        try:
            train_dir_data = datasets.ImageFolder(DATA_CONFIG['data']['train_path'], transform=self.train_transforms)
        except Exception as e:
            print(f'Error while Loading the data as: {e}')
            exit()
        
        classes = train_dir_data.classes
        if self.val_in:
            try:
                val_ratio = 0.25
                val_size = int(val_ratio * len(train_dir_data))
                train_size = len(train_dir_data) - val_size
                train_dataset, val_dataset = random_split(train_dir_data, [train_size, val_size])
            except Exception as e:
                print(f'Error while splitting dataset: {e}')
                exit()
        else:
            val_dataset = datasets.ImageFolder(DATA_CONFIG['data']['val_path'], transform=self.val_transforms)
        
        print('Creating data loaders:')
        
        train_loader = DataLoader(train_dataset, batch_size=DATA_CONFIG['data']['batch_size'],
                                  shuffle=DATA_CONFIG['data']['shuffle'], drop_last=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=DATA_CONFIG['data']['batch_size'],
                                shuffle=False, drop_last=False, num_workers=2)
        
        print(f'Dataset has {len(classes)} classes: {classes}')
        print(f'Train data: {len(train_dataset)}, Validation data: {len(val_dataset)}')
        
        return train_loader, val_loader, self.create_test_data()
