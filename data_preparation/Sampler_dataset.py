from torchvision import transforms
from torch.utils.data import WeightedRandomSampler
from  torchvision import datasets
from torch.utils.data import random_split
from tqdm import tqdm 
from torch.utils.data import DataLoader
import json
import os 
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from files.utils import DATA_CONFIG
import json
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from tqdm import tqdm

class Weight_samplerDataset:
    def __init__(self, val_in=True, test=True, trans=True):
        self.train_size = DATA_CONFIG['data']['train_size']
        self.val_size = DATA_CONFIG['data']['val_size']
        self.val_in = DATA_CONFIG['train_val']['train_with_val']
        self.batch_size = DATA_CONFIG['data']['batch_size']
        
        if trans:
            normalize = transforms.Normalize(DATA_CONFIG['data']['train_transforms'][2]['Normalize']['mean'],
                                             DATA_CONFIG['data']['train_transforms'][2]['Normalize']['std'])
            resize = transforms.Resize(size=DATA_CONFIG['data']['train_transforms'][0]['Resize']['size'])
            make_tensor = transforms.ToTensor()
            self.train_transforms = transforms.Compose([make_tensor, normalize, resize])
            self.val_transforms = transforms.Compose([make_tensor, normalize, resize])
        
        self.test = test
    
    def create_test_data(self):
        if DATA_CONFIG['data']['test_data']['create']:
            test_dataset = datasets.ImageFolder(DATA_CONFIG['data']['test_path'], transform=self.val_transforms)
            return DataLoader(test_dataset, batch_size=DATA_CONFIG['data']['batch_size'], shuffle=False, drop_last=True)
    
    def data_loaders(self):
        try:
            train_dir_data = datasets.ImageFolder(DATA_CONFIG['data']['train_path'], transform=self.train_transforms)
        except Exception as e:
            print(f'Error while Loading the data: {e}')
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
        
        if DATA_CONFIG['data_ratio']['unbalanced']:
            class_counts = [0] * len(classes)
            try:
                for _, label in tqdm(train_dir_data):
                    class_counts[label] += 1
                weights = [1.0 / class_counts[label] for _, label in tqdm(train_dir_data)]
            except Exception as e:
                print(f'Error while calculating class weights: {e}')
                exit()
            
            class_weights_file = 'class_weights.json'
            
            if not os.path.isfile(class_weights_file):
                with open(class_weights_file, 'w') as f:
                    json.dump({}, f)
            
            with open(class_weights_file, 'r') as f:
                class_weights = json.load(f)
            
            wsampler = WeightedRandomSampler(weights, len(train_dir_data), replacement=True)
            
            print('Weight-sampling:')
        
        train_loader = DataLoader(train_dataset, batch_size=DATA_CONFIG['data']['batch_size'],
                                  shuffle=DATA_CONFIG['data']['shuffle'], drop_last=True, sampler=wsampler, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=DATA_CONFIG['data']['batch_size'],
                                shuffle=False, drop_last=False, num_workers=2)
        
        print(f'Dataset has {len(classes)} classes: {classes}')
        print(f'Train data: {len(train_dataset)}, Validation data: {len(val_dataset)}')
        
        return train_loader
