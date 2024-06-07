from torch.utils.data import Dataset
from PIL import Image

import os 
import json
import torchvision.transforms.functional as VF
import torch

def get_data_loader(dataset, batch_size = 128, shuffle = False):

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size, shuffle=shuffle)
    
    return loader

class ImageNetSubset(Dataset):

    def __init__(self, root, classes = None):

        self.samples = []
        self.targets = []
        self.syn_to_class = {}
        
        with open(os.path.join(root, "imagenet_class_index.json"), "rb") as f:
            js = json.load(f)
            for class_id, (syn, _) in js.items():
                self.syn_to_class[syn] = int(class_id)

        with open(os.path.join(root, "ILSVRC2012_val_labels.json"), "rb") as f:
            val_syns = json.load(f)
        
        samples_dir = os.path.join(root, "ILSVRC", "Data", "CLS-LOC", "val")

        for filename in os.listdir(samples_dir):
            target = self.syn_to_class[val_syns[filename]]

            if classes == None or target in classes:
                self.targets += [target]
                self.samples += [os.path.join(samples_dir, filename)]
    
    def __len__(self):
        return len(self.samples)    
    
    def __getitem__(self, idx):
        
        x = Image.open(self.samples[idx]).convert("RGB")
        x = VF.center_crop(x, (224, 224))
        x = VF.to_tensor(x)
        x = VF.normalize(x, *self.mean_std())

        return x, self.targets[idx]
    
    def mean_std(self):
        return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    def prep_for_viz(self, x):
        
        mean, std = self.mean_std()
        
        mean = torch.tensor(mean).reshape((1, 3, 1, 1))
        std = torch.tensor(std).reshape((1, 3, 1, 1))

        x.cpu()

        x = x * std + mean
        x = x.permute((0,2,3,1))

        return x.detach().numpy()
            


    



