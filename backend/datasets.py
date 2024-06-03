from torch.utils.data import Dataset
from PIL import image
import os 
import json

class ImageNetSubset(Dataset):

    def __init__(self, root, classes = None):
        
        self.paths = []
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
                self.paths += [os.path.join(samples_dir, filename)]
    
    def __len__(self):
        return len(self.samples)    
    
    def __getitem__(self, idx):
        
        x = Image.open(self.samples[idx]).convert("RGB")
        
        if self.transform:
            x = self.transform(x)

        return x, self.targets[idx]
        



