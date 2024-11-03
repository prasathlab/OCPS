from PIL import Image
import numpy as np
import pandas as pd 
import os
import torch
from torch.utils.data import Dataset



class feature_dataset(Dataset):
    def __init__(self, df_feature, df_label):
        self.feature = df_feature
        self.label = df_label

    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):

        feature = self.feature[self.feature['path'] == self.label.iloc[idx, 0]].iloc[:, 2:]
        label = self.label['label'].iloc[idx]

        # Convert feature DataFrame to tensor
        feature = torch.tensor(feature.values, dtype=torch.float32).squeeze()
        return feature, label







    
    






class cytology_dataset(Dataset):
    def __init__(self, img_dir, annotation_file, img_transform = None):
        self.img_dir = img_dir
        self.img_labels = pd.read_csv(annotation_file)
        self.img_transform = img_transform

    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])

        image = Image.open(img_path)
        if self.img_transform is not None:
            image = image.resize((224, 224))
        image = np.array(image)
        image = np.transpose(image, (2, 0, 1))
        image = image / 255.0
        

        label = self.img_labels.iloc[idx, -1]

        return image, label
    



class feature_extraction_dataset(Dataset):
    def __init__(self, img_dir, annotation_file, img_transform = None):
        self.img_dir = img_dir
        self.img_labels = pd.read_csv(annotation_file)
        self.img_transform = img_transform

    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])

        image = Image.open(img_path)
        if self.img_transform is not None:
            image = image.resize((224, 224))
        image = np.array(image)
        image = np.transpose(image, (2, 0, 1))
        image = image / 255.0
        

        label = self.img_labels.iloc[idx, -1]

        return image, img_path, label
    


