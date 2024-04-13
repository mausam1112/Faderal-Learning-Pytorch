import os 
import cv2
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split


# train_images_path and test_images_path
def get_image_paths(data_path, classes_train):
    images_paths = []
    for class_ in classes_train:
        image_directory = os.path.join(data_path, class_)
        images_paths = [os.path.join(image_directory, image_name) for image_name in os.listdir(image_directory)]
    return images_paths


class ClassificationDataset(Dataset):
    def __init__(self, image_paths, class_to_index, transform=None) -> None:
        self.image_paths = image_paths
        self.transform = transform
        self.class_to_index = class_to_index

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        image_filepath = self.image_paths[index]
        image = cv2.imread(image_filepath)
        image = np.array(image, np.float32)

        image = torch.from_numpy(image).permute(2, 0, 1)

        label = Path(image_filepath).parts[-2]
        label = self.class_to_index[label]

        return image, label

