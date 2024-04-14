import os
import cv2
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, random_split
import torch.optim as optim

from core.configs import (
    TRAIN_DATA_PATH, TEST_DATA_PATH, NUM_CLIENTS, BATCH_SIZE,
    EPOCHS
)
from core.dataset import ClassificationDataset
from core.model import ClassificationNetwork
from core.test_model import test
from core.train_model import train
from core.utils import get_target_classes, get_image_paths



def main():
    target_classes = get_target_classes(TRAIN_DATA_PATH, TEST_DATA_PATH)
    class_to_index = {
        class_name:index 
        for index, class_name in enumerate(target_classes)
    }
    
    train_image_paths = get_image_paths(TRAIN_DATA_PATH, target_classes)
    test_image_paths = get_image_paths(TEST_DATA_PATH, target_classes)

    # splitting train set into train and validation sets
    thresh = int(0.8*len(train_image_paths))
    train_img_paths = train_image_paths[:thresh]
    valid_img_paths = train_image_paths[thresh:] 

    # creating dataset for each sets
    train_dataset = ClassificationDataset(train_img_paths, class_to_index)
    validation_dataset = ClassificationDataset(valid_img_paths, class_to_index)
    test_dataset = ClassificationDataset(test_image_paths, class_to_index)

    # split for each client trainers
    records_per_split = int(len(train_dataset) / NUM_CLIENTS)
    train_dataset_split = random_split(
        train_dataset,
        [records_per_split] * NUM_CLIENTS
    )

    # creating datasloader
    train_loaders = [
        DataLoader(split, batch_size=BATCH_SIZE, shuffle=True)
        for split in train_dataset_split
    ]

    valid_loaders = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loaders = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    global_model = ClassificationNetwork()
    optimizers = [optim.SGD(global_model.parameters(), lr=0.0000001) for _ in range(NUM_CLIENTS)]

    global_model = train(EPOCHS, NUM_CLIENTS, global_model, train_loaders, valid_loaders, optimizers)

    test_loss, acc = test(global_model, test_loaders)

    print(f"Test loss={test_loss}, Test Accuracy={acc}")

if __name__ == "__main__":
    main()