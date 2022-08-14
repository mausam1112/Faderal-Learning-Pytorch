from torch.utils.data import Dataset
import albumentations as aug
from albumentations.pytorch import ToTensorV2
import cv2 
import os
from pandas.core.common import flatten

train_transforms = aug.Compose(
    [
        aug.RandomBrightnessContrast(p=0.5),
        aug.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        aug.RandomCrop(height=256, width=256),
        aug.SmallestMaxSize(max_size=350),
        ToTensorV2(),
    ]
)

test_transforms = aug.Compose(
    [
        aug.SmallestMaxSize(max_size=350),
        aug.CenterCrop(height=256, width=256),
        aug.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)


train_data_path = './dataset/dataset/train'
test_data_path = './dataset/dataset/test'

classes_train = os.listdir(train_data_path)
classes_test = os.listdir(train_data_path)

assert classes_train==classes_test



# train_images_path and test_images_path
def get_image_paths(data_path, classes_train):
    for class_ in classes_train:
        image_directory = os.path.join(data_path, class_)
        images_paths = [os.path.join(image_directory, image_name) for image_name in os.listdir(image_directory)]
        images_paths = list(flatten(images_paths))
        return images_paths

train_image_paths = get_image_paths(train_data_path, classes_train)
train_image_paths, valid_image_paths = train_image_paths[:int(0.8*len(train_image_paths))], train_image_paths[int(0.8*len(train_image_paths)):] 

test_images_paths = get_image_paths(test_data_path, classes_train)


#create index_to_classes and classes
idx_to_class = {idx:class_name for idx, class_name in enumerate(classes_train)}
class_to_idx = {class_name:idx for idx, class_name in idx_to_class.items()}

print(idx_to_class, class_to_idx)


class DriverDataset(Dataset):
    def __init__(self, image_paths, transform=False):
        self.image_paths = image_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        label = image_filepath.split('/')[-2]
        label = class_to_idx[label]
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        
        return image, label

train_dataset = DriverDataset(train_image_paths, train_transforms)
valid_dataset = DriverDataset(valid_image_paths, test_transforms)
test_dataset = DriverDataset(test_images_paths, test_transforms)

