import os

def are_classes_same(train_classes: list|tuple, test_classes:list|tuple) -> bool:
    return train_classes == test_classes

def get_target_classes(train_data_path, test_data_path) -> list:
    "Returns classification classes if sub-directories in train and test directories matches"
    classes_train = os.listdir(train_data_path)
    classes_test = os.listdir(test_data_path)
    if are_classes_same(classes_train, classes_test):
        return classes_train
    else:
        raise Exception("Train and test directories must have same target classes.")

def path_exists(paths):
    return all(
        list(
            map(
                lambda filepath: os.path.exists(filepath), 
                paths
            )
        )
    )

# train_images_path and test_images_path
def get_image_paths(data_path, classes_train) -> list:
    images_paths = []
    for class_ in classes_train:
        image_directory = os.path.join(data_path, class_)
        images_paths = [os.path.join(image_directory, image_name) for image_name in os.listdir(image_directory)]
    if not path_exists(images_paths):
        raise Exception("Path to file doesn't exist")
    return images_paths
