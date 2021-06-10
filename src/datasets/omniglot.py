import os
from utils.data import ImageDataset

OMNIGLOT_FILES = {
    "train": "val_train_train",
    "train+val": "train",
    "val_train": "val_test_train",
    "val_test": "val_test_test",
    "test_train": "test_train",
    "test_test": "test_test",
}


def get_omniglot_dataset(
    split,
    image_size,
    train=True,
    all=False,
    data_dir="../omniglot",
):
    split = OMNIGLOT_FILES[split]
    if train:
        split = f"background_{split}"
    if all:
        split = f"{split}_all"
    split = f"{split}_{image_size}.npz"
    # fp = f"../omniglot/{split}"
    fp = os.path.join(data_dir, split)
    return ImageDataset(fp)