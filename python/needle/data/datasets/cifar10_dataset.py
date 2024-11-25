import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
import numpy as np
from ..data_basic import Dataset

class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        ### BEGIN YOUR SOLUTION
        self.transforms = transforms

        if train:
            batch_files = [f"data_batch_{i}" for i in range(1, 6)]
        else:
            batch_files = ["test_batch"]

        images = []
        labels = []

        for file in batch_files:
            file_path = os.path.join(base_folder, file)
            with open(file_path, "rb") as f:
                batch = pickle.load(f, encoding="bytes")
                images.append(batch[b"data"])
                labels.append(batch[b"labels"])

        self.X = np.concatenate(images).reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
        self.y = np.concatenate(labels).astype(np.int64)
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        ### BEGIN YOUR SOLUTION
        if self.transforms:
            image = np.array([self.apply_transforms(img) for img in self.X[index]])
        else:
            image = self.X[index]
        label = self.y[index]
        return image, label
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        res = len(self.y)
        return res
        ### END YOUR SOLUTION
