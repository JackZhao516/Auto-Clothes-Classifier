"""
EECS 445 - Introduction to Machine Learning
Fall 2020 - Project 2
Fashion Dataset
    Class wrapper for interfacing with the dataset of fashion images
"""
import os
import numpy as np
import pandas as pd
import torch
from imageio import imread
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from utils import config
from scipy.linalg import eigh

def get_train_val_test_loaders(num_classes):
    tr, va, te, _ = get_train_val_test_dataset(num_classes=num_classes)

    batch_size = config('cnn.batch_size')
    tr_loader = DataLoader(tr, batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(va, batch_size=batch_size, shuffle=False)
    te_loader = DataLoader(te, batch_size=batch_size, shuffle=False)

    return tr_loader, va_loader, te_loader, tr.get_semantic_label

def get_train_val_test_dataset(num_classes=5):
    tr = FashionDataset('train', num_classes)
    va = FashionDataset('val', num_classes)
    te = FashionDataset('test', num_classes)

    # Resize
    tr.X = resize(tr.X)
    va.X = resize(va.X)
    te.X = resize(te.X)

    # Standardize
    standardizer = ImageStandardizer()
    standardizer.fit(tr.X)
    tr.X = standardizer.transform(tr.X)
    va.X = standardizer.transform(va.X)
    te.X = standardizer.transform(te.X)

    # Transpose the dimensions from (N,H,W,C) to (N,C,H,W)
    tr.X = tr.X.transpose(0,3,1,2)
    va.X = va.X.transpose(0,3,1,2)
    te.X = te.X.transpose(0,3,1,2)
    #
    return tr, va, te, standardizer
    # print(tr.X.shape)
    # return tr.X

def get_unlabeled_loader(num_classes):
    unl = get_unlabeled_dataset(num_classes=num_classes)

    batch_size = config('cnn.batch_size')
    unl_loader = DataLoader(unl, batch_size=batch_size, shuffle=False)

    return unl_loader

def get_unlabeled_dataset(num_classes=5):
    tr = FashionDataset('train', num_classes)
    unl = FashionDataset('unlabeled', num_classes)
    unl.X = resize(unl.X)

    standardizer = ImageStandardizer()
    standardizer.fit(tr.X)
    unl.X = standardizer.transform(unl.X)
    unl.X = unl.X.transpose(0,3,1,2)

    return unl

def resize(X):
    """
    Resizes the data partition X to the size specified in the config file.
    Uses bicubic interpolation for resizing.

    Returns:
        the resized images as a numpy array.
    """
    resized = []
    image_dim = config('image_dim')
    for x in X:
        tmp = Image.fromarray(x)
        img = tmp.resize((image_dim,image_dim), resample=Image.BICUBIC)
        # img = x.transform((image_dim,image_dim,3), resample=Image.BICUBIC)
        resized.append(np.asarray(img))
    return np.array(resized)

class ImageStandardizer(object):
    """
    Channel-wise standardization for batch of images to mean 0 and variance 1.
    The mean and standard deviation parameters are computed in `fit(X)` and
    applied using `transform(X)`.

    X has shape (N, image_height, image_width, color_channel)
    """
    def __init__(self):
        super().__init__()
        self.image_mean = []
        self.image_std = []

    def fit(self, X):
        self.image_mean = np.mean(X, axis=(0,1,2))
        self.image_std = np.std(X, axis=(0,1,2))

    def transform(self, X):
        X = X.astype(np.float64)
        for i in range(X.shape[3]):
            X[...,i] = np.subtract(X[...,i],self.image_mean[i])
            X[...,i] = np.divide(X[...,i],self.image_std[i])
        return X


class FashionDataset(Dataset):

    def __init__(self, partition, num_classes=10):
        """
        Reads in the necessary data from disk.
        """
        super().__init__()

        if partition not in ['train', 'val', 'test', 'unlabeled']:
            raise ValueError('Partition {} does not exist'.format(partition))

        np.random.seed(0)
        self.partition = partition
        self.num_classes = num_classes

        # Load in all the data we need from disk
        self.metadata = pd.read_csv(config('csv_file'))
        self.X, self.y = self._load_data()

        self.semantic_labels = dict(zip(
            self.metadata['numeric_label'],
            self.metadata['semantic_label']
        ))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]).float(), torch.tensor(self.y[idx]).long()

    def _load_data(self):
        """
        Loads a single data partition from file.
        """
        print("loading %s..." % self.partition)

        if self.partition == 'test':
            if self.num_classes == 5:
                df = self.metadata[self.metadata.partition == self.partition]
            else:
                raise ValueError('Unsupported test partition: num_classes must be 5')
        else:
            df = self.metadata[
                (self.metadata.numeric_label < self.num_classes) &
                (self.metadata.partition == self.partition)
            ]

        X, y = [], []
        for i, row in df.iterrows():
            image = imread(os.path.join(config('image_path'), row['filename']))
            X.append(image)
            y.append(row['numeric_label'])

        return np.array(X), np.array(y)

    def get_semantic_label(self, numeric_label):
        """
        Returns the string representation of the numeric class label (e.g.,
        the numberic label 1 maps to the semantic label 'sweaters').
        """
        return self.semantic_labels[numeric_label]

if __name__ == '__main__':
    # print(torch.cuda.is_available())
    ## Future note: check scipy imread and imresize
    import warnings
    # warnings.filterwarnings("ignore", category=DeprecationWarning)

    # np.set_printoptions(precision=3)

    # tr, va, te, standardizer = get_train_val_test_dataset()
    # print("Train:\t", len(tr.X))
    # print("Val:\t", len(va.X))
    # print("Test:\t", len(te.X))
    # print("Mean:", standardizer.image_mean)
    # print("Std: ", standardizer.image_std)

    A = np.array([[19/12, -2/3, -1/3, -1/4, -1/3], [-2/3, 7/6, 0, -1/2, 0], [-1/3, 0, 4/3, -3/4, -1/4], [-1/4, -1/2, -3/4, 3/2, 0], [-1/3, 0, -1/4, 0, 7/12]])
    w, v = eigh(A, subset_by_index=[0, 2])
    w = np.round(w, decimals=4)
    v = np.round(v, decimals=4)
    print('eigenvalues:\n', w)
    print('eigenvectors:')
    print(v)

