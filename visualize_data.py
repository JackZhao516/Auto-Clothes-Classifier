"""
EECS 445 - Introduction to Machine Learning
Fall 2020 - Project 2
Visualize Fashion
    This will open up a window displaying randomly selected training
    images. The label of the image is shown. Click on the figure to
    refresh with a set of new images. You can save the images using
    the save button. Close the window to break out of the loop.

    The success of this script is a good indication that the data flow
    part of this project is running smoothly.

    Usage: python visualize_data.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from imageio import imread
import time

from dataset import resize, ImageStandardizer, FashionDataset
from utils import config, denormalize_image

if __name__ == '__main__':
    training_set = FashionDataset('train', config('autoencoder.num_classes'))
    training_set.X = resize(training_set.X)
    standardizer = ImageStandardizer()
    standardizer.fit(training_set.X)

    metadata = pd.read_csv(config('csv_file'))
    metadata = metadata[metadata.notna()['semantic_label']]

    N = 4
    fig, axes = plt.subplots(nrows=2, ncols=N, figsize=(2*N,2*2))

    pad = 3
    axes[0,0].annotate('Original', xy=(0, 0.5), xytext=(-axes[0,0].yaxis.labelpad - pad, 0),
        xycoords=axes[0,0].yaxis.label, textcoords='offset points',
        size='large', ha='right', va='center', rotation='vertical')
    axes[1,0].annotate('Preprocessed', xy=(0, 0.5), xytext=(-axes[1,0].yaxis.labelpad - pad, 0),
        xycoords=axes[1,0].yaxis.label, textcoords='offset points',
        size='large', ha='right', va='center', rotation='vertical')

    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])

    np.random.seed(int(time.time()))
    rand_idx = np.random.choice(np.arange(len(metadata)), size=N, replace=False)
    X, y = [], []
    for idx in rand_idx:
        filename = os.path.join(
            config('image_path'), metadata.loc[idx, 'filename'])
        X.append(imread(filename))
        y.append(metadata.loc[idx, 'semantic_label'])

    for i, (xi, yi) in enumerate(zip(X, y)):
        axes[0,i].imshow(xi)
        axes[0,i].set_title(yi)

    X_ = resize(np.array(X))
    X_ = standardizer.transform(X_)

    for i, (xi, yi) in enumerate(zip(X_, y)):
        axes[1,i].imshow(denormalize_image(xi), interpolation='bicubic')

    fig.savefig('visualize_data.png'.format(yi), dpi=200, bbox_inches='tight')
