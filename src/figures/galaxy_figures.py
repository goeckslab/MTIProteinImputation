#!/usr/bin/env python

import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors
from skimage.io import imread
from skimage.util import map_array
from pathlib import Path

load_path: Path = Path('/home/groups/OMSAtlas/Code/kirchgae/MTIProteinImputation/results/evaluation/')
galaxy_image_path: Path = Path('/home/groups/OMSAtlas/Code/kirchgae/MTIProteinImputation/data/galaxy/')
# PARAMS ------------------------------------------------------------
# paths to original and imputed intensity CSV files

parser = ArgumentParser()
parser.add_argument("--protein", "-p", type=str, required=True, help="The protein")
parser.add_argument("--biopsy", "-b", type=str, required=True, help="The biopsy")
parser.add_argument("--vmin", "-vmin", type=float, required=True)

args = parser.parse_args()

protein: str = args.protein
biopsy: str = args.biopsy
vmin: float = args.vmin

original_fh = Path(load_path,biopsy, f"{biopsy}_{protein}_original.csv")
imputed_fh = Path(load_path,biopsy, f"{biopsy}_{protein}_imputed.csv")
galaxy_image_path = Path(galaxy_image_path, f"{biopsy}_{protein.lower()}.png")


# load the galaxy image
galaxy_image = imread(galaxy_image_path)

# path to mask
mask_fh = '/home/groups/OMSAtlas/Staging_Data/HTA9_2/HMS-SORGER/t-CycIF_Tumor_Panel/0000342251/level3/BEMS342251_Mesmer_mask.ome.tiff'


# PARAMS ------------------------------------------------------------

# read input CSV files
original = pd.read_csv(original_fh)
imputed = pd.read_csv(imputed_fh)

# add a 1-based cell index to match the mask
original['CellID'] = range(1, len(original)+1)

# concatenate original and imputed data
df = pd.concat([original[['CellID', 'Original Expression']], imputed[['Imputed Expression']]], axis=1)
df.columns = ['CellID', 'original', 'imputed']

# read the mask image
mask = imread(mask_fh)

# turn background pixels black using colormap (to differentiate from lowest expressing cells)
cmap = plt.get_cmap('viridis')
cmap.set_under(color='black')

# Create a panel with consistent size for each image
fig, ax = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

# Ensure all images have the same aspect ratio
aspect_ratio = galaxy_image.shape[1] / galaxy_image.shape[0]

# Display the galaxy image with the same aspect ratio
ax[0].imshow(galaxy_image, aspect='auto')
ax[0].set_title('Galaxy Image')
ax[0].set_xticks([])  # Remove x-axis ticks
ax[0].set_yticks([])  # Remove y-axis ticks

# Display the original expression without a colorbar
colored_mask_original = map_array(mask, np.array(df['CellID']), np.array(df['original']))
ax[1].imshow(colored_mask_original, interpolation='none', cmap=cmap, vmin=vmin, aspect='auto')
ax[1].set_title('Original Expression')
ax[1].set_xticks([])
ax[1].set_yticks([])

# Display the imputed expression with a colorbar
colored_mask_imputed = map_array(mask, np.array(df['CellID']), np.array(df['imputed']))
s2 = ax[2].imshow(colored_mask_imputed, interpolation='none', cmap=cmap, vmin=0.000000001, aspect='auto')
fig.colorbar(s2, ax=ax[2], fraction=0.025, pad=0.04)
ax[2].set_title('Imputed Expression')
ax[2].set_xticks([])
ax[2].set_yticks([])

# Save the figure
fig.savefig(f'colorized_mask_{biopsy}_{protein}_panel.png', dpi=500)
