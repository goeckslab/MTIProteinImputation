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

args = parser.parse_args()

protein: str = args.protein
biopsy: str = args.biopsy

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


# create a panel using the galaxy image on the right hand side, the original in the middle and the imputed data on the left hand side
fig, ax = plt.subplots(1, 3, figsize=(15, 5))

# display the galaxy image
ax[0].imshow(galaxy_image)
colored_mask = map_array(mask, np.array(df['CellID']), np.array(df['original']))
s = ax[1].imshow(colored_mask, interpolation='none', cmap=cmap, vmin=0.000000001)
fig.colorbar(s, ax=ax[1], fraction=0.046, pad=0.04)
ax[1].set_axis_off()

colored_mask = map_array(mask, np.array(df['CellID']), np.array(df['imputed']))
s = ax[2].imshow(colored_mask, interpolation='none', cmap=cmap, vmin=0.000000001)
fig.colorbar(s, ax=ax[2], fraction=0.046, pad=0.04)
ax[2].set_axis_off()

fig.savefig(f'colorized_mask_{biopsy}_{protein}_panel.png', dpi=500)