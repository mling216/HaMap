# %%
import h5py
import numpy as np
from PIL import Image
import os


# set the path to the folder containing slides' tiles
tiles_path = '/fs/scratch/PAS1575/Pathology/CAMELYON16/nonOverlapTestStainNorm'

# Find all the folders in the tiles_path
folders = [f for f in os.listdir(tiles_path) if os.path.isdir(os.path.join(tiles_path, f))]

res = []
for i, folder in enumerate(sorted(folders)):
    # if folder != 'tumor_111':
    #     continue
    if "normal" in folder or "tumor" in folder:
        continue
    print(folder)

    # find number of stained tiles in the folder
    folder_path = os.path.join(tiles_path, folder)
    num_stained_tiles = len([name for name in os.listdir(folder_path) if name.endswith('.png')])

    # find number of original tiles in the folder
    folder_path = os.path.join(tiles_path.replace('nonOverlapTestStainNorm', 'nonOverlapTest'), folder)
    num_tiles = len([name for name in os.listdir(folder_path) if name.endswith('.png')])

    # # check the number of files in f'/fs/ess/PAS1575/stain_norm_tiles_h5py/{folder}.hdf5'
    # hdf5_file_path = f'/fs/ess/PAS1575/stain_norm_tiles_h5py/{folder}.hdf5'

    # # check number of tiles in the existing hdf5 file
    # with h5py.File(hdf5_file_path, 'r') as hdf5_file:
    #     num_save_tiles = len(hdf5_file[folder].keys())

    print(f"Tiles {num_tiles}, stained tiles {num_stained_tiles}")  # tiles in hdf5: {num_save_tiles}")
    if num_tiles != num_stained_tiles:  # or num_tiles != num_save_tiles:
        res.append({
            'folder': folder,
            'num_tiles': num_tiles,
            'num_stained_tiles': num_stained_tiles,
            # 'num_save_tiles': num_save_tiles,
        })


# %%
for r in res:
    # print(f"Folder: {r['folder']}, Total Tiles: {r['num_tiles']}, Stained Tiles: {r['num_stained_tiles']}, Tiles in HDF5: {r['num_save_tiles']}")
    if r['num_tiles'] != r['num_stained_tiles']:
        print(f"Folder: {r['folder']} has {r['num_tiles']} total tiles but only {r['num_stained_tiles']} stained tiles. Please check!")

# %%
