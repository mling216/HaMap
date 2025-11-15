# run this script to save all the non-overlapping stain normalized tiles in hdf5 format
import h5py
import numpy as np
from PIL import Image
from imageio import imread
import os

def save_tiles_to_hdf5(folder_path, hdf5_file_path):
    with h5py.File(hdf5_file_path, 'w') as hdf5_file:
        for root, _, files in os.walk(folder_path):
            for i, file in enumerate(files):
                if i % 1000 == 0:
                    print(f"{i}")
                # check if the file is less than 10kb
                if os.path.getsize(os.path.join(root, file)) < 10000:
                    print(f"File {file} is empty, skipping {i}.")
                    continue
                if file.endswith('.png'):
                    slide_name, x, y = parse_filename(file)
                    img = Image.open(os.path.join(root, file))
                    img_array = np.array(img)
                    # img_array = imread(os.path.join(root, file))
                    hdf5_file.create_dataset(f"{slide_name}/{x}_{y}", data=img_array)

def parse_filename(filename):
    parts = filename.split('_')
    slide_name = '_'.join(parts[:2])
    x = parts[2]
    y = parts[3].split('.')[0]
    return slide_name, x, y

# set the path to the folder containing slides' tiles
tiles_path = '/fs/scratch/PAS1575/Pathology/CAMELYON16/nonOverlapTestStainNorm'

# Find all the folders in the tiles_path
folders = [f for f in os.listdir(tiles_path) if os.path.isdir(os.path.join(tiles_path, f))]

folders_to_skip = ['test_033', 'test_042', 'test_069', 'test_105']

for i, folder in enumerate(sorted(folders)):
    # if folder in folders_to_skip:
    #     continue
    if folder != 'test_105':
        continue
    print(folder)

    save_tiles_to_hdf5(os.path.join(tiles_path, folder), 
                    f'/fs/ess/PAS1575/stain_norm_tiles_h5py/{folder}.hdf5')

print("All folders processed.")


# # %% open a thumbnail of a WSI image
# import openslide
# import matplotlib.pyplot as plt
# def open_thumbnail(slide_path, thumbnail_size=(500, 500)):
#     slide = openslide.OpenSlide(slide_path)
#     # show the dimensions of the slide
#     print(f"Slide dimensions: {slide.dimensions}")
#     thumbnail = slide.get_thumbnail(thumbnail_size)
#     return thumbnail

# # Example usage
# thumbnail = open_thumbnail('/fs/ess/PAS1575/Dataset/CAMELYON16/training/all/tumor_111.tif')
# plt.imshow(thumbnail)
# plt.show()


