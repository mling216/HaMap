# %%
import h5py
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def stain(img):
    # Your stain function implementation here
    pass

def get_all_coordinates(hdf5_file):
    coordinates = []
    def visit_func(name, node):
        if isinstance(node, h5py.Dataset):
            coordinates.append(name)
    hdf5_file.visititems(visit_func)
    return coordinates

def process_tiles(hdf5_file_path):
    with h5py.File(hdf5_file_path, 'r') as hdf5_file:
        coordinates = get_all_coordinates(hdf5_file)
        for coord in coordinates:
            img_array = np.array(hdf5_file[coord])
            # img = Image.fromarray(img_array)
            # plt.imshow(img)            
            print(f'Processing tile at coordinate: {coord}')
            print(img_array)
            # show the image

            break

# Example usage
process_tiles(hdf5_file_path = '/fs/ess/PAS1575/stain_norm_tiles_h5py/normal_042_PIL.hdf5')

# %%
