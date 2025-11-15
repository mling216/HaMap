# %%
### Load libraries
#
import numpy as np
import glob
from imageio import imread, imsave
import staintools
import os
import os.path as osp
import argparse as ap

# %%
# conduct the stain normalization
def main(data_path, output_path, n_prop, batch):

    # read images and sort
    data_tiles = glob.glob(osp.join(data_path, '*.png'))
    data_tiles = sorted(data_tiles)

    # get the batch
    batch_size = len(data_tiles) // n_prop
    if batch==n_prop:
        tiles_to_stain = data_tiles[(batch-1)*batch_size : ]
    else:    
        tiles_to_stain = data_tiles[(batch-1)*batch_size : batch*batch_size]

    data_tiles = [osp.basename(x) for x in tiles_to_stain]
    output_tiles = glob.glob(osp.join(output_path, '*.png'))
    output_tiles = [osp.basename(x) for x in output_tiles]

    # calculate what's left to stain
    tiles_path = []
    for x in data_tiles:
        if not x in output_tiles:
            tiles_path.append(osp.join(data_path, x))

    print('total tiles %d, batch tiles %d, to-stain tiles %d' % (len(output_tiles), 
            len(tiles_to_stain), len(tiles_path)))

    # color standardizer the reference image
    st_img = 'tumor_065_43224_100098.png'
    imagest = staintools.read_image(st_img)
    imagest = staintools.LuminosityStandardizer.standardize(imagest)

    # fit a stain normalizer
    stain_normalizer = staintools.StainNormalizer(method='vahadane')
    stain_normalizer.fit(imagest)

    # stain normalize all tiles
    num_non_std = 0
    num_non_norm = 0
    for i, tile in enumerate(tiles_path):
        if i % 100 == 0:
            print(i)
        img = imread(tile)
        img = img[:,:,:3]   # testing images are 4 channels
        # standardize brightness
        try:
            img_standard = staintools.LuminosityStandardizer.standardize(img)
        except:
            num_non_std += 1
            img_standard = img
        # use exception to jupm over "Empty tissue mask computed"
        try:
            img_norm = stain_normalizer.transform(img_standard)
        except:
            num_non_norm += 0
            # print(batch_sample.tile_loc[::-1], np.amin(img), np.amax(img)))
            img_norm = img_standard
        # save the result
        imsave(osp.join(output_path, osp.basename(tile)), img_norm)
    print(num_non_std, num_non_norm)


# load in command line parameter, and call the main fucntion
if __name__ == '__main__':
    
    parser = ap.ArgumentParser(description='Get dataset relative path')
    parser.add_argument('-d', metavar='--subfolder', type=str, action='store', 
                        dest='dataset', required=True, 
                        help='Tile set subfolder, e.g., train/normal')
    parser.add_argument('-n', metavar='--num_proportion', type=int, action='store', 
                        dest='n_prop', required=True, 
                        help='number of proportions to stain, e.g., 10')
    parser.add_argument('-b', metavar='--batch', type=int, action='store', 
                        dest='batch', required=True, 
                        help='batch no. from the number of proportions, e.g., 1')
            
    # Gather the provided arguments as an array.
    args = parser.parse_args()
    folder = vars(args)['dataset']
    print('dataset path:', folder)
    data_path = osp.join('/fs/scratch/PAS1575/Pathology/CAMELYON16/tissueTiles', folder)
    n_prop = vars(args)['n_prop']
    batch = vars(args)['batch']
    print('number of proportions', n_prop, 'prop no.', batch)

    if not osp.exists(data_path):
        print('dataset path %s does not exist!' % data_path)
    elif batch<1 or batch>n_prop:
        print('batch no. should be > 0 and <= num of proportion')
    else:
        output_path = data_path.replace('tissueTiles', 'tissueTilesStainNorm')  
        os.makedirs(output_path, exist_ok=True)
        main(data_path, output_path, n_prop, batch)

