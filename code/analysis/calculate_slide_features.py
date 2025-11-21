import numpy as np
import pandas as pd
import os
import glob
from scipy.spatial.distance import pdist, squareform

    
### Function to aggregate tile-level predictions into slide-level features
def extract_slide_features(slide_id, group, tumor_threshold=0.5):

    tumor_probs = group['yhat'].values
    tumor_tiles = (tumor_probs > tumor_threshold).sum()
    total_tiles = len(tumor_probs)

    # Spatial Distribution Features
    x_coords = group['x'].values
    y_coords = group['y'].values
    
    if tumor_tiles > 0:
        tumor_x = x_coords[tumor_probs > tumor_threshold]
        tumor_y = y_coords[tumor_probs > tumor_threshold]
        tumor_centroid_x = np.mean(tumor_x)
        tumor_centroid_y = np.mean(tumor_y)
        tumor_spread_x = np.std(tumor_x)
        tumor_spread_y = np.std(tumor_y)
        
        # Compute Tumor Cluster Density (Mean pairwise distance between tumor tiles)
        tumor_coords = np.column_stack((tumor_x, tumor_y))
        if len(tumor_coords) > 1:
            pairwise_distances = pdist(tumor_coords)
            mean_tumor_cluster_dist = np.mean(pairwise_distances)
        else:
            mean_tumor_cluster_dist = np.nan
    else:
        tumor_centroid_x = tumor_centroid_y = np.nan
        tumor_spread_x = tumor_spread_y = np.nan
        mean_tumor_cluster_dist = np.nan
    
    # Compute Neighboring Tile Relationship (Mean difference in tumor probability between adjacent tiles)
    if len(tumor_probs) > 1:
        spatial_distances = squareform(pdist(np.column_stack((x_coords, y_coords))))
        nearest_neighbors = np.sort(spatial_distances, axis=1)[:, 1]  # Ignore self-distance (0)
        mean_neighbor_dist = np.mean(nearest_neighbors)
    else:
        mean_neighbor_dist = np.nan
    
    features = {
        'slide_id': slide_id,
        'mean_tumor_prob': np.mean(tumor_probs),
        'max_tumor_prob': np.max(tumor_probs),
        'min_tumor_prob': np.min(tumor_probs),
        'std_tumor_prob': np.std(tumor_probs),
        'tumor_tile_fraction': tumor_tiles / total_tiles,
        'tumor_centroid_x': tumor_centroid_x,
        'tumor_centroid_y': tumor_centroid_y,
        'tumor_spread_x': tumor_spread_x,
        'tumor_spread_y': tumor_spread_y,
        'entropy_tumor_prob': -np.sum(tumor_probs * np.log2(tumor_probs + 1e-9)) / total_tiles,  # Entropy
        'skew_tumor_prob': pd.Series(tumor_probs).skew(),
        'kurtosis_tumor_prob': pd.Series(tumor_probs).kurtosis(),
        'mean_tumor_cluster_dist': mean_tumor_cluster_dist,
        'mean_neighbor_dist': mean_neighbor_dist
    }
    
    df_slide_features = pd.DataFrame([features])
    return df_slide_features


# Load the tile-level predictions for each slide
path_to_slides = glob.glob('./whole_slide_prediction/test_*.csv')
list_slides = [os.path.basename(x) for x in path_to_slides]
# problematic_slides = ['normal_066', 'normal_076', 'normal_091', 'normal_114', 'tumor_092']

# Step 1: Extract slide-level features
for slide_id in list_slides:
    file_path = f'slide_classification/slide_features_{slide_id}'    
    if os.path.exists(file_path):
        print(f"File {file_path} already exists, skipping.")
        continue
    
    # if slide_id in problematic_slides:
    #     print(f"Skipping problematic slide: {slide_id}")
    #     continue

    print(f"Processing slide: {slide_id}")
    try:
        df_pred = pd.read_csv(f'./whole_slide_prediction/{slide_id}')
        slide_features = extract_slide_features(slide_id, df_pred)
        slide_features.to_csv(file_path, index=False)
    except Exception as e:
        print(f"Error processing slide {slide_id}: {e}")
        continue
