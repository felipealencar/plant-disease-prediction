import geopandas as gpd
import rasterio
import rasterio.features
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import tifffile
from osgeo import gdal, ogr

import networkx as nx
import dgl
from dgl.data import DGLDataset
from dgl import DGLGraph
import torch

import cv2

from skimage.segmentation import slic

def load_images(folder, label=""):
    imgs = []
    target = 1
    labels = []
    for i in os.listdir(folder):
        if i.endswith(label):
            img_dir = os.path.join(folder,i)
            try:
                img = tifffile.imread(img_dir)
                if img.shape[1] > 128:
                    img = img[:,:,:5]
                    img = cv2.resize(img, (128,128))
                    import sys
                    import numpy
                    numpy.set_printoptions(threshold=sys.maxsize)
                    img = (img / 256).astype(np.uint8)
                if 'synthetic' not in i:
                    img = (img - img.min()) / (img.max() - img.min())
                print(i)
                if 'synthetic_image_19_1007-35-PREVIOUSDATE_PPGAN_HEALTHY' in i:
                    print('synthetic', img)
                imgs.append(img)
                labels.append(target)
            except:
                continue
                
    imgs = np.array(imgs)
    labels = np.array(labels)

    return imgs, labels

def split_plots(tiff_file, shp_file, output_folder, prefix, label, field="OBJECTID"):
    # Open the TIFF image file
    tiff_dataset = gdal.Open(tiff_file, gdal.GA_ReadOnly)
    if tiff_dataset is None:
        raise Exception(f"Failed to open TIFF file: {tiff_file}")

    # Open the shapefile
    shp_dataset = ogr.Open(shp_file)
    if shp_dataset is None:
        raise Exception(f"Failed to open shapefile: {shp_file}")

    layer = shp_dataset.GetLayer()

    # Create a new TIFF file for each feature in the shapefile
    for i in tqdm(range(layer.GetFeatureCount())):
        feature = layer.GetFeature(i)
        geometry = feature.GetGeometryRef()
        xmin, xmax, ymin, ymax = geometry.GetEnvelope()

        # Define the options for clipping
        options = gdal.WarpOptions(
            outputBounds=(xmin, ymin, xmax, ymax),
            format="GTiff",
            cutlineDSName=shp_file,
            cropToCutline=True,
            cutlineWhere=f"{field}='{feature.GetField(field)}'",
        )

        # Save the clipped image to a new TIFF file
        output_filename = os.path.join(
            output_folder,
            f"imagem_segmentada_{feature.GetField(field)}_{prefix}_{feature.GetField(label)}.tif",
        )
        gdal.Warp(output_filename, tiff_dataset, options=options)

    # Close the datasets
    tiff_dataset = None
    shp_dataset = None


def sample_images(
    generator,
    noise,
    width,
    height,
    channels,
    subplots,
    figsize=(22, 8),
    prefix=None,
    model=None,
    path=None,
    save=False,
):
    samples = []
    images = generator.predict(noise)
    plt.figure(figsize=figsize)
    # print(np.amax(generated_images))
    for i, image in enumerate(images):
        plt.subplot(subplots[0], subplots[1], i + 1)
        if channels == 1:
            plt.imshow(image.reshape((width, height)), cmap="gray")

        else:
            if model == "WGAN":
                image = (image - image.min()) / (image.max() - image.min())
                plt.imshow(image[:, :, :3])
            else:
                plt.imshow(image[:, :, :3])
            image = (image - image.min()) / (image.max() - image.min())

        if save == True:
            plt.figure(
                figsize=(figsize[0] / subplots[1], figsize[1] / subplots[0])
            )  # Create a new figure for each image
            plt.imshow(image[:, :, :3])
            plt.axis("off")
            img_name = f"synthetic_image_{i}_{prefix}_{model}"
            full_path = os.path.join(path, img_name)
            # Save RGB-based image as PNG
            plt.savefig(full_path + ".png")
            plt.close()  # Close the figure after saving

            # Save image with all 5 channels as TIFF
            image_data = np.transpose(image, (2, 0, 1))
            print(image_data.shape)
            tifffile.imwrite(full_path + ".tiff", image_data, photometric="rgb")
        plt.subplots_adjust(wspace=None, hspace=None)
        plt.axis("off")
        joined_channels_image = np.concatenate(
            [np.expand_dims(image[:, :, i], axis=0) for i in range(image.shape[2])],
            axis=0,
        )
        joined_channels_image = np.transpose(
            joined_channels_image, (1, 2, 0)
        )  # Transpose dimensions

        samples.append(joined_channels_image)
    plt.tight_layout()
    plt.show()
    return samples


def graph_segmentation(image, num_days, num_segments):
    print(image.dtype)
    print(image.shape)
    # Perform superpixel segmentation using SLIC
    segments = slic(image, n_segments=num_segments)

    # Create a DGL graph
    g = dgl.DGLGraph()

    # Add nodes to the graph
    num_nodes = np.max(segments)
    g.add_nodes(num_nodes)

    # Initialize a dictionary to store "num_days" for each superpixel
    num_days_dict = {}

    # Iterate through the superpixels and connect neighboring superpixels
    src, dst = [], []
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            node_id = segments[i, j]
            if j < image.shape[1] - 1:
                neighbor_id = segments[i, j + 1]
                if node_id != neighbor_id and node_id < num_nodes and neighbor_id < num_nodes:
                    src.append(node_id)
                    dst.append(neighbor_id)
            if i < image.shape[0] - 1:
                neighbor_id = segments[i + 1, j]
                if node_id != neighbor_id and node_id < num_nodes and neighbor_id < num_nodes:
                    src.append(node_id)
                    dst.append(neighbor_id)


            # Store "num_days" information for each superpixel
            if node_id not in num_days_dict:
                num_days_dict[node_id] = []  # Initialize an empty list
            num_days_dict[node_id].append(num_days)

    g.add_edges(src, dst)

    # You can add node features to the graph based on superpixel data
    # For example, you can compute the average color or other statistics for each superpixel
    superpixel_features = np.zeros((num_nodes, image.shape[2]), dtype=np.float32)
    
    for node_id in range(num_nodes):
        # Compute superpixel features (e.g., average color)
        superpixel_mask = (segments == node_id)
        for channel in range(image.shape[2]):
            superpixel_features[node_id, channel] = np.mean(image[..., channel][superpixel_mask])

    g.ndata['feat'] = torch.tensor(superpixel_features)
    # Add the "num_days" attribute to the nodes in the graph
    num_days_attr = [np.mean(num_days_dict[node_id]) for node_id in range(1, num_nodes + 1)]
    g.ndata['days'] = torch.tensor(num_days_attr, dtype=torch.float)

    return g