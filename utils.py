import geopandas as gpd
import rasterio
import rasterio.features
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt 
import tifffile
from osgeo import gdal, ogr

def split_plots(tiff_file, shp_file, output_folder, prefix, label, field='OBJECTID'):
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
            format='GTiff',
            cutlineDSName=shp_file,
            cropToCutline=True,
            cutlineWhere=f"{field}='{feature.GetField(field)}'"
        )

        # Save the clipped image to a new TIFF file
        output_filename = os.path.join(output_folder, f"imagem_segmentada_{feature.GetField(field)}_{prefix}_{feature.GetField(label)}.tif")
        gdal.Warp(output_filename, tiff_dataset, options=options)

    # Close the datasets
    tiff_dataset = None
    shp_dataset = None


def sample_images(generator, noise, width, height, channels, subplots, figsize=(22,8), prefix=None, sufix=None, path=None, save=False):
    samples = []
    images = generator.predict(noise)
    plt.figure(figsize=figsize)
    #print(np.amax(generated_images))
    for i, image in enumerate(images):
        plt.subplot(subplots[0], subplots[1], i+1)
        if channels == 1:
            plt.imshow(image.reshape((width, height)), cmap='gray')    
                                                                            
        else:
            plt.imshow(image[:,:,:3])
            image = (image - image.min()) / (image.max() - image.min())

        if save == True:
            plt.figure(figsize=(figsize[0]/subplots[1], figsize[1]/subplots[0]))  # Create a new figure for each image
            plt.imshow(image[:,:,:3])
            plt.axis('off')
            img_name = f"synthetic_image_{i}_{prefix}_{sufix}"
            full_path = os.path.join(path, img_name)
            # Save RGB-based image as PNG
            plt.savefig(full_path + '.png')
            plt.close()  # Close the figure after saving

            # Save image with all 5 channels as TIFF
            image_data = np.transpose(image, (2, 0, 1))
            tifffile.imwrite(full_path + '.tiff', image_data, photometric='rgb')
        plt.subplots_adjust(wspace=None, hspace=None)
        plt.axis('off')
        joined_channels_image = np.concatenate([np.expand_dims(image[:, :, i], axis=0) for i in range(image.shape[2])], axis=0)
        joined_channels_image = np.transpose(joined_channels_image, (1, 2, 0))  # Transpose dimensions

        samples.append(joined_channels_image)
    plt.tight_layout()
    plt.show()
    return samples