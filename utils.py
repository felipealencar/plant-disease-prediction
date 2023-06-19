import geopandas as gpd
import rasterio
import rasterio.features
import numpy as np
import os
from tqdm import tqdm
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