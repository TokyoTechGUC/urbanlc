import rasterio
import os
from pathlib import Path
import numpy as np
from typing import Optional, Dict, Any
from rasterio.enums import Resampling

def open_at_size(
    path: str,
    ref: np.ndarray,
) -> np.ndarray:
    """

    Open a .tif file and downsample it to match the size of another .tif file.

    This function opens a .tif file specified by the path and downsamples it to match the size of
    a reference array provided. The downsampling is performed using Resampling.mode.

    :param path: Path to the input .tif file.
    :type path: str
    :param ref: Reference array to determine the desired size for downsampling.
    :type ref: np.ndarray

    :return: Downsampled data.
    :rtype: np.ndarray
    """
    with rasterio.open(path) as dataset:
        data = dataset.read(
            out_shape=ref.shape,
            resampling=Resampling.mode
        )
    
    return data


def open_at_scale(
    path: str,
    downsample_scale: float,
) -> np.ndarray:
    """
    Open a .tif file and downsample it by a constant factor.

    This function opens a .tif file specified by the path and downsamples it by a constant factor.
    The downsampling factor is applied to both the height and width dimensions of the input data.

    :param path: Path to the input .tif file.
    :type path: str
    :param downsample_scale: Constant factor for downsampling.
    :type downsample_scale: float

    :return: Downsampled data.
    :rtype: np.ndarray
    """
    with rasterio.open(path) as dataset:
        data = dataset.read(
            out_shape=(
                dataset.count,
                int(dataset.height / downsample_scale),
                int(dataset.width / downsample_scale)
            ),
            resampling=Resampling.mode
        )
    
    return data


def export_geotiff(
    img: np.ndarray,
    save_path: str,
    output_meta: Dict[str, Any],
    compress: Optional[str] = None,
    tiled: Optional[bool] = True,
    blockxsize: Optional[int] = 256,
    blockysize: Optional[int] = 256,
    interleave: Optional[str] = "band",
) -> None:
    """
    Export a NumPy array as a GeoTIFF file.

    This function exports a NumPy array as a GeoTIFF file specified with the provided metadata.
    Compression, tiling, and block size options can be customized.

    :param img: Input data to be exported.
    :type img: np.ndarray
    :param save_path: File path to save the GeoTIFF file.
    :type save_path: str
    :param output_meta: Metadata dictionary for the GeoTIFF file.
    :type output_meta: Dict[str, Any]
    :param compress: Compression method for the GeoTIFF file. Defaults to None.
    :type compress: str, optional
    :param tiled: Whether to use tiled format for the GeoTIFF file. Defaults to True.
    :type tiled: bool, optional
    :param blockxsize: Block size for x-dimension. Defaults to 256.
    :type blockxsize: int, optional
    :param blockysize: Block size for y-dimension. Defaults to 256.
    :type blockysize: int, optional
    :param interleave: Interleave format for the GeoTIFF file. Defaults to "band".
    :type interleave: str, optional

    :return: None
    """
    output_meta.update({
        "driver": "GTiff",
        "count": img.shape[0],
        "height": img.shape[1],
        "width": img.shape[2],
    })
    try:
        os.makedirs(Path(save_path).parent, exist_ok=True)
        with rasterio.open(
            save_path,
            "w",
            compress=compress,
            tiled=tiled,
            blockxsize=blockxsize,
            blockysize=blockysize,
            interleave=interleave,
            **output_meta
        ) as f:
            f.write(img)
    except Exception as e:
        raise (e)