from __future__ import annotations

import ee
import rasterio
from rasterio.merge import merge

import os
import glob
import shutil
from pathlib import Path
from natsort import natsorted
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass

from abc import ABC, abstractmethod
import itertools
import numpy as np
from tqdm.auto import tqdm

from concurrent.futures import as_completed
from requests_futures.sessions import FuturesSession

from .logger import logger
from ..utils import export_geotiff

try:
    ee.Initialize()
except Exception:
    ee.Authenticate()
    ee.Initialize()

@dataclass
class BoundingBox:
    """
    BoundingBox class for representing geographic bounding boxes.

    .. autosummary::
       :toctree: generated/
       :exclude: bounds region
    """

    __slots__ = ["bounds", "region"]
    bounds: List[float]
    region: ee.geometry.Geometry

    @classmethod
    def from_bounds(cls, lon_min: float, lat_min: float, lon_max: float, lat_max: float) -> BoundingBox:
        """
        Create a BoundingBox instance from specified bounding box coordinates.

        :param lon_min: Minimum longitude.
        :type lon_min: float
        :param lat_min: Minimum latitude.
        :type lat_min: float
        :param lon_max: Maximum longitude.
        :type lon_max: float
        :param lat_max: Maximum latitude.
        :type lat_max: float
        :return: BoundingBox instance.
        :rtype: BoundingBox
        """

        # Coord format: lon_min, lat_min, lon_max, lat_max
        bounds = [lon_min, lat_min, lon_max, lat_max]
        region = ee.Geometry.BBox(*bounds)
        return BoundingBox(bounds, region)

    @classmethod
    def from_point(cls, lon: float, lat: float, radius: float = 20000) -> BoundingBox:
        """
        Create a BoundingBox instance from a specified point and radius.

        :param lon: Longitude of the point.
        :type lon: float
        :param lat: Latitude of the point.
        :type lat: float
        :param radius: Radius from the point.
        :type radius: float
        :return: BoundingBox instance.
        :rtype: BoundingBox
        """

        # Coord format: E, N
        point = ee.Geometry.Point(lon, lat)
        intermediate_circle = point.buffer(radius, 0.0)
        region = intermediate_circle.bounds()

        # extract bounding boxes for further processing
        coordinates = region.getInfo()["coordinates"]
        min_xy = np.min(coordinates, axis=1).squeeze()
        max_xy = np.max(coordinates, axis=1).squeeze()
        bounds = list(np.concatenate([min_xy, max_xy]))
        return BoundingBox(bounds, region)

    def get_partition(self, step: float = 0.07) -> List[List[float]]:
        """
        Divide the bounding box into patches of specified step size.

        :param step: Step size for partitioning.
        :type step: float
        :return: List of partitioned bounding boxes.
        :rtype: List[List[float]]
        """

        initial_x, initial_y, final_x, final_y = self.bounds
        x_list = np.arange(initial_x, final_x, step)
        y_list = np.arange(initial_y, final_y, step)

        patch_bounds = list(itertools.product(x_list, y_list))
        patch_boxes = [
            [x, y, min(x + step, final_x), min(y + step, final_y)]
            for x, y in patch_bounds
        ]
        return patch_boxes

class BaseDownloader(ABC):
    """
    BaseDownloader class serving as an abstract base class for various data downloaders.
    """

    def __init__(self, root: str, clear_cache: Optional[bool] = True) -> None:
        """
        Initializes a BaseDownloader instance.

        :param root: Root directory for storing downloaded data.
        :type root: str
        :param clear_cache: Flag indicating whether to clear the cache after merging patches.
        :type clear_cache: Optional[bool]
        """

        self.set_root(root)
        self.clear_cache = clear_cache

    def set_root(self, root: str) -> None:
        """
        Sets the root directory for storing downloaded data.

        :param root: Root directory path.
        :type root: str
        """

        self.root = root
        self.cache_dir = os.path.join(self.root, "cache")
        os.makedirs(self.root, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

    def remove_cache(self) -> None:
        """
        Removes the cache directory and recreates it.

        :return: None
        """

        try:
            shutil.rmtree(self.cache_dir)
        except OSError:
            pass
        os.makedirs(self.cache_dir, exist_ok=True)

    def get_spatial_filter(
        self, bounds: List[float], radius: float
    ) -> BoundingBox:
        """
        Returns a BoundingBox instance based on the provided bounds and radius.

        :param bounds: List of bounding box coordinates [lon_min, lat_min, lon_max, lat_max] or center coordinates [lon, lat].
        :type bounds: List[float]
        :param radius: Radius around the center (only used if center coordinates are provided).
        :type radius: float
        :return: BoundingBox instance.
        :rtype: BoundingBox
        :raises ValueError: If the bounds format is invalid.
        """

        if len(bounds) == 4:
            return BoundingBox.from_bounds(*bounds)
        elif len(bounds) == 2:
            return BoundingBox.from_point(*bounds, radius)
        else:
            raise ValueError(
                "bounds must be (lon, lat) or (min_lon, min_lat, max_lon, max_lat)"
            )

    def merge_patches(
        self,
        base_filename: str,
        compress:str = "PACKBITS",
    ) -> None:
        """
        Merges downloaded GEOTIFF image patches and saves the merged file with the specified compression algorithm.

        :param base_filename: Base filename for the merged file.
        :type base_filename: str
        :return: None
        """

        # retrieve all patches
        filenames = natsorted(
            glob.glob(os.path.join(self.cache_dir, f"{self.dataset_name}_{base_filename}_*.tif"))
        )
        patches = []
        for filename in filenames:
            patch = rasterio.open(filename)
            patches.append(patch)

        # merge
        mosaic, output = merge(patches)
        output_meta = patches[0].meta.copy()
        output_meta.update({"transform": output})

        filename = f"{self.dataset_name}_{base_filename}.tif"
        outfile = os.path.join(self.root, filename)
        export_geotiff(mosaic, outfile, output_meta, compress=compress,)

        logger.info(f"Successfully merge patches, saved at {outfile}")

        if self.clear_cache:
            self.remove_cache()
            logger.info(f"Deleted all files in {self.cache_dir}")

    def download_image(
        self,
        image: ee.image.Image,
        bbox: BoundingBox,
        bands: Optional[List[str]] = None,
        scale: int = 10,
        format: str = "GEOTIFF",
        base_filename: str = "",
        step: float = 0.07,
        merge_files: bool = True,
        num_workers: int = 8,
    ) -> None:
        """
        Parallel download image patches based on specified parameters.

        :param image: Earth Engine image to download.
        :type image: ee.image.Image
        :param bbox: BoundingBox instance representing the region to download.
        :type bbox: BoundingBox
        :param bands: List of image bands to download.
        :type bands: Optional[List[str]]
        :param scale: Spatial resolution (meters) of the downloaded image.
        :type scale: int
        :param format: Format of the downloaded image.
        :type format: str
        :param base_filename: Base filename for downloaded files.
        :type base_filename: str
        :param step: Step size for partitioning the region of interests.
        :type step: float
        :param merge_files: Flag indicating whether to merge downloaded patches.
        :type merge_files: bool
        :param num_workers: Number of workers for concurrent downloads.
        :type num_workers: int
        :return: None
        """

        patch_boxes = bbox.get_partition(step=step)
        save_params_list = [
            {
                "bands": self.bands if bands is None else bands,
                "region": ee.Geometry.BBox(*box),
                "scale": scale,
                "format": format,
            }
            for box in patch_boxes
        ]

        logger.info("Downloading image patches...")
        session = FuturesSession(max_workers=num_workers)
        futures = []
        for i, (box, save_params) in enumerate(zip(patch_boxes, save_params_list)):
            url = image.getDownloadUrl(save_params)
            future = session.get(url)
            future.filename = os.path.join(
                self.cache_dir, f"{self.dataset_name}_{base_filename}_{i}.tif"
            )
            future.i = i
            future.box = box
            futures.append(future)

        failed_patches = []
        for future in as_completed(futures):
            response = future.result()
            with open(future.filename, "wb") as fd:
                fd.write(response.content)
            
            logger.info(f"Downloaded and saved at {future.filename}")

            # verify file content validaity
            try:
                img = rasterio.open(future.filename)
            except rasterio.errors.RasterioIOError:
                failed_patches.append((future.i, future.box))
                os.remove(future.filename)

        # re-download fail patches by halfing the step size (for data with many bands)
        if len(failed_patches) > 0:
            logger.warn(f"#Failed patches: {len(failed_patches)}, recursively re-downloading")
            for i, (postfix, box) in enumerate(failed_patches):
                params = {
                    "image": image,
                    "bbox": BoundingBox.from_bounds(*box),
                    "bands": bands,
                    "scale": scale,
                    "format": format,
                    "base_filename": f"{base_filename}_{postfix}",
                    "step": step / 2,
                    "merge_files": False,
                }
                self.download_image(**params)

        # merge
        if merge_files:
            self.merge_patches(f"{base_filename}")

    @abstractmethod
    def download_images(self):
        """
        Abstract method for downloading images.
        Subclasses must implement this method.
        """

        pass
