"""
modified from 
https://github.com/microsoft/torchgeo/blob/main/torchgeo/datasets/landsat.py
https://github.com/microsoft/torchgeo/blob/main/torchgeo/datasets/geo.py
"""
import glob
import os
import sys
from typing import Dict, List, Tuple, cast, Optional
from natsort import natsorted

import rasterio
from rasterio.crs import CRS
from rasterio.vrt import WarpedVRT

from torch.utils.data import DataLoader

from .train_utils import set_seed
import numpy as np
import random
import warnings

try:
    from torchgeo.datasets import GeoDataset, RasterDataset, stack_samples
    from torchgeo.samplers import RandomGeoSampler
except Exception:
    warnings.warn("torchgeo is not installed, dataloder is disabled")

set_seed(0)

import os
import sys
import glob
import random
from typing import List, Optional, cast
from rasterio.crs import CRS
import rasterio.errors
from natsort import natsorted
import numpy as np
from torch.utils.data import DataLoader

def parse_paths(root: str, filename_glob: str, exclude: Optional[List[str]] = None):
    """
    Parse paths of files matching a certain pattern in the given root directory, excluding any paths that contain the specified exclusion keywords.

    :param root: Root directory to start the search.
    :param filename_glob: Glob pattern to match the filenames.
    :param exclude: List of strings to exclude from the matched filenames.

    :return: List of paths matching the criteria.
    """
    pathname = os.path.join(root, filename_glob)
    all_paths = list(glob.glob(pathname, recursive=True))

    # Naive implementation to exclude paths
    if exclude is not None:
        if isinstance(exclude, str):
            exclude = [exclude]
        for keyword in exclude:
            all_paths = [val for val in all_paths if keyword not in val]

    return natsorted(all_paths)

# https://stackoverflow.com/questions/62744176/how-to-overwrite-parent-class-init-method-and-use-super-to-call-grandparent-ini
class CustomRasterDataset(RasterDataset):
    """
    Custom raster dataset class for training DL model
    """
    def __init__(self, root=None, crs=None, res=None, bands=None, transforms=None, cache=False, exclude=None, all_paths=None):
        """
        Initializes a custom raster dataset by processing the specified file paths.

        :param root: Root directory of the dataset.
        :param crs: Coordinate reference system.
        :param res: Resolution of the dataset.
        :param bands: List of bands in the dataset.
        :param transforms: List of data transformations to apply.
        :param cache: Whether to cache the dataset.
        :param exclude: List of strings to exclude from file paths.
        :param all_paths: List of precomputed file paths.
        """
        GeoDataset.__init__(self, transforms)

        self.root = root
        self.cache = cache

        if all_paths is None:
            all_paths = parse_paths(root, self.filename_glob, exclude)
        else:
            all_paths = natsorted(all_paths)

        # Populate the dataset index
        i = 0
        for filepath in all_paths:
            try:
                with rasterio.open(filepath) as src:
                    # See if the file has a color map
                    if len(self.cmap) == 0:
                        try:
                            self.cmap = src.colormap(1)
                        except ValueError:
                            pass

                    if crs is None:
                        crs = src.crs
                        self.transform = src.transform
                    if res is None:
                        res = src.res[0]

                    with WarpedVRT(src, crs=crs) as vrt:
                        minx, miny, maxx, maxy = vrt.bounds
            except rasterio.errors.RasterioIOError:
                continue
            else:
                mint: float = 0
                maxt: float = sys.maxsize

                coords = (minx, maxx, miny, maxy, mint, maxt)
                self.index.insert(i, coords, filepath)
                i += 1

        if i == 0:
            raise FileNotFoundError(
                f"No {self.__class__.__name__} data was found in '{root}'"
            )

        if bands and self.all_bands:
            band_indexes = [self.all_bands.index(i) + 1 for i in bands]
            self.bands = bands
            assert len(band_indexes) == len(self.bands)
        elif bands:
            msg = (
                f"{self.__class__.__name__} is missing an `all_bands` attribute,"
                " so `bands` cannot be specified."
            )
            raise AssertionError(msg)
        else:
            band_indexes = None
            self.bands = self.all_bands

        self.band_indexes = band_indexes
        self._crs = cast(CRS, crs)
        self.res = cast(float, res)

class Landsat(CustomRasterDataset):
    """
    Landsat dataset for training/inference
    """
    separate_files = False
    def __init__(self, root, filename_glob, all_bands, *args, **kwargs):
        """
        Initialize Landsat dataset class.

        :param root: Root directory of the dataset.
        :param filename_glob: Glob pattern to match the filenames.
        :param all_bands: List of all available bands.
        :param args: Additional positional arguments.
        :param kwargs: Additional keyword arguments.
        """
        self.all_bands = all_bands
        self.filename_glob = filename_glob
        self.is_image = True

        super().__init__(root, *args, **kwargs)

    @classmethod
    def MSS(cls, root=None, filename_glob=None, *args, **kwargs):
        """
        Initialize Landsat dataset class for MSS sensor

        :param root: Root directory of the dataset.
        :param filename_glob: Glob pattern to match the filenames.
        :param args: Additional positional arguments.
        :param kwargs: Additional keyword arguments.

        :return: Instance of the Landsat dataset for MSS sensor.
        """
        all_bands = ["B4", "B5", "B6", "B7"]
        return Landsat(root, filename_glob, all_bands, *args, **kwargs)

    @classmethod
    def TM(cls, root=None, filename_glob=None, *args, **kwargs):
        """
        Initialize Landsat dataset class for TM sensor

        :param root: Root directory of the dataset.
        :param filename_glob: Glob pattern to match the filenames.
        :param args: Additional positional arguments.
        :param kwargs: Additional keyword arguments.

        :return: Instance of the Landsat dataset for TM sensor.
        """
        all_bands = ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7", "ST_B6"]
        return Landsat(root, filename_glob, all_bands, *args, **kwargs)

    @classmethod
    def OLITIRS(cls, root=None, filename_glob=None, *args, **kwargs):
        """
        Initialize Landsat dataset class for OLI/TIRS sensor

        :param root: Root directory of the dataset.
        :param filename_glob: Glob pattern to match the filenames.
        :param args: Additional positional arguments.
        :param kwargs: Additional keyword arguments.

        :return: Instance of the Landsat dataset for OLI/TIRS sensor.
        """
        all_bands = ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7", "ST_B10"]
        return Landsat(root, filename_glob, all_bands, *args, **kwargs)

class ESA2021(CustomRasterDataset):
    """
    ESA World Cover 2021 v200 dataset for training/inference
    """
    separate_files = False
    def __init__(self, root=None, filename_glob=None, crs=None, res=None, bands=None, transforms=None, cache=True, exclude=None, all_paths=None):
        """
        Initialize an ESA2021 (ESA WorldCover 2021 v200) dataset by processing the specified file paths.

        :param root: Root directory of the dataset.
        :param filename_glob: Glob pattern to match the filenames.
        :param crs: Coordinate reference system.
        :param res: Resolution of the dataset.
        :param bands: List of bands in the dataset.
        :param transforms: List of data transformations to apply.
        :param cache: Whether to cache the dataset.
        :param exclude: List of strings to exclude from file paths.
        :param all_paths: List of precomputed file paths.
        """
        self.all_bands = ["Map"]
        self.filename_glob = filename_glob
        self.is_image = False

        super().__init__(root, crs, res, bands, transforms, cache, exclude, all_paths)

def get_dataloader(landsat, esa, sensor_type, tile_per_loader=24, input_size=(224, 224), length=25000, batch_size=64, num_workers=8):
    """
    Get a PyTorch DataLoader for Landsat and ESA2021 datasets.

    :param landsat: Dictionary containing Landsat dataset parameters.
    :param esa: Dictionary containing ESA2021 dataset parameters.
    :param sensor_type: Type of Landsat sensor ("MSS", "TM", "OLITIRS").
    :param tile_per_loader: Number of tiles per loader.
    :param input_size: Size of input data.
    :param length: Length of the dataset.
    :param batch_size: Batch size.
    :param num_workers: Number of workers for DataLoader.

    :return: List of PyTorch DataLoaders.
    """
    assert sensor_type in ["MSS", "TM", "OLITIRS"], "valid sensor_type are ['MSS', 'TM', 'OLITIRS']"

    all_img_paths = natsorted(parse_paths(**landsat))
    all_label_paths = natsorted(parse_paths(**esa))
    assert len(all_img_paths) == len(all_label_paths)

    idxs = list(range(len(all_img_paths)))
    random.shuffle(idxs)

    loaders = []
    TILE_PER_LOADER = tile_per_loader
    num_loaders = int(np.ceil(float(len(idxs)) / TILE_PER_LOADER))
    for i in range(num_loaders):
        selected_idx = idxs[i*TILE_PER_LOADER : min(len(idxs), (i+1)*TILE_PER_LOADER)]
        selected_img = np.array(all_img_paths)[selected_idx]
        selected_gt = np.array(all_label_paths)[selected_idx]

        img_dataset = getattr(Landsat, sensor_type)(all_paths=selected_img, cache=False)
        label_dataset = ESA2021(all_paths=selected_gt, cache=True)
        dataset = img_dataset & label_dataset

        sampler = RandomGeoSampler(dataset, size=input_size, length=length)
        loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=stack_samples,
                            shuffle=False, num_workers=num_workers, pin_memory=False)
        loaders.append(loader)

    return loaders
