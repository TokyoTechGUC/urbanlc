from __future__ import annotations

import ee
from typing import List, Optional
from pydantic import validate_arguments, Field
from pydantic.typing import Annotated

from .base import BaseDownloader
from .logger import logger
from .downloader_constant import ESA2021_COLLECTION_PATH

class ESAWorldCover(BaseDownloader):
    """
    ESAWorldCover class for downloading ESA WorldCover 10m v100/v200
    """

    def __init__(self, dataset_name, *args, **kwargs):
        """
        Initializes a ESAWorldCover instance.

        :param dataset_name: Name of the dataset.
        :type dataset_name: str
        :param args: Additional positional arguments passed to the parent class.
        :param kwargs: Additional keyword arguments passed to the parent class.
        """

        super().__init__(*args, **kwargs)
        self.dataset_name = dataset_name
        self.collection_path = ESA2021_COLLECTION_PATH[self.dataset_name]
        self.bands = ["Map"]

    @classmethod
    @validate_arguments
    def initialize(cls, year: Annotated[int, Field(ge=2020, le=2021)], *args, **kwargs):
        """
        Initializes an ESAWorldCover instance based on the specified year.

        :param year: The target year (2020 or 2021).
        :type year: int
        :param args: Additional positional arguments passed to the constructor.
        :param kwargs: Additional keyword arguments passed to the constructor.
        :return: An instance of ESAWorldCover.
        :rtype: ESAWorldCover
        """

        dataset_name = "ESAv200" if year == 2021 else "ESAv100"
        return ESAWorldCover(dataset_name, *args, **kwargs)

    def download_images(
        self,
        bounds: List[float],
        radius: Optional[bool] = None,
        verbose: Optional[bool] = True,
        num_workers: int = 8,
    ) -> None:
        """
        Downloads ESA WorldCover data based on specified parameters.
        Outputs are saved at ...

        :param bounds: The bounding box or center coordinates.
        :type bounds: List[float]
        :param radius: The radius around the center (only used if center coordinates are provided).
        :type radius: Optional[bool]
        :param verbose: Flag indicating whether to display log messages.
        :type verbose: Optional[bool]
        :param num_workers: Number of workers for concurrent downloads.
        :type num_workers: int
        """

        logger.disabled = False if verbose else True

        bbox = self.get_spatial_filter(bounds, radius)
        land_cover = (
            ee.ImageCollection(self.collection_path)
            .filterBounds(bbox.region)
            .first()
            .select(self.bands)
        )
        self.download_image(land_cover, bbox, bands=self.bands, num_workers=num_workers)