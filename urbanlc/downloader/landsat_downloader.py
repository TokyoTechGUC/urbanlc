from __future__ import annotations

import json
import os
from abc import abstractmethod
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Union

import ee
from pydantic import Field, validate_arguments
from pydantic.typing import Annotated

from .logger import logger

from .base import BaseDownloader
from .downloader_constant import (
    LANDSAT_SURFACE_COLLECTION_PATH,
    LANDSAT_OPERATIONAL_TIME,
    LANDSAT_SURFACE_VALID_BANDS
)

class Landsat_Collection2(BaseDownloader):
    """
    Landsat_Collection2 abstract class for downloading Landsat data from GEE
    """

    def __init__(self, dataset_name, *args, **kwargs):
        """
        Initializes a Landsat_Collection2 instance.

        :param dataset_name: Name of the Landsat dataset.
        :type dataset_name: str
        :param args: Additional positional arguments passed to the parent class.
        :param kwargs: Additional keyword arguments passed to the parent class.
        """

        super().__init__(*args, **kwargs)
        self.dataset_name = dataset_name
        self.collection_path = LANDSAT_SURFACE_COLLECTION_PATH[self.dataset_name]
        self.bands = LANDSAT_SURFACE_VALID_BANDS[self.dataset_name]
        self.valid_years = LANDSAT_OPERATIONAL_TIME[self.dataset_name]

    @abstractmethod
    def apply_scale_factors(self):
        """
        Abstract method for applying scale factors to Landsat image bands.
        Subclasses must implement this method.
        """

        pass

    def mask_clouds(self, image: ee.image.Image) -> ee.image.Image:
        """
        Masks clouds, cloud shadows, and snow in the Landsat image using outputs from CFMask Algorithm
        
        references:
            https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T1_L2
            https://www.usgs.gov/landsat-missions/cfmask-algorithm

        :param image: Landsat image.
        :type image: ee.image.Image
        :return: Landsat image with clouds masked.
        :rtype: ee.image.Image
        """

        qa = image.select("QA_PIXEL")
        mask = (
            qa.bitwiseAnd(1 << 3)
            .And(qa.bitwiseAnd(1 << 9))
            .Or(qa.bitwiseAnd(1 << 4).And(qa.bitwiseAnd(1 << 11)))
            .Or(qa.bitwiseAnd(1 << 5).And(qa.bitwiseAnd(1 << 13)))
        )
        image = image.updateMask(mask.Not())

        return image

    def retrieve_image(
        self,
        region: ee.geometry.Geometry,
        year: List[int],
        months: List[int],
        use_tier2: Optional[bool] = False,
    ) -> Tuple[ee.image.Image, Union[None, List[str]]]:
        """
        Retrieves a Landsat image for a specific region, year, and months.
        Outputs are cloud-masked scaled annual pixel-wise median data, reprojected ESPG:4326.

        :param region: The region of interest.
        :type region: ee.geometry.Geometry
        :param year: The target year.
        :type year: List[int]
        :param months: The target months.
        :type months: List[int]
        :param use_tier2: Flag indicating whether to use Tier 2 data.
        :type use_tier2: Optional[bool]
        :return: Tuple containing the Landsat image and a list of dates.
        :rtype: Tuple[ee.image.Image, Union[None, List[str]]]
        """

        collection_path = self.collection_path
        if use_tier2:
            collection_path = collection_path.replace("T1", "T2")

        annual_images = None
        for month in months:
            images = (
                ee.ImageCollection(collection_path)
                .map(self.apply_scale_factors)
                .filterBounds(region)
                .filter(ee.Filter.calendarRange(year, year, "year"))
                .filter(ee.Filter.calendarRange(month, month, "month"))
                .map(self.mask_clouds)
            )
            annual_images = (
                images if annual_images is None else annual_images.merge(images)
            )

        # reduce to median to avoid cloud mask
        image = annual_images.median()

        if len(image.bandNames().getInfo()) > 0:
            # get time stamp
            extract_time_func = lambda x: datetime.fromtimestamp(x / 1e3).strftime(
                "%Y/%m/%d %H:%M"
            )
            time_stamps = annual_images.aggregate_array("system:time_start").getInfo()
            dates = list(map(extract_time_func, time_stamps))

            # project Landsat (EPSG:32647) to ESA (ESPG:4326) for image dimension consistency
            reprojection = ee.Projection(
                "EPSG:4326", [8.33333e-5, 0, -180, 0, -8.33333e-5, 84]
            )
            image_proj = image.reproject(reprojection)
        else:
            image_proj = None
            dates = None

        return image_proj, dates

    def download_images(
        self,
        bounds: List[float],
        radius: Optional[float] = None,
        years: Optional[Union[int, List[int]]] = list(range(2013, 2022)),
        months: Optional[List[int]] = list(range(1, 13)),
        verbose: Optional[bool] = True,
        allow_tier2: Optional[bool] = True,
        overwrite: Optional[bool] = True,
        num_workers: int = 8,
    ) -> None:
        """
        Downloads Landsat images based on specified parameters.
        Outputs are saved at ...

        :param bounds: The bounding box or center coordinates.
        :type bounds: List[float]
        :param radius: The radius around the center (only used if center coordinates are provided).
        :type radius: Optional[float]
        :param years: The target year or a list of years.
        :type years: Optional[Union[int, List[int]]]
        :param months: The target months for pixel-wise median
        :type months: Optional[List[int]]
        :param verbose: Flag indicating whether to display log messages.
        :type verbose: Optional[bool]
        :param allow_tier2: Flag indicating whether to allow falling back to Tier 2 data if Tier 1 data is unavailable.
        :type allow_tier2: Optional[bool]
        :param overwrite: Flag indicating whether to overwrite existing files.
        :type overwrite: Optional[bool]
        :param num_workers: Number of workers for concurrent downloads.
        :type num_workers: int
        """

        logger.disabled = False if verbose else True

        # temporal filter
        years = [years] if isinstance(years, int) else years
        assert len(set(years).difference(set(self.valid_years))) == 0

        # spatial filter
        bbox = self.get_spatial_filter(bounds, radius)

        logger.info(f"Dataset: {self.dataset_name}")
        logger.info(f"Years : {years}")
        logger.info(f"Bounding boxes : {bbox.bounds}")
        logger.info(f"Months for calculating median: {months}")

        # query images
        all_images = []
        for year in years:
            filename = f"{self.dataset_name}_{year}.tif"
            outfile = os.path.join(self.root, filename)
            if os.path.exists(outfile) and (not overwrite):
                print(f"Skipped because {outfile} already existed.")
                continue

            filename = f"{self.dataset_name}_{self.dataset_name}_{year}.tif"
            outfile = os.path.join(self.root, filename)
            if os.path.exists(outfile) and (not overwrite):
                print(f"Skipped because {outfile} already existed.")
                continue

            image_tier = 1
            image, dates = self.retrieve_image(bbox.region, year, months, use_tier2=False)

            if image is None:
                if not allow_tier2:
                    logger.warning(
                        f"Image in year {year} is unavailable, skipped since falling back to tier 2 is disabled."
                    )
                    continue
                else:
                    logger.warning(f"Fall back to tier 2")
                    image_tier = 2
                    image, dates = self.retrieve_image(
                        bbox.region, year, months, use_tier2=True
                    )
                    if image is None:
                        logger.error(
                            f"Failed to retrieve image for year {year} from {self.dataset_name}"
                        )
                        continue

            logger.info(f"Retrieved image for year {year} from {self.dataset_name}")
            all_images.append((year, image, image_tier, dates))

        # download images
        meta_dict = {"images": []}
        for year, image, tier, dates in all_images:
            self.download_image(image, bbox, base_filename=year, num_workers=num_workers)
            meta_data = {
                "name": f"{self.dataset_name}_{year}",
                "year": year,
                "region": bbox.bounds,
                "tier": tier,
                "dates": dates,
            }
            meta_dict["images"].append(meta_data)

        # dump log files
        logname = os.path.join(self.root, "log", f"{self.dataset_name}.log")
        os.makedirs(Path(logname).parent, exist_ok=True)
        with open(logname, "w") as f:
            json.dump(meta_dict, f)


##########################################
# Multispectral Scanner System (MSS)
##########################################
"""
DATA = DN values

(60 meter) B4: green (0.5 - 0.6 μm)
(60 meter) B5: red (0.6 - 0.7 μm)
(60 meter) B6: Near Infrared 1 (0.7 - 0.8 μm)
(30 meter) B7: Near Infrared 2 (0.8 - 1.1 μm)
(30 meter) QA_PIXEL
"""

class LandsatMSS(Landsat_Collection2):
    """
    LandsatMSS class for downloading DN (digital numbers) data of Landsat 1 - 5 (using MSS sensor)
    """

    @classmethod
    @validate_arguments
    def initialize(cls, n_landsat: Annotated[int, Field(ge=1, le=5)], *args, **kwargs):
        """
        Initializes a LandsatMSS instance.

        :param n_landsat: Annotated[int, Field(ge=1, le=5)] - Version of Landsat satellite (from 1 to 5).
        :return: LandsatMSS instance.
        """

        dataset_name = f"landsat{n_landsat}" if n_landsat <= 3 else f"landsat{n_landsat}-MSS"
        return LandsatMSS(dataset_name, *args, **kwargs)

    def apply_scale_factors(self, image: ee.image.Image) -> ee.image.Image:
        """
        Applies scale factors to Landsat image bands.

        :param image: ee.image.Image - Landsat image.
        :return: ee.image.Image - Image with scale factors applied.
        """

        image = image.select(self.bands)
        return image

##########################################
# (E)TM ([Enhanced] Thematic Mapper)
##########################################
"""
DATA = surface reflectance

SR_B1: blue (0.45-0.52 μm)
SR_B2: green (0.52-0.60 μm)
SR_B3: red (0.63-0.69 μm)
SR_B4: near infrared (0.77-0.90 μm)
SR_B5: shortwave infrared 1 (1.55-1.75 μm)
SR_B7: shortwave infrared 2 (2.08-2.35 μm)

ST_B6: surface temperature (10.40-12.50 μm)
"""

class LandsatTM(Landsat_Collection2):
    """
    LandsatTM class for downloading surface reflectance data of Landsat 4 - 7 (using TM sensor)
    """

    @classmethod
    @validate_arguments
    def initialize(cls, n_landsat: Annotated[int, Field(ge=4, le=7)], *args, **kwargs):
        """
        Initializes a LandsatTM instance.

        :param n_landsat: Annotated[int, Field(ge=4, le=7)] - Version of Landsat satellite (4, 5, or 7).
        :return: LandsatTM instance.
        """

        if n_landsat == 6:
            raise ValueError("Unfortunately, Landsat-6 failed to reach orbit...")
        return LandsatTM(f"landsat{n_landsat}", *args, **kwargs)

    # https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LT04_C02_T1_L2#description
    def apply_scale_factors(self, image: ee.image.Image) -> ee.image.Image:
        """
        Applies scale factors to Landsat image bands.

        :param image: ee.image.Image - Landsat image.
        :return: ee.image.Image - Image with scale factors applied.
        """

        optical_bands = image.select("SR_B.").multiply(0.0000275).add(-0.2)
        thermal_bands = image.select("ST_B6").multiply(0.00341802).add(149.0)

        image = (
            image.addBands(optical_bands, None, True)
            .addBands(thermal_bands, None, True)
            .select(self.bands)
        )
        return image


##########################################
# OLI/TIRS (Operational Land Imager and Thermal Infrared Sensor)
##########################################
"""
DATA = surface reflectance

SR_B1: ultra blue, coastal aerosol (0.435-0.451 μm)
SR_B2: blue (0.452-0.512 μm)
SR_B3: green (0.533-0.590 μm)
SR_B4: red (0.636-0.673 μm)
SR_B5: near infrared (0.851-0.879 μm)
SR_B6: shortwave infrared 1 (1.566-1.651 μm)
SR_B7: shortwave infrared 2 (2.107-2.294 μm)

SR_B10: surface temperature (10.60-11.19 μm)
"""

class LandsatOLITIRS(Landsat_Collection2):
    """
    LandsatOLITIRS class for downloading surface reflectance data of Landsat 8 - 9 (using OLI/TIRS sensor)
    """

    @classmethod
    @validate_arguments
    def initialize(cls, n_landsat: Annotated[int, Field(ge=8, le=9)], *args, **kwargs):
        """
        Initializes a LandsatOLITIRS instance.

        :param n_landsat: Annotated[int, Field(ge=8, le=9)] - Version of Landsat satellite (8 or 9).
        :return: LandsatOLITIRS instance.
        """

        return LandsatOLITIRS(f"landsat{n_landsat}", *args, **kwargs)

    # https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T1_L2#description
    def apply_scale_factors(self, image: ee.image.Image) -> ee.image.Image:
        """
        Applies scale factors to Landsat image bands.

        :param image: ee.image.Image - Landsat image.
        :return: ee.image.Image - Image with scale factors applied.
        """

        optical_bands = image.select("SR_B.").multiply(0.0000275).add(-0.2)
        thermal_bands = image.select("ST_B.*").multiply(0.00341802).add(149.0)

        image = (
            image.addBands(optical_bands, None, True)
            .addBands(thermal_bands, None, True)
            .select(self.bands)
        )
        return image
