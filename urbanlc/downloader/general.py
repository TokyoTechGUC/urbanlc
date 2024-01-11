import ee
from typing import Optional, Callable, Dict, Any
from pydantic import validate_arguments

from .base import BoundingBox, BaseDownloader

class GeneralDownloader(BaseDownloader):
    def __init__(self, dataset_name: str, root: str, clear_cache: Optional[bool]=True):
        super().__init__(root, clear_cache)
        self.dataset_name = dataset_name

    @validate_arguments
    def download_images(
        self,
        func: Callable,
        args: Dict[str, Any]
    ) -> None:

        imgs, bboxes, bands_list, filenames = func(**args)

        assert isinstance(imgs[0], ee.image.Image)
        assert isinstance(bboxes[0], BoundingBox)
        assert isinstance(bands_list[0], list)
        
        for img, bbox, bands, filename in zip(imgs, bboxes, bands_list, filenames):
            self.download_image(img, bbox, bands=bands, base_filename=filename)