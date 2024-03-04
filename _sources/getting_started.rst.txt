Getting Started
====================================

The main features provided UrbanLC are downloading Landsat data and using them to infer land cover maps.
A set of quickstart Jupyter notebooks can be found `here <https://github.com/TokyoTechGUC/urbanlc>`_.

====================================
Downloading Landsat data
====================================

We provide illustrative examples as follows, which downloads surface reflectance data from Landsat-8.

**Providing a center coordinate and radius**

.. code-block:: python

    from urbanlc.downloader import LandsatOLITIRS

    save_path = "./demo_data/bangkok_landsat8"
    downloader = LandsatOLITIRS.initialize(n_landsat=8, root=save_path)

    coordinate = [139.6500, 35.6764] # latitude, longitude
    radius = 18000
    years = [2022]
    downloader.download_images(coordinate, radius=radius, years=years)

* The downloader used is ``LandsatOLITIRS`` because Landsat-8 is only equipped with OLI/TIRS sensor.
* The defined area is a :math:`18 \times 18 \text{km}^2` square centered at Bangkok, Thailand with ``(latitude, longitude) = (139.6500, 35.6764)``.
* The downloader will download the data from year ``2022`` and saved at ``./demo_data/bangkok_landsat8``

**Providing a bounding box**

.. code-block:: python

    from urbanlc.downloader import LandsatOLITIRS

    save_path = "./demo_data/bangkok_landsat8"
    downloader = LandsatOLITIRS.initialize(n_landsat=8, root=save_path)

    coordinate = [139.6500, 35.6764] # latitude, longitude
    radius = 18000
    years = [2022]
    downloader.download_images(coordinate, radius=radius, years=years)

* In this case, the area is defined using the upper-left and lower-right coordinates.

.. note::

    You can download data from other Landsat satellites, areas, and time periods by switch things out in the obvious way

    * The downloader can be changed
        * Use ``LandsatTM`` for Landsat 4 - 7 (well, except Landsat-6 which fails to reach the orbit).
        * Use ``LandsatMSS`` for Landsat 1 - 5.
        * Modify ``n_landsat`` to the correspond Landsat satellite, e.g., ``n_landsat = 7`` when downloads data from Landsat-7.
    * Retrieving data from multiple years is supported
        * Simply adding more years into the list, e.g., ``years = [2021, 2022]``.
        * Valid years for each downloader are listed `here <https://github.com/TokyoTechGUC/urbanlc>`_. 
    * Where to save the result (e.g. to store the Landsat images) can be adjusted by changing ``save_path``.

====================================
Predicting land cover maps
====================================

.. code-block:: python

    import torch
    import rasterio
    import os
    from pathlib import Path
    from tqdm.auto import tqdm

    from urbanlc import export_geotiff
    from urbanlc.model import LCClassifier

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LCClassifier.from_pretrained(sensor="OLITIRS", pretrained_model_name_or_path="OLITIRS_resnet50")
    model.to(device)
    model.model.eval()

    img_paths = ["./demo_data/bangkok_landsat8/landsat8_2022.tif"]
    output_paths = ["./demo_output/bangkok_landsat8/landsat8_2022.tif"]

    with torch.no_grad():
        for img_path, save_path in tqdm(zip(img_paths, output_paths), total=len(img_paths)):
            preds = model.infer(img_path, convert_numpy=True) # predictions are in 0 to N-1 classes
            land_cover = model.denormalize_class(preds) # convert labels to 10, 20, ..., 100 (ESA Worldcover format)

            # save prediction
            os.makedirs(Path(save_path).parent, exist_ok=True)
            output_meta = rasterio.open(img_path).meta
            output_meta["dtype"] = "uint8"
            output_meta["nodata"] = "0.0"
            params = {
                "img": land_cover,
                "save_path": save_path,
                "output_meta": output_meta,
                "compress": "PACKBITS",
            }
            export_geotiff(**params)

* The function ``LCClassifier.from_pretrained`` will automatically download the pretrained model corresponded to the given ``sensor`` from `HuggingFace <https://huggingface.co/sincostanx/urbanlc/tree/main>`_.
* Currently, this library only supports setting ``pretrained_model_name_or_path`` as ``f"{sensor}_resnet50"`` where ``sensor`` is ``MSS``, ``TM``, or ``OLITIRS``.
* The function ``denormalize_class`` convert the normalized model predictions to the values used in ESA World Cover.

====================================
Visualizing results
====================================

.. code-block:: python

    import matplotlib.pyplot as plt
    from urbanlc import plot_landsat, plot_land_cover

    fig, ax = plt.subplots(1, 2, figsize=(15, 10))

    image = rasterio.open("./demo_data/bangkok_landsat8/landsat8_2022.tif").read()
    plot_landsat(image, dataset="landsat8", ax=ax[0])

    land_cover = rasterio.open("./demo_output/bangkok_landsat8/landsat8_2022.tif").read()
    plot_land_cover(land_cover, ax=ax[1])


* Here, ``plot_landsat`` visualizes the RGB band of the given Landsat surface reflectance data
* Similarly, ``plot_land_cover`` visualizes LC maps in the 11-classes scheme adopted by ESA World Cover.

.. note::

    * ``dataset`` in ``plot_landsat`` should correspond to the Landsat satellite.