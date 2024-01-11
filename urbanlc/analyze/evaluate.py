import rasterio
import numpy as np
from ..utils import export_geotiff
from tqdm.auto import tqdm
from .metrics import (
    confusion_matrix, accuracy, user_accuracy, producer_accuracy, cohen_kappa
)
from ast import literal_eval
from copy import copy
tqdm.pandas()

def parse_numpy(
    df,
    cols = ["confusion_matrix"],
    sep=",",
):
    for col in cols:
        vfunc = lambda x: np.array(literal_eval(x, sep=sep)) if x is not None else x
        df[col] = df[col].replace({np.nan: None}).apply(vfunc)

    return df

def get_invalid_pixel_rate(
    path,
    invalid_value=0,
):
    with rasterio.open(path) as dataset:
        img = dataset.read(dataset.count)

    total = img.shape[0] * img.shape[1]
    return (total - np.count_nonzero(img)) / total
    # return len(img[img == invalid_value]) / (img.shape[0] * img.shape[1])

def compute_eval_metrics(
    input_df,
    metrics=["accuracy", "producer_accuracy", "user_accuracy", "cohen_kappa"],
    save_path=None,
    confusion_matrix_args={},
):
    df = copy(input_df)
    # calculate confusion matrix
    if "confusion_matrix" not in list(df.columns):
        vfunc = lambda x: confusion_matrix(x.pred_path, x.gt_path, **confusion_matrix_args)
        df["confusion_matrix"] = df.progress_apply(vfunc, axis=1)
    
    # calculate other metrics (derived from confusion matrix)
    for metric in metrics:
        # if metric == "cohen_kappa": continue
        vfunc = lambda x: globals()[metric](x) if x is not None else x
        df[metric] = df["confusion_matrix"].apply(vfunc)

    # calculate cohen kappa
    # if "cohen_kappa" in metrics:
    #     vfunc = lambda x: cohen_kappa(x.pred_path, x.gt_path, **confusion_matrix_args)
    #     df["cohen_kappa"] = df.progress_apply(vfunc, axis=1)

    if save_path is not None:
        # convert numpy arrays to list for 
        cols = ["confusion_matrix"]
        vfunc = lambda x: x.tolist() if x is not None else x
        for col in cols:
            if col in list(df.columns):
                df[col] = df[col].apply(vfunc)
        
        df.to_csv(save_path, index=False)
    
    return df

def extract_esa300(city, bounds, start, end):
    for year in range(start, end + 1):
        esa300_path = f"./data_GUC/land_cover_ESA/original/{year}.tif"
        with rasterio.open(esa300_path) as src:

            corner1 = rasterio.transform.rowcol(src.transform, bounds[0], bounds[1])
            corner2 = rasterio.transform.rowcol(src.transform, bounds[2], bounds[3])
            start_col = min(corner1[0], corner2[0])
            end_col = max(corner1[0], corner2[0])
            start_row = min(corner1[1], corner2[1])
            end_row = max(corner1[1], corner2[1])
            # get (lon, lat) in at each pixel coordinate
            cols, rows = np.meshgrid(np.arange(start_row, end_row + 1), np.arange(start_col, end_col + 1))
            xs, ys = rasterio.transform.xy(src.transform, rows, cols)
            lons = np.array(xs)
            lats = np.array(ys)

            # get value of pixel at each pixel coordinate
            window = rasterio.windows.Window.from_slices(rows=[start_col, end_col+1], cols=[start_row, end_row+1])
            refs = src.read(window=window)

            output_meta = src.meta
            save_path = f"./data_GUC/land_cover_ESA/{city}/{year}.tif"
            params = {
                "img": refs,
                "save_path": save_path,
                "output_meta": output_meta,
                "compress": "PACKBITS",
            }
            export_geotiff(**params)
    
    return refs