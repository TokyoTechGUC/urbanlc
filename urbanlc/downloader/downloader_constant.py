# references
# https://developers.google.com/earth-engine/datasets/catalog/landsat
# https://developers.google.com/earth-engine/datasets/catalog/ESA_WorldCover_v200

"""
Dictionary mapping Landsat surface reflectance dataset names to their Earth Engine collection paths.
"""
LANDSAT_SURFACE_COLLECTION_PATH = {
    "landsat1": "LANDSAT/LM01/C02/T1",
    "landsat2": "LANDSAT/LM02/C02/T1",
    "landsat3": "LANDSAT/LM03/C02/T1",
    "landsat4-MSS": "LANDSAT/LM04/C02/T1",
    "landsat5-MSS": "LANDSAT/LM05/C02/T1",
    "landsat4": "LANDSAT/LT04/C02/T1_L2",
    "landsat5": "LANDSAT/LT05/C02/T1_L2",
    "landsat7": "LANDSAT/LE07/C02/T1_L2",
    "landsat8": "LANDSAT/LC08/C02/T1_L2",
    "landsat9": "LANDSAT/LC09/C02/T1_L2",
}

"""
Dictionary mapping Landsat surface reflectance dataset names to their target bands used in data downloaders.
"""
LANDSAT_SURFACE_VALID_BANDS = {
    "landsat1": ["B4", "B5", "B6", "B7", "QA_PIXEL"],
    "landsat2": ["B4", "B5", "B6", "B7", "QA_PIXEL"],
    "landsat3": ["B4", "B5", "B6", "B7", "QA_PIXEL"],
    "landsat4-MSS": ["B1", "B2", "B3", "B4", "QA_PIXEL"],
    "landsat5-MSS": ["B1", "B2", "B3", "B4", "QA_PIXEL"],
    "landsat4": ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7", "ST_B6", "QA_PIXEL"],
    "landsat5": ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7", "ST_B6", "QA_PIXEL"],
    "landsat7": ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7", "ST_B6", "QA_PIXEL"],
    "landsat8": ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7", "ST_B10", "QA_PIXEL"],
    "landsat9": ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7", "ST_B10", "QA_PIXEL"],
}

"""
Dictionary mapping ESA WorldCover dataset names to their Earth Engine collection paths.
"""
ESA2021_COLLECTION_PATH = {
    "ESAv100": "ESA/WorldCover/v100",
    "ESAv200": "ESA/WorldCover/v200",
}

"""
Dictionary mapping Landsat dataset names to their operational time periods.
"""
LANDSAT_OPERATIONAL_TIME = {
    "landsat1": list(range(1972, 1979)),
    "landsat2": list(range(1975, 1983)),
    "landsat3": list(range(1978, 1984)),
    "landsat4-MSS": list(range(1982, 1994)),
    "landsat5-MSS": list(range(1984, 2013)),
    "landsat4": list(range(1982, 1994)),
    "landsat5": list(range(1984, 2013)),
    "landsat7": list(range(1999, 2024)),
    "landsat8": list(range(2013, 2024)),
    "landsat9": list(range(2021, 2024)),
}