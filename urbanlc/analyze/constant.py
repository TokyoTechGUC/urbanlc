
######################################
# Landsat constants
######################################
LANDSAT_RGB = {
    "landsat1": [1],
    "landsat2": [1],
    "landsat3": [1],
    "landsat4-MSS": [1],
    "landsat5-MSS": [1],
    "landsat4": [2, 1, 0],
    "landsat5": [2, 1, 0],
    "landsat7": [2, 1, 0],
    "landsat8": [3, 2, 1],
}

######################################
# ESA 10 m resolution data constant
######################################
ESA2021_LABEL = {
    10: ("#006400", "Tree"),  # Tree cover
    20: ("#ffbb22", "Shrubland"),  # Shrubland
    30: ("#ffff4c", "Grassland"),  # Grassland
    40: ("#f096ff", "Cropland"),  # Cropland
    50: ("#fa0000", "Built-up"),  # Built-up
    60: ("#b4b4b4", "Bare"),  # Bare / sparse vegetation2
    70: ("#f0f0f0", "Snow"),  # Snow and ice
    80: ("#0064c8", "Water"),  # Permanent water bodies
    90: ("#0096a0", "Wetland"),  # Herbaceous wetland
    95: ("#00cf75", "Mangroves"),  # Mangroves
    100: ("#fae6a0", "Moss"),  # Moss and lichen
}
ESA2021_CLASSES = sorted(ESA2021_LABEL.keys())

######################################
# JAXA v21.11 class
######################################
JAXA2111_LABEL = {
    1: (None, "Tree"),  # Water bodies
    2: (None, "Built-up"),  # Built-up
    3: (None, "Paddy"),  # Paddy field
    4: (None, "Cropland"),  # Cropland
    5: (None, "Grassland"),  # Grassland
    6: (None, "DBF"),  # DBF (deciduous broad-leaf forest)
    7: (None, "DNF"),  # DNF (deciduous needle-leaf forest)
    8: (None, "EBF"),  # EBF (evergreen broad-leaf forest)
    9: (None, "ENF"),  # ENF (evergreen needle-leaf forest)
    10: (None, "Bare"),  # Bare
    11: (None, "Bamboo"),  # Bamboo forest
    12: (None, "Solar"),  # Solar panel
}

######################################
# Evaluation constants
######################################

# Super class for evaluation
super_class = {
    0: ("#006400", "Vegetation"),
    1: ("#fa0000", "Built-up"),
    2: ("#0064c8", "Water"),
    3: ("#b4b4b4", "Bare"),
    4: ("#f0f0f0", "Snow"),
}

# Map label in ESA 300 m resolution dataset to super class for evaluation
ESA1992_map = {
    0: -1,                    # No Data
    10: 0,                   # Cropland, rainfed
        11: 0,                   # Herbaceous cover
        12: 0,                   # Tree or shrub cover
    20: 0,                   # Cropland, irrigated or post‐flooding
    30: 0,                   # Mosaic cropland (>50%) / natural vegetation (tree, shrub, herbaceous cover) (<50%)
    40: 0,                   # Mosaic  natural vegetation (tree, shrub, herbaceous cover) (>50%) / cropland (<50%)
    50: 0,                   # Tree cover, broadleaved, evergreen, closed to open (>15%)
    60: 0,                   # Tree cover, broadleaved, deciduous, closed to open (>15%)
        61: 0,                   # Tree cover, broadleaved, deciduous, closed (>40%)
        62: 0,                   # Tree cover, broadleaved, deciduous, open (15‐40%)
    70: 0,                   # Tree cover, needleleaved, evergreen, closed to open (>15%)
        71: 0,                   # Tree cover, needleleaved, evergreen, closed (>40%)
        72: 0,                   # Tree cover, needleleaved, evergreen, open (15‐40%)
    80: 0,                   # Tree cover, needleleaved, deciduous, closed to open (>15%)
        81: 0,                   # Tree cover, needleleaved, deciduous, closed (>40%)
        82: 0,                   # Tree cover, needleleaved, deciduous, open (15‐40%)
    90: 0,                   # Tree cover, mixed leaf type (broadleaved and needleleaved)
    100: 0,                  # Mosaic tree and shrub (>50%) / herbaceous cover (<50%)
    110: 0,                  # Mosaic herbaceous cover (>50%) / tree and shrub (<50%)
    120: 0,                  # Shrubland
        121: 0,                  # Evergreen shrubland
        122: 0,                  # Deciduous shrubland
    130: 0,                  # Grassland
    140: 0,                  # Lichens and mosses
    150: 3,                  # Sparse vegetation (tree, shrub, herbaceous cover) (<15%)
        151: 3,                  # Sparse tree (<15%)
        152: 3,                  # Sparse shrub (<15%)
        153: 3,                  # Sparse herbaceous cover (<15%)
    160: 0,                  # Tree cover, flooded, fresh or brakish water
    170: 0,                  # Tree cover, flooded, saline water
    180: 0,                  # Shrub or herbaceous cover, flooded, fresh/saline/brakish water
    190: 1,                  # Urban areas
    200: 3,                  # Bare areas
        201: 3,                  # Consolidated bare areas
        202: 3,                  # Unconsolidated bare areas
    210: 2,                  # Water bodies
    220: 4,                  # Permanent snow and ice
}

# Map label in ESA 10 m resolution dataset to super class for evaluation
ESA2021_map = {
    0: -1,                    # No Data
    10: 0,        # Tree cover
    20: 0,   # Shrubland
    30: 0,   # Grassland
    40: 0,    # Cropland
    50: 1,    # Built-up
    60: 3,  # Bare / sparse vegetation2
    70: 4,        # Snow and ice
    80: 2,       # Permanent water bodies
    90: 0,     # Herbaceous wetland
    95: 0,   # Mangroves
    100: 0,       # Moss and lichen
}

# Map label in JAXA 21.11 dataset to super class for evaluation
JAXA21_map = {
    0: -1,                    # No Data
    1: 2,       #1: Water bodies
    2: 1,       #2: Built-up
    3: 0,       #3: Paddy field
    4: 0,       #4: Cropland
    5: 0,       #5: Grassland
    6: 0,       #6: DBF (deciduous broad-leaf forest)
    7: 0,       #7: DNF (deciduous needle-leaf forest)
    8: 0,       #8: EBF (evergreen broad-leaf forest)
    9: 0,       #9: ENF (evergreen needle-leaf forest)
    10: 3,      #10: Bare
    11: 0,      #11: Bamboo forest
    12: 1,      #12: Solar panel
}

def get_normalized_map(labels):
    classes = sorted(list(labels.keys()))
    return {val: i for i, val in enumerate(classes)}
