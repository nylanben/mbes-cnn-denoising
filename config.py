# Defining parameters

VOXEL_SIZE    = 1.0 # meters
GRID_SHAPE    = (32, 32, 32) # must be divisible by 8 (three MaxPool3d layers)
DENSITY_SCALE = 100.0 # used only if PER_DATASET_SCALE = False - based on p95 of sounding density — see preprocessing sanity check
PER_DATASET_SCALE = True  # True = use dataset p95, False = use global DENSITY_SCALE 
CHANNELS      = 2 # channel 0: density, channel 1: occupancy
SEED          = 42 # for reproducibility
MIN_SOUNDINGS = 10 # for filtering cuboids with too few soundings