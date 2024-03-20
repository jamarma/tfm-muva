import os
from pathlib import Path

# Paths
LANEATT_PATH = os.path.join(Path(__file__).parent, 'laneatt')
MODEL_PATH = os.path.join(LANEATT_PATH, 'pretrained_models/laneatt_r34_culane/models/model_0015.pt')

# LaneATT model parameters
BACKBONE = 'resnet34'
IMG_HEIGHT = 360
IMG_WIDTH = 640
TOPK_ANCHORS = 1000
ANCHORS_FREQ_PATH = os.path.join(LANEATT_PATH, '.cache/culane_anchors_freq.pt')

# Inference parameters
CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 50.
MAX_LANES = 2
