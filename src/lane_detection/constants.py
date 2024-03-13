# LaneATT model parameters
BACKBONE = 'resnet34'
IMG_HEIGHT = 360
IMG_WIDTH = 640
TOPK_ANCHORS = 1000
ANCHORS_FREQ_PATH = 'laneatt/.cache/culane_anchors_freq.pt'

# Inference parameters
CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 50.
MAX_LANES = 4
