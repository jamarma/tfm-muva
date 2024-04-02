import cv2
import argparse
import glob
import numpy as np

from natsort import natsorted
from pathlib import Path

import constants
from lane_detector import LaneDetector
from utils.drawing import draw_lanes


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames', help='Path of input frames', required=True)
    args = parser.parse_args()
    return args


def get_frame_paths(path: str) -> list:
    abs_path = str(Path(path).absolute())
    return natsorted(glob.glob(f'{abs_path}/*.png'))


def main():
    args = parse_args()
    frame_paths = get_frame_paths(args.frames)

    lane_detector = LaneDetector()

    for frame_path in frame_paths:
        frame = cv2.imread(frame_path)
        frame = cv2.resize(frame, (constants.IMG_WIDTH, constants.IMG_HEIGHT))
        output_frame = np.copy(frame)

        lanes = lane_detector.detect(frame)
        draw_lanes(output_frame, lanes)

        cv2.imshow('Output', output_frame)
        cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
