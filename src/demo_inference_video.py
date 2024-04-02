import cv2
import argparse
import numpy as np

import constants
from lane_detector import LaneDetector
from utils.drawing import draw_lanes, draw_fps
from utils.fps import FPS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', help='Path of input video', required=True)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    lane_detector = LaneDetector()

    cap = cv2.VideoCapture(args.video)
    fps = FPS().start()

    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (constants.IMG_WIDTH, constants.IMG_HEIGHT))
        output_frame = np.copy(frame)

        if not ret:
            break

        lanes = lane_detector.detect(frame)
        draw_lanes(output_frame, lanes)

        fps.update()
        draw_fps(output_frame, fps.fps())

        cv2.imshow('Output', output_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    fps.stop()
    print("Approximate time per frame {}".format(fps.fps()))
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
