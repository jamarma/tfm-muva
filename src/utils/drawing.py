import cv2
import numpy as np


def draw_lanes(frame: np.array, lanes: list):
    if lanes is not None:
        for lane in lanes:
            points = lane.points
            points[:, 0] *= frame.shape[1]
            points[:, 1] *= frame.shape[0]
            points = points.round().astype(int)
            for curr_p, next_p in zip(points[:-1], points[1:]):
                cv2.line(frame, tuple(curr_p), tuple(next_p),
                         color=(0, 255, 0), thickness=3)


def draw_fps(frame: np.array, fps: float):
    fps_text = 'FPS = {:.1f}'.format(fps)
    cv2.putText(frame, fps_text, (24, 40), cv2.FONT_HERSHEY_PLAIN,
                2, (0, 0, 255), 2)
