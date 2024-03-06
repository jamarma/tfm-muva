import cv2
import argparse
import glob
import torch

from pathlib import Path
from natsort import natsorted
from torchvision import transforms

import sys
sys.path.append('laneatt/')

from lib.models.laneatt import LaneATT


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((360, 640)),
    transforms.ToTensor(),
])

test_parameters = {'conf_threshold': 0.5, 'nms_thres': 50., 'nms_topk': 4}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames', help='Path of input frames', required=True)
    parser.add_argument('--model', help='Path of model checkpoint', required=True)
    args = parser.parse_args()
    return args


def get_frame_paths(path):
    abs_path = str(Path(path).absolute())
    return natsorted(glob.glob(f'{abs_path}/*.png'))


def draw_prediction(frame, prediction):
    if prediction is not None:
        for pred in prediction:
            for lane in pred:
                points = lane.points
                points[:, 0] *= frame.shape[1]
                points[:, 1] *= frame.shape[0]
                points = points.round().astype(int)
                for curr_p, next_p in zip(points[:-1], points[1:]):
                    cv2.line(frame, tuple(curr_p), tuple(next_p),
                             color=(0, 255, 0), thickness=3)


def main():
    args = parse_args()
    frame_paths = get_frame_paths(args.frames)

    model = LaneATT(topk_anchors=1000, anchors_freq_path='laneatt/.cache/culane_anchors_freq.pt')
    model.load_state_dict(torch.load(args.model)['model'])
    model = model.to('cuda')
    model.eval()

    with torch.no_grad():
        for frame_path in frame_paths:
            frame = cv2.imread(frame_path)

            frame_tensor = transform(frame).unsqueeze(0).to('cuda')
            output = model(frame_tensor, **test_parameters)
            prediction = model.decode(output, as_lanes=True)

            output_frame = cv2.resize(frame, (640, 360))
            draw_prediction(output_frame, prediction)

            cv2.imshow('Output', output_frame)
            cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
