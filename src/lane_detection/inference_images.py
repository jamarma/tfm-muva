import cv2
import argparse
import glob
import torch

from pathlib import Path
from natsort import natsorted
from torchvision import transforms

import constants
from laneatt.lib.models.laneatt import LaneATT
from utils.drawing import draw_lanes


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((constants.IMG_HEIGHT, constants.IMG_WIDTH)),
    transforms.ToTensor(),
])

test_parameters = {'conf_threshold': constants.CONF_THRESHOLD,
                   'nms_thres': constants.NMS_THRESHOLD,
                   'nms_topk': constants.MAX_LANES}

model_parameters = {'backbone': constants.BACKBONE,
                    'topk_anchors': constants.TOPK_ANCHORS,
                    'anchors_freq_path': constants.ANCHORS_FREQ_PATH}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames', help='Path of input frames', required=True)
    parser.add_argument('--model', help='Path of model checkpoint', required=True)
    args = parser.parse_args()
    return args


def get_frame_paths(path: str) -> list:
    abs_path = str(Path(path).absolute())
    return natsorted(glob.glob(f'{abs_path}/*.png'))


def main():
    args = parse_args()
    frame_paths = get_frame_paths(args.frames)

    model = LaneATT(**model_parameters)
    model.load_state_dict(torch.load(args.model)['model'])
    model = model.to('cuda')
    model.eval()

    for frame_path in frame_paths:
        frame = cv2.imread(frame_path)

        with torch.no_grad():
            frame_tensor = transform(frame).unsqueeze(0).to('cuda')
            output = model(frame_tensor, **test_parameters)
            prediction = model.decode(output, as_lanes=True)

        output_frame = cv2.resize(frame, (constants.IMG_WIDTH, constants.IMG_HEIGHT))
        draw_lanes(output_frame, prediction[0])

        cv2.imshow('Output', output_frame)
        cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
