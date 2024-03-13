import cv2
import argparse
import torch

from torchvision import transforms

import constants
from laneatt.lib.models.laneatt import LaneATT
from utils.drawing import draw_lanes, draw_fps
from utils.fps import FPS


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
    parser.add_argument('--video', help='Path of input video', required=True)
    parser.add_argument('--model', help='Path of model checkpoint', required=True)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    model = LaneATT(**model_parameters)
    model.load_state_dict(torch.load(args.model)['model'])
    model = model.to('cuda')
    model.eval()

    cap = cv2.VideoCapture(args.video)
    fps = FPS().start()

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        with torch.no_grad():
            frame_tensor = transform(frame).unsqueeze(0).to('cuda')
            output = model(frame_tensor, **test_parameters)
            prediction = model.decode(output, as_lanes=True)

        output_frame = cv2.resize(frame, (constants.IMG_WIDTH, constants.IMG_HEIGHT))
        draw_lanes(output_frame, prediction[0])

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
