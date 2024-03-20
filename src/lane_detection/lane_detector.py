import torch
import numpy as np
from torchvision import transforms

import constants
from laneatt.lib.models.laneatt import LaneATT
from laneatt.lib.lane import Lane


class LaneDetector:
    def __init__(self):
        self.model = self.__init_model()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((constants.IMG_HEIGHT, constants.IMG_WIDTH)),
            transforms.ToTensor(),
        ])

    def detect(self, frame: np.array) -> list[Lane]:
        with torch.no_grad():
            frame_tensor = self.transform(frame).unsqueeze(0).to('cuda')
            output = self.model(frame_tensor,
                                conf_threshold=constants.CONF_THRESHOLD,
                                nms_thres=constants.NMS_THRESHOLD,
                                nms_topk=constants.MAX_LANES)
            prediction = self.model.decode(output, as_lanes=True)
            lanes = prediction[0]
            return lanes

    @staticmethod
    def __lanes_scaling(lanes: list[Lane], frame: np.array) -> list[Lane]:
        if lanes is not None:
            for lane in lanes:
                lane.points[:, 0] *= frame.shape[1]
                lane.points[:, 1] *= frame.shape[0]
                lane.points = lane.points.round().astype(int)
        return lanes

    @staticmethod
    def __init_model() -> LaneATT:
        model = LaneATT(backbone=constants.BACKBONE,
                        topk_anchors=constants.TOPK_ANCHORS,
                        anchors_freq_path=constants.ANCHORS_FREQ_PATH)
        model.load_state_dict(torch.load(constants.MODEL_PATH)['model'])
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        model.eval()
        return model
