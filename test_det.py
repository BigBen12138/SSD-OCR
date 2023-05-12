from pathlib import Path
import sys
import os

import numpy as np
import cv2

from ocr.det import YOLO
from utils.general import LOGGER, is_picture, increment_path, scale_boxes
from utils.handle import get_rec_label
from utils.dataloaders import LoadImages
from utils.postprocess import non_max_suppression, xywh2xyxy


def run():
    base: Path
    if getattr(sys, 'frozen', False):
        base = Path(os.path.dirname(sys.executable)).resolve()
    elif __file__:
        base = Path(os.path.dirname(__file__)).resolve()
    w = base / 'models' / 'det.onnx'
    source = base / 'data' / 'test'
    project = f"{(base)}/runs/detect"
    name = "det"
    img_size = 1280
    save_dir = increment_path(Path(project) / name, )
    conf_thres, iou_thres = 0.45, 0.45

    LOGGER.info(f"Base: {base}\n"
                f"Project: {project}\n"
                f"save_dir: {save_dir}\n"
                f"weights: {w}")
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir
    model = YOLO(str(w), img_size, save_dir)
    dataset = LoadImages(source, img_size=img_size, stride=32, auto=False, vid_stride=1)
    seen = 0
    for path, im, im0s, vid_cap, s in dataset:
        LOGGER.info(f"{path}"
                    f"{im.shape}")
        # Inference
        pred = model(im)

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres)

        for i, det in enumerate(pred):
            seen += 1
            im0 = im0s.copy()
            LOGGER.info(f"im0.shape: {im0.shape} \n")
            save_path = str(save_dir / os.path.basename(path))  # im.jpg
            y = det
            boxes = y[..., :4].astype(np.int32)
            scores = y[..., 4]
            imh, imw, _ = im0.shape
            y_diff = (imw - imh) // 2
            position = []
            for box, score in zip(boxes, scores):
                x1, y1, x2, y2 = box
                x1, x2 = int(x1), int(x2)
                y1, y2 = int(y1), int(y2)
                # TODO: 宽高的差值可以被32整除时，才会使用固定值填充
                position.append([x1, y1-y_diff, x2, y2-y_diff, score])
            position.sort(key=lambda x: (x[0] + x[1]))
            LOGGER.info(f"y: {position}")
            for pos in position:
                x1, y1, x2, y2, score = pos
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2
                text = f"{pos[4]:.2f}"
                (w, h), _ = cv2.getTextSize(text, font, font_scale, thickness)
                text_org = (x1, y1 - 5)
                text_bg_org = (x1, y1 - 5)
                text_bg_size = (w, h + 10)
                cv2.rectangle(im0, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cv2.rectangle(im0, text_bg_org, (text_bg_org[0] + text_bg_size[0], text_bg_org[1] + text_bg_size[1]), (0, 255, 0), -1)
                cv2.putText(im0, text, text_org, font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

            LOGGER.info(f"save: {str(save_dir / os.path.basename(path))}")
            cv2.imwrite(str(save_dir / os.path.basename(path)), im0);


if __name__ == '__main__':
    run()
    pass