from typing import *
import pathlib
import os
import imghdr


import cv2

from ocr.det import YOLO as Det
from ocr.rec import Rec
from utils.general import LOGGER

class OCR:
    def __init__(self, det_model: str, rec_model: str) -> None:
        self.det_model = Det(det_model, 1280)
        self.rec_model = Rec(rec_model, '/home/xla/code/python/SSD-OCR/models/en_dict.txt')
        self.LOGGER = LOGGER
        pass

    def detect(self, img):
        crop_imgs = self.det_model.inference(img)
        return crop_imgs
    
    def recognize(self, img):
        return self.rec_model.inference(img)
        
    def predict(self, img: str, save_crop=None):
        self.LOGGER.info(img)
        crop_imgs = self.detect(img)
        if save_crop:
            save_crop = os.path.abspath(save_crop)
            if not os.path.exists(save_crop):
                os.mkdir(save_crop, 0o755)
            img_type = imghdr.what(img)
            for i, crop in enumerate(crop_imgs):
                cv2.imwrite(f"{os.path.join(save_crop, os.path.basename(img).split('.')[0])}-{i}.{img_type}", crop)
        predict_res = self.recognize(crop_imgs)
        return list(map(int, [d[0] for d in predict_res]))

    def inference(self, img_path: str, save_crop: str):
        img = cv2.imread(img_path)
        return self.predict(img)

    def score_file(self, x, y):
        ...

    def score_folder(self, x, y):
        ...

    def score(self, img, expect, save_crop=None):
        self.LOGGER.info(img)
        crop_imgs = self.detect(img)
        self.LOGGER.info(len(crop_imgs))
        predict_res = self.recognize(crop_imgs)
        predict = list(map(int, [d[0] for d in predict_res]))
        total = wrong = 0
        for i, (x, y, crop) in enumerate(zip(predict, expect, crop_imgs)):
            if x != y:
                if save_crop:
                    save_crop = os.path.abspath(save_crop)
                    if not os.path.exists(save_crop):
                        os.mkdir(save_crop, 0o755)
                    img_type = imghdr.what(img)
                    cv2.imwrite(f"{os.path.join(save_crop, os.path.basename(img).split('.')[0])}-{i}.{img_type}", crop)
                wrong += 1
                
            total += 1
        return total, wrong
            
        
        
if __name__ == '__main__':
    ocr = OCR(det_model='/home/xla/code/python/SSD-OCR/models/det.onnx', rec_model='/home/xla/code/python/SSD-OCR/models/rec.onnx')
    # ocr.inference('/home/xla/code/python/SSD-OCR/img/欧姆龙HEM-713(没背光)-22.png')
    ocr.inference_batch("")
    pass