from typing import *
import pathlib
from functools import cmp_to_key
from typing import List, Any

import onnxruntime
import cv2
import numpy as np

from utils.general import LOGGER
from utils.postprocess import non_max_suppression

conf_thres = 0.25
iou_thres = 0.45


class YOLO:

    def __init__(self, w, img_size, save_crop: str = None, verbose=False):
        providers = ['CPUExecutionProvider']
        session = onnxruntime.InferenceSession(w, providers=providers)
        self.input_name = [x.name for x in session.get_inputs()]
        self.output_name = [x.name for x in session.get_outputs()]
        LOGGER.info(f"input: {self.input_name}, output: {self.output_name}")
        meta = session.get_modelmeta().custom_metadata_map  # metadata
        self.session = session
        if 'stride' in meta:
            stride, names = int(meta['stride']), eval(meta['names'])
            
        self.img_size = img_size
        self.save_crop = pathlib.Path(save_crop) if save_crop else None

        self.verbose = verbose

    def get_input_feed(self, img_tensor):
        input_feed = {}
        for name in self.input_name:
            input_feed[name] = img_tensor
        return input_feed

    def __call__(self, im) -> Any:
        return self.predict(im)

    def predict(self, img: "cv2.Mat"):
        """
           1.cv2读取图像并resize
           2.图像转BGR2RGB和HWC2CHW
           3.图像归一化
           4.图像增加维度
           5.onnx_session 推理
           """
        im = img.copy()
        # 数据增强
        gamma = 0.75
        gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
        gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
        im = cv2.LUT(im, gamma_table)
        LOGGER.info(f"{type(im)}"
                    f"{im.shape}"
                    f"{im.dtype}")
        # im = cv2.resize(im, (self.img_size, self.img_size))
        # LOGGER.info(type(im))
        # im = im[:, :, ::-1].transpose(2, 0, 1)  # BGR2RGB和HWC2CHW
        im = im.astype(dtype=np.float32)
        im /= 255.0
        im = np.expand_dims(im, axis=0)
        # input_feed = self.get_input_feed(img)
        # pred = self.onnx_session.run(None, input_feed)[0]
        pred = self.session.run(self.output_name, {self.session.get_inputs()[0].name: im})
        return pred

    def inference(self, img: Union[str, cv2.Mat]) -> List[Any]:
        file = img
        if isinstance(img, str):
            img = cv2.imread(img)
        pred, or_img = self.predict(img)
        # crop = self.postprocess(output, or_img)
        pred = non_max_suppression(pred, conf_thres, iou_thres, max_det=1000)
        return crop(or_img, pred[0], self.img_size)

    def postprocess(self, output, or_img):
        outbox = filter_box(output, conf_thres, iou_thres)
        return crop(or_img, outbox, self.img_size)


def crop(image: cv2.Mat, box_data: List, img_size: int) -> List[cv2.Mat]:
    boxes = box_data[..., :4].astype(np.int32)
    scores = box_data[..., 4]
    classes = box_data[..., 5].astype(np.int32)

    img_height, img_width = image.shape[0], image.shape[1]
    x_ratio = img_width / img_size
    y_ratio = img_height / img_size
    idx = 1
    crop_img = []
    position = []
    for box, score, cl in zip(boxes, scores, classes):
        x1, y1, x2, y2 = box  # x, y, w, h
        x1, x2 = int(x1 * x_ratio), int(x2 * x_ratio)
        y1, y2 = int(y1 * y_ratio), int(y2 * y_ratio)
        position.append([x1, y1, x2, y2])
    # position = sorted(position, key=cmp_to_key(cmp))
    position = sorted(position, key=lambda x: (x[0] + x[1]))
    # LOGGER.info(position)
    for pos in position:
        x1, y1, x2, y2 = pos
        im = image[y1:y2, x1:x2, ::(-1)]
        crop_img.append(im)
        idx += 1
    return crop_img


# dets:  array [x,6] 6个值分别为x1,y1,x2,y2,score,class
# thresh: 阈值
def nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    # -------------------------------------------------------
    #   计算框的面积
    #   置信度从大到小排序
    # -------------------------------------------------------
    areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    scores = dets[:, 4]
    keep = []
    index = scores.argsort()[::-1]

    while index.size > 0:
        i = index[0]
        keep.append(i)
        # -------------------------------------------------------
        #   计算相交面积
        #   1.相交
        #   2.不相交
        # -------------------------------------------------------
        x11 = np.maximum(x1[i], x1[index[1:]])
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])

        w = np.maximum(0, x22 - x11 + 1)
        h = np.maximum(0, y22 - y11 + 1)

        overlaps = w * h
        # -------------------------------------------------------
        #   计算该框与其它框的IOU，去除掉重复的框，即IOU值大的框
        #   IOU小于thresh的框保留下来
        # -------------------------------------------------------
        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)
        idx = np.where(ious <= thresh)[0]
        index = index[idx + 1]
    return keep


def xywh2xyxy(x):
    # [x, y, w, h] to [x1, y1, x2, y2]
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


# def nms(dets, thresh):
#     """
#     :param boxes:  (Tensor[N, 4])): are expected to be in ``(x1, y1, x2, y2)
#     :param scores: (Tensor[N]): scores for each one of the boxes
#     :param iou_thres: discards all overlapping boxes with IoU > iou_threshold
#     :return:keep (Tensor): int64 tensor with the indices
#             of the elements that have been kept
#             by NMS, sorted in decreasing order of scores
#     """
#     # 按conf从大到小排序
#     scores=dets[:, 4]
#     boxes=dets[:,:4]
#     B = scores.argsort()[::-1]
#     keep = []
#     while B.size > 0:
#         # 取出置信度最高的
#         index = B[0]
#         keep.append(index)
#         if B.size == 1: break
#         # 计算iou,根据需求可选择GIOU,DIOU,CIOU
#         iou = bbox_iou(boxes[index, :], boxes[B[1:], :], True, True, False,False)
#         # 找到符合阈值的下标
#         inds = np.where(iou <= thresh)[0]
#         B = B[inds + 1]
#     return keep

# def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-9):
#     # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
#     box2 = box2.T

#     # Get the coordinates of bounding boxes
#     if x1y1x2y2:  # x1, y1, x2, y2 = box1
#         b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
#         b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
#     else:  # transform from xywh to xyxy
#         b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
#         b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
#         b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
#         b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

#     # Intersection area
#     inter = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
#             (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)

#     # Union Area
#     w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
#     w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
#     union = w1 * h1 + w2 * h2 - inter + eps

#     iou = inter / union
#     if GIoU or DIoU or CIoU:
#         cw = np.maximum(b1_x2, b2_x2) - np.minimum(b1_x1, b2_x1)  # convex (smallest enclosing box) width
#         ch = np.maximum(b1_y2, b2_y2) - np.minimum(b1_y1, b2_y1)  # convex height
#         if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
#             c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
#             rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
#                     (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
#             if DIoU:
#                 return iou - rho2 / c2  # DIoU
#             elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
#                 v = (4 / np.math.pi ** 2) * np.pow(np.atan(w2 / h2) - np.atan(w1 / h1), 2)
#                 with torch.no_grad():
#                     alpha = v / ((1 + eps) - iou + v)
#                 return iou - (rho2 / c2 + v * alpha)  # CIoU
#         else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
#             c_area = cw * ch + eps  # convex area
#             return iou - (c_area - union) / c_area  # GIoU
#     else:
#         return iou  # IoU

def filter_box(org_box: "np.ndarray", conf_thres, iou_thres):  # 过滤掉无用的框
    """
    org_box: an array per image [n, 6]
    """
    # -------------------------------------------------------
    #   删除为1的维度
    #   删除置信度小于conf_thres的BOX
    # -------------------------------------------------------
    # LOGGER.info(org_box.shape)
    org_box = np.squeeze(org_box)
    # LOGGER.info(org_box.shape)
    conf = org_box[..., 4] > conf_thres
    box = org_box[conf == True]
    # LOGGER.info(f"box.shape: {box.shape}")
    # -------------------------------------------------------
    #   通过argmax获取置信度最大的类别
    # -------------------------------------------------------
    cls_cinf = box[..., 5:]
    cls = []
    for i in range(len(cls_cinf)):
        cls.append(int(np.argmax(cls_cinf[i])))
    # LOGGER.info(f"cls: {cls}")
    # -------------------------------------------------------
    #   分别对每个类别进行过滤
    #   1.将第6列元素替换为类别下标
    #   2.xywh2xyxy 坐标转换
    #   3.经过非极大抑制后输出的BOX下标
    #   4.利用下标取出非极大抑制后的BOX
    # -------------------------------------------------------
    output = []

    curr_cls = 0
    curr_cls_box = []
    curr_out_box = []
    for j in range(len(cls)):
        box[j][5] = curr_cls
        curr_cls_box.append(box[j][:6])
    curr_cls_box = np.array(curr_cls_box)
    # curr_cls_box_old = np.copy(curr_cls_box)
    curr_cls_box = xywh2xyxy(curr_cls_box)
    curr_out_box = nms(curr_cls_box, iou_thres)
    LOGGER.info(f"curr_out_box: {curr_out_box}")
    for k in curr_out_box:
        output.append(curr_cls_box[k])
    output = np.array(output)
    LOGGER.info(output.shape)
    return output


def draw(image, box_data, crop=False):
    # -------------------------------------------------------
    #   取整，方便画框
    # -------------------------------------------------------
    boxes = box_data[..., :4].astype(np.int32)
    scores = box_data[..., 4]
    classes = box_data[..., 5].astype(np.int32)

    img_height_o = image.shape[0]
    img_width_o = image.shape[1]
    x_ratio = img_width_o / 1280
    y_ratio = img_height_o / 1280
    idx = 1
    crop_img = []
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = box  # x, y, w, h
        # print('class: {}, score: {}'.format(0, score))
        # print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(top, left, right, bottom))

        x1 = int(top * x_ratio)
        x2 = int(right * x_ratio)
        y1 = int(left * y_ratio)
        y2 = int(bottom * y_ratio)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        im = image[y1:y2, x1:x2, ::(-1)]
        crop_img.append(im)
        idx += 1
    return crop_img


if __name__ == "__main__":
    onnx_path = '/home/xla/code/python/SSD-OCR/ocr/models/det.onnx'
    model = YOLO(onnx_path, 1280)
    output, or_img = model.inference('/home/xla/code/python/SSD-OCR/img/omron-1.png')
    cv2.imwrite('res.jpg', or_img)
    print(type(or_img))
