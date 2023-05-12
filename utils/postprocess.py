from typing import List
import time
from pathlib import Path

import numpy as np

from utils.general import LOGGER

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def nms(boxes, scores, iou_thres):
    """
    :param boxes: np.array([[x1, y1, x2, y2], [x1, y1, x2, y2], ...]), N个检测框
    :param scores: np.array([score1, score2, ...]), N个检测框对应的分数
    :param iou_thres: float, IoU阈值
    :return: np.array([keep1, keep2, ...]), 保留的检测框的下标
    """

    B = scores.argsort()[::-1]
    keep = []
    while B.size > 0:
        # 取出置信度最高的
        index = B[0]
        keep.append(index)
        if B.size == 1: break
        # 计算iou,根据需求可选择GIOU,DIOU,CIOU
        iou = bbox_iou(boxes[index, :], boxes[B[1:], :], True, True, False,False)
        # 找到符合阈值的下标
        inds = np.where(iou <= iou_thres)[0]
        B = B[inds + 1]
    return keep

def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-9):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
            (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if GIoU or DIoU or CIoU:
        cw = np.maximum(b1_x2, b2_x2) - np.minimum(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = np.maximum(b1_y2, b2_y2) - np.minimum(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / np.math.pi ** 2) * np.pow(np.atan(w2 / h2) - np.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / ((1 + eps) - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU


def box_iou(box1, box2, eps=1e-7):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (ndarray[N, 4])
        box2 (ndarray[M, 4])
    Returns:
        iou (ndarray[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    
    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clip(0).prod(2)
    a1, a2 = np.expand_dims(box1[:, :2], axis=1), np.expand_dims(box1[:, 2:], axis=1)
    b1, b2 = np.expand_dims(box2[:, :2], axis=0), np.expand_dims(box2[:, 2:], axis=0)
    inter = np.clip(np.min(a2, b2) - np.max(a1, b1), 0, None).prod(axis=2)

    # IoU = inter / (area1 + area2 - inter)
    area1 = np.prod(a2 - a1, axis=2)
    area2 = np.prod(b2 - b1, axis=2)
    return inter / (area1[:, None] + area2 - inter + eps)

def clip_boxes(boxes, shape):
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2

def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    # Rescale boxes (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[..., [0, 2]] -= pad[0]  # x padding
    boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes

def non_max_suppression(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nm=0,  # number of masks
):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping detections

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
    if isinstance(prediction, (list, tuple)):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    # device = prediction.device
    # mps = 'mps' in device.type  # Apple MPS
    # if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
    #     prediction = prediction.cpu()
    LOGGER.info(prediction.shape)
    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - nm - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.5 + 0.05 * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    mi = 5 + nc  # mask start index
    output = [np.zeros((0, 6 + nm))] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = np.zeros((len(lb), nc + nm + 5))
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = np.concatenate((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box/Mask
        box = xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        mask = x[:, mi:]  # zero columns if no masks

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
            x = np.concatenate((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            # LOGGER.info(type(x))
            # conf, j = x[:, 5:mi].max(1, keepdim=True) 
            # to numpy:
            conf = np.amax(x[:, 5:mi], axis=1, keepdims=True)
            j = np.argmax(x[:, 5:mi], axis=1, keepdims=True)
            # x = np.concatenate((box, conf, j.astype(np.float32), mask), 1)[conf.view(-1) > conf_thres]
            # to numpy:
            conf_mask = conf.reshape(-1) > conf_thres
            x = np.concatenate((box, conf.reshape(-1, 1), j.astype(np.float32), mask), axis=1)[conf_mask]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == np.array(classes)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        # 取前 max_nms 个 box 进行 NMS
        sorted_indices = np.argsort(-x[:, 4])
        x = x[sorted_indices][:max_nms]
        # x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        # TODO: implement nms with numpy
        i = nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[:, :4] = np.dot(x[:, :4].astype(np.float32), weights.T.astype(np.float32)) / weights.sum(1, keepdims=True)
            # x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        # if mps:
        #     output[xi] = output[xi].to(device)
        if (time.time() - t) > time_limit:
            LOGGER.warning(f'WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded')
            break  # time limit exceeded
    return output

# TODO: process
# def process_prediction(pred: List):
#     assert isinstance(pred, List), f"the type of pred should be List, but {type(pred)}"

#     for i, det in enumerate(pred):  # per image
#         seen += 1
#         p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

#         p = Path(p)  # to Path
#         save_path = str(save_dir / p.name)  # im.jpg
#         txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
#         s += '%gx%g ' % im.shape[2:]  # print string
#         gn = np.array(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
#         imc = im0.copy() if save_crop else im0  # for save_crop
#         # annotator = Annotator(im0, line_width=line_thickness, example=str(names))
#         if len(det):
#             # Rescale boxes from img_size to im0 size
#             det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

#             # Print results
#             for c in det[:, 5].unique():
#                 n = (det[:, 5] == c).sum()  # detections per class
#                 s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

#             # Write results
#             for *xyxy, conf, cls in reversed(det):
#                 if save_img or save_crop or view_img:  # Add bbox to image
#                     c = int(cls)  # integer class
#                     label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
#                     annotator.box_label(xyxy, label, color=colors(c, True))
#                 if save_crop:
#                     save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

        # Stream results
        # im0 = annotator.result()
    
        # Save results (image with detections)
        # if save_img:
        #     if dataset.mode == 'image':
        #         cv2.imwrite(save_path, im0)
        #     else:  # 'video' or 'stream'
        #         if vid_path[i] != save_path:  # new video
        #             vid_path[i] = save_path
        #             if isinstance(vid_writer[i], cv2.VideoWriter):
        #                 vid_writer[i].release()  # release previous video writer
        #             if vid_cap:  # video
        #                 fps = vid_cap.get(cv2.CAP_PROP_FPS)
        #                 w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        #                 h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        #             else:  # stream
        #                 fps, w, h = 30, im0.shape[1], im0.shape[0]
        #             save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
        #             vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        #         vid_writer[i].write(im0)